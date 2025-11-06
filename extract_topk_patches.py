import os
import re
import math
import glob
import h5py
import yaml
import argparse
import numpy as np
from PIL import Image, ImageDraw
import openslide

# ---------- IO: load scores/coords ----------

def load_scores_coords(h5_path):
    with h5py.File(h5_path, "r") as f:
        scores = f["attention_scores"][:].astype(np.float32).reshape(-1)
        coords = f["coords"][:]
    # coords may be float or int depending on producer; cast to int
    coords = coords.astype(np.int64)
    m = np.isfinite(scores)
    return scores[m], coords[m]

# ---------- indexing helpers ----------

def topk_indices(scores, k):
    k = min(int(k), len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(-scores, kth=k-1)[:k]
    return idx[np.argsort(-scores[idx])]

def topk_with_min_dist(scores, coords, k, min_dist):
    # Greedy NMS-like selection in patch_level pixels
    order = np.argsort(-scores)
    selected = []
    for i in order:
        if len(selected) >= k:
            break
        if min_dist <= 0:
            selected.append(i)
            continue
        if not selected:
            selected.append(i)
            continue
        d = np.sqrt(((coords[selected] - coords[i]) ** 2).sum(axis=1))
        if np.all(d >= min_dist):
            selected.append(i)
    return np.array(selected, dtype=np.int64)

# ---------- slide path inference ----------

KNOWN_EXTS = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".svslide", ".bif"]

def guess_slide_path(wsi_dir, slide_id):
    # exact match with known exts
    for ext in KNOWN_EXTS:
        p = os.path.join(wsi_dir, slide_id + ext)
        if os.path.isfile(p):
            return p
    # fallback: glob any file that starts with slide_id
    cands = glob.glob(os.path.join(wsi_dir, slide_id + ".*"))
    # prefer known exts if present
    cands_sorted = sorted(cands, key=lambda x: (0 if os.path.splitext(x)[1].lower() in KNOWN_EXTS else 1, x))
    return cands_sorted[0] if cands_sorted else None

# ---------- coords <-> level transform ----------

def level0_xy_from_patch_coord(coord_at_level, slide, patch_level):
    ds = float(slide.level_downsamples[patch_level])
    x0 = int(round(coord_at_level[0] * ds))
    y0 = int(round(coord_at_level[1] * ds))
    return (x0, y0)

def read_patch(slide, xy_lvl0, patch_level, patch_size):
    # read_region coords at level-0
    return slide.read_region(xy_lvl0, patch_level, (patch_size, patch_size)).convert("RGB")

# ---------- visualization ----------

def save_grid(patches, labels, out_path, ncols=5, pad=4, bg=(255, 255, 255)):
    if len(patches) == 0:
        return
    w, h = patches[0].size
    ncols = max(1, min(ncols, len(patches)))
    nrows = math.ceil(len(patches) / ncols)
    W = ncols * w + (ncols + 1) * pad
    H = nrows * h + (nrows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    for i, (im, txt) in enumerate(zip(patches, labels)):
        r, c = divmod(i, ncols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + 4), txt, fill=(0, 0, 0))
    canvas.save(out_path, quality=100)

# ---------- config helpers ----------

def try_load_patch_params_from_config(h5_dir):
    """
    Try to load patch_level/patch_size from config.yaml at:
        <.../HEATMAP_OUTPUT_XXX>/config.yaml
    where h5_dir is usually .../<HEATMAP_OUTPUT_XXX>/<class>/<slide_id>
    """
    root = os.path.dirname(h5_dir)            # .../<HEATMAP_OUTPUT_XXX>/<class>
    exp_root = os.path.dirname(root)         # .../<HEATMAP_OUTPUT_XXX>
    cfg = os.path.join(exp_root, "config.yaml")
    if not os.path.isfile(cfg):
        return None, None
    with open(cfg, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    try:
        patch_level = int(d["patching_arguments"]["patch_level"])
        patch_size = int(d["patching_arguments"]["patch_size"])
        return patch_level, patch_size
    except Exception:
        return None, None

def infer_slide_id_from_h5(h5_path):
    # Use parent directory name as slide_id: .../<class>/<slide_id>/<file.h5>
    return os.path.basename(os.path.dirname(h5_path))

def default_out_dir_from_h5(h5_path, source_tag="fine"):
    # .../<HEATMAP_OUTPUT_XXX>/<class>/<slide_id>
    slide_id = infer_slide_id_from_h5(h5_path)
    slide_dir = os.path.dirname(h5_path)
    class_dir = os.path.dirname(slide_dir)
    exp_dir = os.path.dirname(class_dir)
    exp_code = os.path.basename(exp_dir)
    class_name = os.path.basename(class_dir)
    base = os.path.join("heatmaps", "patches", exp_code, class_name, slide_id, f"topk_{source_tag}")
    return base

# ---------- main extraction ----------

def extract_topk(
    slide_path,
    h5_path,
    out_dir=None,
    k=15,
    patch_level=None,
    patch_size=None,
    make_grid=True,
    grid_cols=5,
    min_dist=0
):
    """
    Extract top-k patches by attention score.
    min_dist: minimum distance in patch-level pixels between chosen patches (0 to disable).
    """
    if out_dir is None:
        source_tag = "blockmap" if h5_path.lower().endswith("blockmap.h5") else "fine"
        out_dir = default_out_dir_from_h5(h5_path, source_tag=source_tag)
    os.makedirs(out_dir, exist_ok=True)

    # auto patch params if missing
    if patch_level is None or patch_size is None:
        auto_pl, auto_ps = try_load_patch_params_from_config(os.path.dirname(h5_path))
        patch_level = auto_pl if patch_level is None else patch_level
        patch_size = auto_ps if patch_size is None else patch_size
    if patch_level is None or patch_size is None:
        raise ValueError("patch_level/patch_size 未提供且未能从 config.yaml 自动读取。请通过参数指定。")

    scores, coords = load_scores_coords(h5_path)

    # select indices
    if min_dist and min_dist > 0:
        sel = topk_with_min_dist(scores, coords, k, min_dist=min_dist)
    else:
        sel = topk_indices(scores, k)

    slide = openslide.OpenSlide(slide_path)

    patches, labels = [], []
    meta_lines = ["rank,score,x_level,y_level,patch_level,patch_size,h5"]
    for rank, i in enumerate(sel, 1):
        xy0 = level0_xy_from_patch_coord(coords[i], slide, patch_level)
        patch = read_patch(slide, xy0, patch_level, patch_size)
        score = float(scores[i])
        fname = f"rank_{rank:02d}_score_{score:.6f}.jpg"
        patch.save(os.path.join(out_dir, fname), quality=100)
        patches.append(patch)
        labels.append(f"{rank}:{score:.4f}")
        meta_lines.append(f"{rank},{score:.6f},{coords[i,0]},{coords[i,1]},{patch_level},{patch_size},{h5_path}")

    if make_grid:
        save_grid(patches, labels, os.path.join(out_dir, f"top{len(patches)}_grid.jpg"), ncols=grid_cols)

    # save csv
    with open(os.path.join(out_dir, "topk.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    slide.close()
    print(f"Saved {len(patches)} patches to: {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Extract top-k patches by attention score")
    ap.add_argument("--h5", required=True, help="Path to fine-attention h5 (..._roi_*.h5) or blockmap.h5")
    ap.add_argument("--slide", default=None, help="Path to WSI file (if omitted, use --wsi-dir + --slide-id)")
    ap.add_argument("--wsi-dir", default=None, help="Directory containing WSI files")
    ap.add_argument("--slide-id", default=None, help="Slide ID (defaults to parent folder name of h5)")
    ap.add_argument("--k", type=int, default=15, help="Top-k per slide")
    ap.add_argument("--patch-level", type=int, default=1, help="Patch level; auto from config.yaml if omitted")
    ap.add_argument("--patch-size", type=int, default=256, help="Patch size; auto from config.yaml if omitted")
    ap.add_argument("--min-dist", type=int, default=0, help="Min distance between patches at patch_level; 0 to disable")
    ap.add_argument("--grid-cols", type=int, default=5, help="Columns in grid image")
    ap.add_argument("--no-grid", action="store_true", help="Disable grid image")
    ap.add_argument("--out-dir", default=None, help="Custom output dir")
    args = ap.parse_args()


    # 推断 slide 路径
    slide_path = args.slide
    slide_id = args.slide_id or infer_slide_id_from_h5(args.h5)
    if slide_path is None:
        if args.wsi_dir is None:
            raise ValueError("未提供 --slide，且 --wsi-dir 为空，无法定位 WSI 路径。")
        
        # 如果 --wsi-dir 实际上是一个文件，直接当作 slide 路径
        if os.path.isfile(args.wsi_dir):
            slide_path = args.wsi_dir
        else:
            slide_path = guess_slide_path(args.wsi_dir, slide_id)

    if slide_path is None:
        raise FileNotFoundError(f"在 {args.wsi_dir} 下未找到 {slide_id} 的 WSI 文件。请检查扩展名和路径。")

    extract_topk(
        slide_path=slide_path,
        h5_path=args.h5,
        out_dir=args.out_dir,
        k=args.k,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        make_grid=not args.no_grid,
        grid_cols=args.grid_cols,
        min_dist=args.min_dist
    )

if __name__ == "__main__":
    main()
