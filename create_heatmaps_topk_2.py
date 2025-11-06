# create_heatmaps_topk_2.py
from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models import get_encoder
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm
from PIL import Image
import gc  # 垃圾回收

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# embed_dim 映射（按 save_exp_code）
task_to_embed_dim = {
    "camelyon16_plip": 512,
    "camelyon16_R50": 1024,
    "TCGA_BC_PLIP": 512,
    "TCGA_BC_R50": 1024,
    "TCGA_NSCLC_PLIP": 512,
    "TCGA_NSCLC_R50": 1024,
    "HEATMAP_OUTPUT_camelyon16_plip": 512,
    "HEATMAP_OUTPUT_BRCA": 1024,
    "HEATMAP_OUTPUT_NSCLC": 1024
}

# 线性映射缓存
linear_adjust_cache = {}

def get_linear_adjust_layer(current_dim, target_dim, device):
    if (current_dim, target_dim) in linear_adjust_cache:
        return linear_adjust_cache[(current_dim, target_dim)]
    print(f"Creating new linear layer for adjusting features from {current_dim} to {target_dim}")
    linear_adjust = nn.Linear(current_dim, target_dim).to(device)
    nn.init.xavier_uniform_(linear_adjust.weight)
    if linear_adjust.bias is not None:
        nn.init.zeros_(linear_adjust.bias)
    linear_adjust_cache[(current_dim, target_dim)] = linear_adjust
    return linear_adjust

def adjust_features(features, target_dim):
    current_dim = features.shape[-1]
    print(f"Adjusting features from {current_dim} to {target_dim}")
    if current_dim != target_dim:
        linear_adjust = get_linear_adjust_layer(current_dim, target_dim, features.device)
        features = linear_adjust(features)
        print(f"Adjusted features shape: {features.shape}")
    else:
        print("No adjustment needed for features.")
    return features

def infer_single_slide(model, features, label, reverse_label_dict, k=1, embed_dim=1024):
    print(f"Features shape before any adjustment: {features.shape}")
    if features.dim() == 2:
        features = features.unsqueeze(0)
        print(f"Added batch dimension. New features shape: {features.shape}")
    elif features.dim() != 3:
        raise ValueError(f"Unexpected feature dimensions: {features.dim()}")

    features = adjust_features(features, target_dim=embed_dim)
    features = features.to(device)
    print(f"Features shape after adjustment and moving to device: {features.shape}")

    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()
            if isinstance(model, CLAM_MB):
                A = A[Y_hat]
            A = A.view(-1, 1).cpu().numpy()
        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(
            reverse_label_dict[Y_hat], 
            label, 
            ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]
        ))
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            try:
                val = dtype(val)
                if isinstance(val, str):
                    if len(val) > 0:
                        params[key] = val
                elif not np.isnan(val):
                    params[key] = val
                else:
                    pdb.set_trace()
            except:
                pdb.set_trace()
    return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

# =========================
# Top-k 导出 & 分位数向量化
# =========================

def _safe_load_scores_coords(h5_path):
    """稳健读取 h5 中的 attention_scores 与 coords，统一形状，去除 NaN/Inf。"""
    with h5py.File(h5_path, "r") as f:
        scores = np.asarray(f["attention_scores"][()]).astype(np.float32)
        coords = np.asarray(f["coords"][()])
    scores = scores.reshape(-1)
    coords = coords.reshape(-1, 2).astype(np.int64)
    N = min(len(scores), coords.shape[0])
    if len(scores) != coords.shape[0]:
        print(f"[WARN] scores (N={len(scores)}) 与 coords (N={coords.shape[0]}) 长度不一致，截断为 N={N}")
    scores = scores[:N]
    coords = coords[:N]
    bad = ~np.isfinite(scores)
    if bad.any():
        print(f"[WARN] attention_scores 中存在 {bad.sum()} 个非有限值，已填充为 -inf")
        scores[bad] = -np.inf
    return scores, coords

def _topk_indices(scores, k):
    k = int(min(max(k, 0), len(scores)))
    if k == 0 or len(scores) == 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]

def _topk_with_min_dist(scores, coords, k, min_dist):
    order = np.argsort(-scores)
    sel = []
    for i in order:
        if len(sel) >= k:
            break
        if not sel or min_dist <= 0:
            sel.append(i)
            continue
        d = np.sqrt(((coords[sel] - coords[i]) ** 2).sum(axis=1))
        if np.all(d >= min_dist):
            sel.append(i)
    return np.array(sel, dtype=np.int64)

def _read_region_rgb_whitebg(wsi, x0, y0, patch_level, patch_size):
    """读取 RGBA 并合成到白底，避免透明变黑。"""
    rgba = wsi.read_region((x0, y0), patch_level, (patch_size, patch_size)).convert("RGBA")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    alpha = rgba.split()[-1]
    white.paste(rgba.convert("RGB"), mask=alpha)
    return white

def _infer_coord_mode(coords, wsi, patch_level, patch_size, sample_n=64):
    """自动判断 coords 是 level-0 还是 patch_level。"""
    W0, H0 = wsi.level_dimensions[0]
    ds = float(wsi.level_downsamples[patch_level])
    step_w0 = int(round(patch_size * ds))
    step_h0 = int(round(patch_size * ds))
    idxs = np.linspace(0, len(coords)-1, num=min(sample_n, len(coords)), dtype=int)
    def valid(x0, y0):
        return (x0 >= 0 and y0 >= 0 and x0 + step_w0 <= W0 and y0 + step_h0 <= H0)
    cnt_patch = 0
    cnt_lvl0 = 0
    for i in idxs:
        xL, yL = coords[i]
        x0p = int(round(xL * ds))
        y0p = int(round(yL * ds))
        if valid(x0p, y0p):
            cnt_patch += 1
        x0l = int(xL)
        y0l = int(yL)
        if valid(x0l, y0l):
            cnt_lvl0 += 1
    mode = "patch_level" if cnt_patch >= cnt_lvl0 else "level0"
    usable = max(cnt_patch, cnt_lvl0) / max(1, len(idxs))
    print(f"[TopK] coord_mode auto={mode} (usable={usable:.2f}, patch={cnt_patch}, level0={cnt_lvl0}, sample={len(idxs)})")
    return mode, ds

def export_topk_patches_from_h5(wsi_object, h5_path, out_dir, k, patch_level, patch_size, min_dist=0):
    """从 h5（细粒度或 blockmap）导出 top-k 补丁；自动判断坐标系；白底合成。"""
    os.makedirs(out_dir, exist_ok=True)
    scores, coords = _safe_load_scores_coords(h5_path)
    if len(scores) == 0:
        print(f"[WARN] {h5_path} 中未找到有效 attention_scores，跳过导出。")
        return
    sel = _topk_with_min_dist(scores, coords, k, min_dist=min_dist) if (min_dist and min_dist > 0) else _topk_indices(scores, k)
    wsi = wsi_object.wsi
    coord_mode, ds = _infer_coord_mode(coords, wsi, patch_level, patch_size)

    meta_lines = ["rank,score,x_level,y_level,patch_level,patch_size,coord_mode,ds,h5"]
    for rank, i in enumerate(sel, 1):
        xL, yL = coords[i]
        if coord_mode == "patch_level":
            x0 = int(round(xL * ds))
            y0 = int(round(yL * ds))
        else:
            x0 = int(xL)
            y0 = int(yL)
        patch_rgb = _read_region_rgb_whitebg(wsi, x0, y0, patch_level, patch_size)
        score_val = float(scores[i])
        fname = f"rank_{rank:02d}_score_{score_val:.6f}.jpg"
        patch_rgb.save(os.path.join(out_dir, fname), quality=100)
        meta_lines.append(f"{rank},{score_val:.6f},{xL},{yL},{patch_level},{patch_size},{coord_mode},{ds},{h5_path}")

    with open(os.path.join(out_dir, "topk.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    print(f"[TopK] 已保存 {len(sel)} 个补丁至: {out_dir}")

def percentiles_from_ref(scores, ref_scores):
    """
    向量化分位数转换（近似 SciPy percentileofscore(kind='rank')）
    p = ((left + right) / 2) / N * 100
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref_scores, dtype=np.float32).reshape(-1)
    if ref.size == 0:
        return np.zeros_like(scores, dtype=np.float32)
    ref_sorted = np.sort(ref)
    N = float(ref_sorted.size)
    left = np.searchsorted(ref_sorted, scores, side='left')
    right = np.searchsorted(ref_sorted, scores, side='right')
    pct = ((left + right) * 0.5) / N * 100.0
    return pct.astype(np.float32)

# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--save_exp_code', type=str, default=None, help='experiment code')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join('heatmaps/configs', args.config_file)
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))
    decision = input('Continue? Y/N ')
    if decision not in ['Y', 'y', 'Yes', 'yes']:
        exit()

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    # 修正：先在字典上添加 n_classes，再转 Namespace
    model_args_dict = dict(args['model_arguments'])
    model_args_dict['n_classes'] = args['exp_arguments']['n_classes']
    model_args = argparse.Namespace(**model_args_dict)
    encoder_args = argparse.Namespace(**args['encoder_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    if not hasattr(heatmap_args, 'convert_to_percentiles'):
        heatmap_args.convert_to_percentiles = True

    patch_size = tuple([patch_args.patch_size for _ in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
        patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))
    
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]
        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]
        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]
        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    model = model.to(device)
    print('Done!')

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    for i in tqdm(range(len(process_stack))):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')
        grouping = reverse_label_dict[label] if not isinstance(label, str) else label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i,'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')
        
        # 参数汇入
        seg_params = load_params(process_stack.loc[i], dict(def_seg_params))
        filter_params = load_params(process_stack.loc[i], dict(def_filter_params))
        vis_params = load_params(process_stack.loc[i], dict(def_vis_params))

        keep_ids = str(seg_params['keep_ids'])
        seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int) if len(keep_ids) > 0 and keep_ids != 'none' else []

        exclude_ids = str(seg_params['exclude_ids'])
        seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int) if len(exclude_ids) > 0 and exclude_ids != 'none' else []

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.h5')
        mask_path = os.path.join(r_slide_save_dir, f'{slide_id}_mask.jpg')
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')

        # 1) 特征
        if not os.path.isfile(h5_path):
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)
            print(f"Processing with embed_dim: {embed_dim}")
            _, _, wsi_object = compute_from_patches(
                wsi_object=wsi_object, 
                model=model, 
                feature_extractor=feature_extractor, 
                img_transforms=img_transforms,
                batch_size=exp_args.batch_size, 
                embed_dim=embed_dim,
                top_left=top_left, 
                bot_right=bot_right, 
                patch_size=patch_size, 
                step_size=step_size,
                custom_downsample=patch_args.custom_downsample,
                patch_level=patch_args.patch_level,
                use_center_shift=heatmap_args.use_center_shift,
                attn_save_path=None, 
                feat_save_path=h5_path, 
                ref_scores=None
            )				

        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:].astype(np.float32))
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)
            print(f"Loaded features shape from h5: {features.shape}")
            features = adjust_features(features, target_dim=embed_dim)
            print(f"Features shape after adjustment: {features.shape}")
            features = features.unsqueeze(0)
            print(f"Features shape after adding batch dimension: {features.shape}")
            torch.save(features, features_path)
            file.close()

        # 2) 推理 -> blockmap.h5
        features = torch.load(features_path)
        print(f"Loaded features shape from pt: {features.shape}")
        process_stack.loc[i, 'bag_size'] = features.size(1)
        wsi_object.saveSegmentation(mask_file)

        task_name = exp_args.save_exp_code
        embed_dim = task_to_embed_dim.get(task_name, 1024)
        print(f"Inferencing single slide with embed_dim: {embed_dim}")
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes, embed_dim)
        del features
        gc.collect()

        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords_nonoverlap = file['coords'][:].astype(np.int32)
            file.close()
            asset_dict = {'attention_scores': A.astype(np.float32), 'coords': coords_nonoverlap}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        for c in range(exp_args.n_classes):
            process_stack.loc[i, f'Pred_{c}'] = Y_hats_str[c]
            process_stack.loc[i, f'p_{c}'] = Y_probs[c]
        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_list_name = os.path.basename(data_args.process_list).replace(".csv", "")
            process_stack.to_csv(f'heatmaps/results/{process_list_name}.csv', index=False)
        else:
            process_stack.to_csv(f'heatmaps/results/{exp_args.save_exp_code}.csv', index=False)

        # 3) 立即导出 blockmap 的 top-k
        try:
            k_top = 15
            try:
                if hasattr(sample_args, 'samples') and isinstance(sample_args.samples, list) and len(sample_args.samples) > 0:
                    s0 = sample_args.samples[0]
                    if isinstance(s0, dict) and 'k' in s0:
                        k_top = int(s0['k'])
                    elif hasattr(s0, 'k'):
                        k_top = int(s0.k)
            except Exception:
                pass

            out_dir_block = os.path.join('heatmaps', 'patches', exp_args.save_exp_code, str(grouping), slide_id, 'topk_blockmap')
            export_topk_patches_from_h5(
                wsi_object=wsi_object,
                h5_path=block_map_save_path,
                out_dir=out_dir_block,
                k=k_top,
                patch_level=patch_args.patch_level,
                patch_size=patch_args.patch_size,
                min_dist=0
            )
        except Exception as e:
            print(f"[WARN] 导出 blockmap top-k 失败: {e}")

        # 4) 细粒度注意力 & 立即导出 fine top-k
        save_path_fine = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_{heatmap_args.use_roi}.h5')
        if heatmap_args.calc_heatmap:
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)
            compute_from_patches(
                wsi_object=wsi_object, 
                img_transforms=img_transforms,
                clam_pred=Y_hats[0], 
                model=model, 
                feature_extractor=feature_extractor, 
                batch_size=exp_args.batch_size, 
                embed_dim=embed_dim,
                attention_only=True,
                top_left=top_left, 
                bot_right=bot_right, 
                patch_size=patch_size, 
                step_size=step_size,
                custom_downsample=patch_args.custom_downsample,
                patch_level=patch_args.patch_level,
                use_center_shift=heatmap_args.use_center_shift,
                attn_save_path=save_path_fine,  
                ref_scores=None
            )

        if os.path.isfile(save_path_fine):
            try:
                out_dir_fine = os.path.join('heatmaps', 'patches', exp_args.save_exp_code, str(grouping), slide_id, 'topk_fine')
                export_topk_patches_from_h5(
                    wsi_object=wsi_object,
                    h5_path=save_path_fine,
                    out_dir=out_dir_fine,
                    k=k_top,
                    patch_level=patch_args.patch_level,
                    patch_size=patch_args.patch_size,
                    min_dist=patch_args.patch_size // 2
                )
            except Exception as e:
                print(f"[WARN] 导出 fine top-k 失败: {e}")
        else:
            print("[INFO] 细粒度 h5 不存在，已导出 blockmap 的 top-k。")

        # 5) 画热力图（向量化分位数）
        # 5.1 blockmap 可视化
        try:
            scores_block, coords_block = _safe_load_scores_coords(block_map_save_path)
            scores_block_vis = percentiles_from_ref(scores_block, scores_block) if heatmap_args.convert_to_percentiles else scores_block
            heatmap_block = drawHeatmap(
                scores=scores_block_vis.flatten(),
                coords=coords_block,
                slide_path=slide_path, 
                wsi_object=wsi_object, 
                cmap=heatmap_args.cmap, 
                alpha=heatmap_args.alpha, 
                use_holes=True, 
                binarize=False, 
                vis_level=-1, 
                blank_canvas=False,
                thresh=-1, 
                patch_size=vis_patch_size, 
                convert_to_percentiles=False
            )
            heatmap_block.save(os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.png'))
            del heatmap_block
            gc.collect()
        except Exception as e:
            print(f"[WARN] blockmap 可视化失败: {e}")

        # 5.2 细粒度可视化
        try:
            if os.path.isfile(save_path_fine):
                scores_fine, coords_fine = _safe_load_scores_coords(save_path_fine)
                if heatmap_args.convert_to_percentiles:
                    ref_for_fine = scores_block if heatmap_args.use_ref_scores else scores_fine
                    scores_fine_vis = percentiles_from_ref(scores_fine, ref_for_fine)
                else:
                    scores_fine_vis = scores_fine

                heatmap_fine = drawHeatmap(
                    scores=scores_fine_vis.flatten(),
                    coords=coords_fine,
                    slide_path=slide_path, 
                    wsi_object=wsi_object, 
                    cmap=heatmap_args.cmap, 
                    alpha=heatmap_args.alpha, 
                    use_holes=True, 
                    binarize=False, 
                    vis_level=-1, 
                    blank_canvas=False,
                    thresh=-1, 
                    patch_size=vis_patch_size, 
                    convert_to_percentiles=False
                )
                heatmap_save_name = (
                    f'{slide_id}_{float(patch_args.overlap)}_roi_{int(heatmap_args.use_roi)}_blur_{int(getattr(heatmap_args,"blur",False))}_'
                    f'rs_{int(heatmap_args.use_ref_scores)}_bc_{int(heatmap_args.blank_canvas)}_a_{float(heatmap_args.alpha)}_'
                    f'l_{int(heatmap_args.vis_level)}_bi_{int(heatmap_args.binarize)}_{float(heatmap_args.binary_thresh)}.{heatmap_args.save_ext}'
                )
                if heatmap_args.save_ext.lower() == 'jpg':
                    heatmap_fine.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap_fine.save(os.path.join(p_slide_save_dir, heatmap_save_name))
                del heatmap_fine
                gc.collect()
        except Exception as e:
            print(f"[WARN] 细粒度可视化失败: {e}")

        # 6) 保存原始视图
        if heatmap_args.save_orig:
            try:
                vis_level_final = heatmap_args.vis_level if heatmap_args.vis_level >= 0 else vis_params['vis_level']
                heatmap_save_name = f'{slide_id}_orig_{int(vis_level_final)}.{heatmap_args.save_ext}'
                if not os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                    heatmap_orig = wsi_object.visWSI(vis_level=vis_level_final, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                    if heatmap_args.save_ext.lower() == 'jpg':
                        heatmap_orig.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                    else:
                        heatmap_orig.save(os.path.join(p_slide_save_dir, heatmap_save_name))
                    del heatmap_orig
                    gc.collect()
            except Exception as e:
                print(f"[WARN] 保存原始视图失败: {e}")

    # 保存 config
    os.makedirs(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code), exist_ok=True)
    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
