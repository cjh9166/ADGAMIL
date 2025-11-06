# create_heatmaps.py
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
import dask.array as da  # 添加 Dask
from scipy.stats import percentileofscore  # 添加用于计算 percentiles
import gc  # 添加垃圾回收

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the task to embed_dim mapping
task_to_embed_dim = {
    "camelyon16_plip": 512,
    "camelyon16_R50": 1024,
    "TCGA_BC_PLIP": 512,
    "TCGA_BC_R50": 1024,
    "TCGA_NSCLC_PLIP": 512,
    "TCGA_NSCLC_R50": 1024,
    "HEATMAP_OUTPUT_camelyon16_plip": 512,  # Added to match save_exp_code
    "HEATMAP_OUTPUT_BRCA": 1024,
    "HEATMAP_OUTPUT_NSCLC": 1024
}

# Cache for linear adjustment layers
linear_adjust_cache = {}

def get_linear_adjust_layer(current_dim, target_dim, device):
    """
    Retrieve a cached linear layer for adjusting feature dimensions.
    If not cached, create, initialize, cache, and return it.
    """
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
    """
    Adjust the features to match the target embedding dimension using a linear layer.
    Utilizes a cache to reuse existing layers for specific dimension adjustments.
    
    Args:
        features (torch.Tensor): Input features tensor.
        target_dim (int): Target embedding dimension.
        
    Returns:
        torch.Tensor: Adjusted features tensor.
    """
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
    """
    Perform inference on a single slide with dynamically adjusted feature dimensions.
    
    Args:
        model (torch.nn.Module): The CLAM model.
        features (torch.Tensor): Features tensor.
        label (str): Ground truth label.
        reverse_label_dict (dict): Mapping from indices to label names.
        k (int, optional): Number of top predictions to consider.
        embed_dim (int, optional): Embedding dimension.
        
    Returns:
        tuple: (Y_hats, Y_hats_str, Y_probs, A)
    """
    # Ensure features have shape (B, N, D)
    print(f"Features shape before any adjustment: {features.shape}")
    if features.dim() == 2:
        features = features.unsqueeze(0)  # Add batch dimension
        print(f"Added batch dimension. New features shape: {features.shape}")
    elif features.dim() == 3:
        print("Features already have batch dimension.")
    else:
        raise ValueError(f"Unexpected feature dimensions: {features.dim()}")

    # Adjust features to match embed_dim
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
    """
    Load parameters from dataframe entry.
    
    Args:
        df_entry (pd.Series): Dataframe row.
        params (dict): Parameters dictionary to update.
        
    Returns:
        dict: Updated parameters dictionary.
    """
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
    """
    Parse and override configuration dictionary based on command-line arguments.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        config_dict (dict): Configuration dictionary loaded from YAML.
        
    Returns:
        dict: Updated configuration dictionary.
    """
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--save_exp_code', type=str, default=None,
                        help='experiment code')
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
    if decision in ['Y', 'y', 'Yes', 'yes']:
        pass
    elif decision in ['N', 'n', 'No', 'NO']:
        exit()
    else:
        raise NotImplementedError

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    encoder_args = args['encoder_arguments']
    encoder_args = argparse.Namespace(**encoder_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    # 设置默认值以防止缺失
    if not hasattr(heatmap_args, 'convert_to_percentiles'):
        heatmap_args.convert_to_percentiles = True  # 默认值

    patch_size = tuple([patch_args.patch_size for i in range(2)])
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
    total = len(process_stack)
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

    blocky_wsi_kwargs = {
        'top_left': None, 
        'bot_right': None, 
        'patch_size': patch_size, 
        'step_size': patch_size, 
        'custom_downsample': patch_args.custom_downsample, 
        'level': patch_args.patch_level, 
        'use_center_shift': heatmap_args.use_center_shift
    }

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

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

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
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # The actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')

        ##### Check if h5_features_file exists ######
        if not os.path.isfile(h5_path):
            # Determine embed_dim based on task
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)  # Default to 1024 if not found
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

        ##### Check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:].astype(np.float32))  # 转换为 float32
            # Determine embed_dim based on task
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)  # Default to 1024 if not found
            print(f"Loaded features shape from h5: {features.shape}")
            # Adjust features to target_dim
            features = adjust_features(features, target_dim=embed_dim)
            print(f"Features shape after adjustment: {features.shape}")
            # Ensure features have shape (1, N, D)
            features = features.unsqueeze(0)
            print(f"Features shape after adding batch dimension: {features.shape}")
            torch.save(features, features_path)
            file.close()

        # Load features 
        features = torch.load(features_path)
        print(f"Loaded features shape from pt: {features.shape}")
        process_stack.loc[i, 'bag_size'] = features.size(1)  # Number of patches
        
        wsi_object.saveSegmentation(mask_file)
        # Determine embed_dim based on task
        task_name = exp_args.save_exp_code
        embed_dim = task_to_embed_dim.get(task_name, 1024)  # Default to 1024 if not found
        print(f"Inferencing single slide with embed_dim: {embed_dim}")
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes, embed_dim)
        del features
        gc.collect()  # 释放内存

        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:].astype(np.int32)  # 确保坐标为整数
            file.close()
            asset_dict = {'attention_scores': A.astype(np.float32), 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # Save top predictions
        for c in range(exp_args.n_classes):
            process_stack.loc[i, f'Pred_{c}'] = Y_hats_str[c]
            process_stack.loc[i, f'p_{c}'] = Y_probs[c]

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_list_name = os.path.basename(data_args.process_list).replace(".csv", "")
            # Ensure directory exists
            os.makedirs("heatmaps/results", exist_ok=True)
            # Save to correct path
            process_stack.to_csv(f'heatmaps/results/{process_list_name}.csv', index=False)
        else:
            process_stack.to_csv(f'heatmaps/results/{exp_args.save_exp_code}.csv', index=False)

        # 修改部分：使用 Dask 分块读取 HDF5 数据并分批生成热力图
        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']

        # 使用 Dask 分块读取
        batch_size = 64  # 根据内存情况调整，尽量设置较小的分块
        scores = da.from_array(dset, chunks=(batch_size, 1))
        coords = da.from_array(coord_dset, chunks=(batch_size, 2))

        # 设置 heatmap_vis_args 和 wsi_kwargs
        convert_to_percentiles = getattr(heatmap_args, 'convert_to_percentiles', True)  # 添加默认值
        heatmap_vis_args = {
            'convert_to_percentiles': convert_to_percentiles, 
            'vis_level': heatmap_args.vis_level, 
            'blur': heatmap_args.blur, 
            'custom_downsample': heatmap_args.custom_downsample
        }
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        # 定义 wsi_kwargs
        wsi_kwargs = {
            'top_left': top_left, 
            'bot_right': bot_right, 
            'patch_size': patch_size, 
            'step_size': step_size, 
            'custom_downsample': patch_args.custom_downsample, 
            'level': patch_args.patch_level, 
            'use_center_shift': heatmap_args.use_center_shift
        }

        heatmap = None  # 初始化 heatmap

        if convert_to_percentiles:
            ref_scores = None
            if heatmap_args.use_ref_scores:
                # 需要先计算 ref_scores
                ref_scores = scores.compute().flatten()
            
            def compute_percentiles(block, ref=None):
                if ref is not None:
                    return np.array([percentileofscore(ref, a) for a in block[:,0]]).reshape(-1,1)
                else:
                    return block

            if heatmap_args.use_ref_scores and ref_scores is not None:
                percentiles = scores.map_blocks(compute_percentiles, ref=ref_scores, dtype=np.float32)
            else:
                percentiles = scores.map_blocks(compute_percentiles, dtype=np.float32)

            # 将 percentiles 转换为 NumPy 数组
            percentiles = percentiles.compute()

            # 同时获取 coords
            coords = coords.compute()

            # 将处理后的数据传递给 drawHeatmap
            heatmap = drawHeatmap(
                scores=percentiles.flatten(),
                coords=coords,
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
                convert_to_percentiles=False  # 已经在这里转换
            )
        else:
            # 如果不需要转换，直接计算 heatmap
            scores = scores.compute().flatten()
            coords = coords.compute()
            heatmap = drawHeatmap(
                scores=scores,
                coords=coords,
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

        heatmap.save(os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.png'))
        del heatmap
        gc.collect()  # 释放内存

        save_path = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_{heatmap_args.use_roi}.h5')

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None

        if heatmap_args.calc_heatmap:
            # Determine embed_dim based on task
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)  # Default to 1024 if not found
            compute_from_patches(
                wsi_object=wsi_object, 
                img_transforms=img_transforms,
                clam_pred=Y_hats[0], 
                model=model, 
                feature_extractor=feature_extractor, 
                batch_size=exp_args.batch_size, 
                embed_dim=embed_dim,
                attention_only=True,  # Add this flag
                **wsi_kwargs,  # type: ignore
                attn_save_path=save_path,  
                ref_scores=ref_scores
            )

        if not os.path.isfile(save_path):
            print(f'heatmap {save_path} not found')
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_False.h5')
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue

        with h5py.File(save_path, 'r') as file:
            dset = file['attention_scores']
            coord_dset = file['coords']
            # 使用 Dask 分块读取
            batch_size = 64  # 根据内存情况调整，尽量设置较小的分块
            scores = da.from_array(dset, chunks=(batch_size, 1))
            coords = da.from_array(coord_dset, chunks=(batch_size, 2))

            if convert_to_percentiles:
                ref_scores = None
                if heatmap_args.use_ref_scores:
                    # 需要先计算 ref_scores
                    ref_scores = scores.compute().flatten()
                
                def compute_percentiles(block, ref=None):
                    if ref is not None:
                        return np.array([percentileofscore(ref, a) for a in block[:,0]]).reshape(-1,1)
                    else:
                        return block

                if heatmap_args.use_ref_scores and ref_scores is not None:
                    percentiles = scores.map_blocks(compute_percentiles, ref=ref_scores, dtype=np.float32)
                else:
                    percentiles = scores.map_blocks(compute_percentiles, dtype=np.float32)

                # 将 percentiles 转换为 NumPy 数组
                percentiles = percentiles.compute()

                # 同时获取 coords
                coords = coords.compute()

                # 将处理后的数据传递给 drawHeatmap
                heatmap = drawHeatmap(
                    scores=percentiles.flatten(),
                    coords=coords,
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
                    convert_to_percentiles=False  # 已经在这里转换
                )
            else:
                # 如果不需要转换，直接计算 heatmap
                scores = scores.compute().flatten()
                coords = coords.compute()
                heatmap = drawHeatmap(
                    scores=scores,
                    coords=coords,
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

        heatmap_save_name = f'{slide_id}_{float(patch_args.overlap)}_roi_{int(heatmap_args.use_roi)}_blur_{int(heatmap_args.blur)}_' \
                             f'rs_{int(heatmap_args.use_ref_scores)}_bc_{int(heatmap_args.blank_canvas)}_a_{float(heatmap_args.alpha)}_' \
                             f'l_{int(heatmap_args.vis_level)}_bi_{int(heatmap_args.binarize)}_' \
                             f'{float(heatmap_args.binary_thresh)}.{heatmap_args.save_ext}'

        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(
                scores=scores,
                coords=coords,
                slide_path=slide_path, 
                wsi_object=wsi_object,  
                cmap=heatmap_args.cmap, 
                alpha=heatmap_args.alpha, 
                binarize=heatmap_args.binarize, 
                blank_canvas=heatmap_args.blank_canvas,
                thresh=heatmap_args.binary_thresh,  
                patch_size=vis_patch_size,
                overlap=patch_args.overlap, 
                top_left=top_left, 
                bot_right=bot_right
            )
            if heatmap_args.save_ext.lower() == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
            del heatmap
            gc.collect()  # 释放内存

        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = f'{slide_id}_orig_{int(vis_level)}.{heatmap_args.save_ext}'
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext.lower() == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
                del heatmap
                gc.collect()  # 释放内存

    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
