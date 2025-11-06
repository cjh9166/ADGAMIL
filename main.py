from __future__ import print_function
import argparse
import pdb
import os
import math
import time

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.utils import collate_MIL_padded
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import logging
import sys

# Import CLAM model
from models.model_clam import CLAM_SB, CLAM_MB

# Import SummaryWriter for TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Import sklearn metrics for additional evaluation
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)

logger = logging.getLogger('main')
logging.getLogger('dataset_modules.dataset_generic').setLevel(logging.ERROR)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

# 新增函数：验证标签分布
def check_label_distribution(dataset, split_name):
    labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    logger.info(f"{split_name} Label Distribution: {label_dist}")

def main(args, dataset):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
        
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_recall = []
    all_val_recall = []
    all_test_f1 = []
    all_val_f1 = []
    all_test_specificity = []
    all_val_specificity = []
    folds = np.arange(start, end)
    
    epoch_times = []
    
    if args.model_type == 'clam_sb':
        clam_model = CLAM_SB(
            gate=not args.no_inst_cluster,
            size_arg=args.model_size,
            dropout=args.drop_out,
            k_sample=args.B,
            n_classes=args.n_classes,
            instance_loss_fn=get_loss_function(args.inst_loss),
            subtyping=args.subtyping,
            embed_dim=args.embed_dim,
            num_neighbors=args.num_neighbors
        )
    elif args.model_type == 'clam_mb':
        clam_model = CLAM_MB(
            gate=not args.no_inst_cluster,
            size_arg=args.model_size,
            dropout=args.drop_out,
            k_sample=args.B,
            n_classes=args.n_classes,
            instance_loss_fn=get_loss_function(args.inst_loss),
            subtyping=args.subtyping,
            embed_dim=args.embed_dim,
            num_neighbors=args.num_neighbors
        )
    else:
        raise NotImplementedError(f"Model type '{args.model_type}' is not implemented.")

    clam_model.to(device)
    
    optimizer = get_optim(clam_model, args)
    
    if args.log_data:
        tensorboard_log_dir = os.path.join(args.results_dir, 'tensorboard')
        if not os.path.isdir(tensorboard_log_dir):
            os.mkdir(tensorboard_log_dir)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"Initialized TensorBoard SummaryWriter at {tensorboard_log_dir}")
    else:
        writer = None
    
    for fold in folds:
        logger.info(f"Starting Fold [{fold}]")
        seed_torch(args.seed)
        split_path = os.path.join(args.split_dir, f"splits_{fold}.csv")
        if not os.path.isfile(split_path):
            logger.error(f"Split file not found: {split_path}")
            raise FileNotFoundError(f"Split file not found: {split_path}")
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, 
            csv_path=split_path
        )
        
        # 验证标签分布
        check_label_distribution(train_dataset, f"Fold [{fold}] Train")
        check_label_distribution(val_dataset, f"Fold [{fold}] Val")
        check_label_distribution(test_dataset, f"Fold [{fold}] Test")
        
        # 修改1: 动态计算类别权重
        labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
        class_counts = np.bincount(labels)
        class_weights = torch.tensor([1.0, 4.0], dtype=torch.float).to(device)  # 调整为更温和的权重
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        logger.info(f"Fold [{fold}] - Class weights set to {class_weights.tolist()}")

        # 修改2: 优化加权采样，反转权重以增加正类采样概率
        if args.weighted_sample:
            weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float)
            sample_weights = weights[labels]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, collate_fn=collate_MIL_padded)
            logger.info(f"Fold [{fold}] - Weighted sampling enabled with class counts: {dict(enumerate(class_counts))}")
        else:
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_MIL_padded)
            logger.info(f"Fold [{fold}] - Weighted sampling disabled, using shuffle")
        
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_MIL_padded)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_MIL_padded)
        
        logger.info(f"Fold [{fold}] - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        
        patience = 20
        best_score = 0.0
        min_delta = 0.001
        counter = 0
        
        # 修改3: 使用 ReduceLROnPlateau 调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5,min_lr=1e-6, verbose=True)
        
        previous_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(args.max_epochs):
            epoch_start_time = time.time()
            logger.info(f"Epoch [{epoch+1}/{args.max_epochs}] Fold [{fold}]")
            clam_model.train()
            
            epoch_loss = 0.0
            for idx, batch in enumerate(train_loader):
                patch_features = batch['features'].to(device)
                slide_labels = batch['labels'].to(device)
                attn_mask = batch['mask'].to(device)
                
                if epoch == 0 and idx == 0:
                    logger.info(f"Fold [{fold}] Epoch [{epoch+1}] - Feature shape: {patch_features.shape}, Sample feature: {patch_features[0, :5]}")
                
                outputs, _, _, _, _ = clam_model(patch_features, label=slide_labels, instance_eval=True, attn_mask=attn_mask)
                loss = criterion(outputs, slide_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            if writer:
                writer.add_scalar(f'Fold_{fold}/Train_Loss', avg_epoch_loss, epoch+1)
            logger.info(f"Fold [{fold}] Epoch [{epoch+1}/{args.max_epochs}] Average Loss: {avg_epoch_loss:.4f}")
            
            val_auc, val_acc, val_recall, val_f1, val_specificity = evaluate(val_loader, clam_model, device)
            logger.info(f"Fold [{fold}] Epoch [{epoch+1}/{args.max_epochs}] Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}, Val Recall={val_recall:.4f}, Val F1={val_f1:.4f}, Val Specificity={val_specificity:.4f}")
            if writer:
                writer.add_scalar(f'Fold_{fold}/Val_AUC', val_auc, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_Acc', val_acc, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_Recall', val_recall, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_F1', val_f1, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_Specificity', val_specificity, epoch+1)
            
            if args.early_stopping:
                if val_auc > best_score + min_delta:
                    best_score = val_auc
                    counter = 0
                    checkpoint_path = os.path.join(args.results_dir, f's_{fold}_best_checkpoint.pt')
                    torch.save(clam_model.state_dict(), checkpoint_path)
                    logger.info(f"New best model saved for Fold [{fold}] at epoch {epoch+1} with Val AUC={val_auc:.4f}")
                else:
                    counter += 1
                    logger.info(f"No improvement in Val AUC for {counter} epochs (Current AUC={val_auc:.4f}, Best AUC={best_score:.4f})")
                
                if counter >= patience:
                    logger.info(f"Early stopping triggered for Fold [{fold}] at epoch {epoch+1}")
                    break
            
            # 修改4: 根据验证集 AUC 更新学习率
            scheduler.step(val_auc)
            
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > 0 and current_lr != previous_lr:
                logger.info(f"Learning rate adjusted to {current_lr:.6f}")
            previous_lr = current_lr
            
            if writer:
                writer.flush()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            if len(epoch_times) > 0:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs_in_fold = args.max_epochs - (epoch + 1)
                remaining_folds = end - (fold + 1)
                total_remaining_epochs = remaining_epochs_in_fold + remaining_folds * args.max_epochs
                estimated_remaining_time = avg_epoch_time * total_remaining_epochs
                time_used = sum(epoch_times)
                logger.info(f"Time used so far: {format_time(time_used)}, Estimated remaining time: {format_time(estimated_remaining_time)}")
        
        logger.info(f"Evaluating Test Set for Fold [{fold}]")
        test_auc, test_acc, test_recall, test_f1, test_specificity = evaluate(test_loader, clam_model, device)
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_recall.append(test_recall)
        all_val_recall.append(val_recall)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        all_test_specificity.append(test_specificity)
        all_val_specificity.append(val_specificity)
        
        if writer:
            writer.add_scalar(f'Fold_{fold}/Test_AUC', test_auc, args.max_epochs)
            writer.add_scalar(f'Fold_{fold}/Test_Acc', test_acc, args.max_epochs)
            writer.add_scalar(f'Fold_{fold}/Test_Recall', test_recall, args.max_epochs)
            writer.add_scalar(f'Fold_{fold}/Test_F1', test_f1, args.max_epochs)
            writer.add_scalar(f'Fold_{fold}/Test_Specificity', test_specificity, args.max_epochs)
            writer.flush()
        
        filename = os.path.join(args.results_dir, f'split_{fold}_results.pkl')
        save_pkl(filename, {
            'val_auc': val_auc,
            'test_auc': test_auc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'val_recall': val_recall,
            'test_recall': test_recall,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'val_specificity': val_specificity,
            'test_specificity': test_specificity
        })
        logger.info(f"Fold [{fold}] Evaluation Results: Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}, Val Recall={val_recall:.4f}, Val F1={val_f1:.4f}, Val Specificity={val_specificity:.4f}, Test AUC={test_auc:.4f}, Test Acc={test_acc:.4f}, Test Recall={test_recall:.4f}, Test F1={test_f1:.4f}, Test Specificity={test_specificity:.4f}")
        
        checkpoint_path = os.path.join(args.results_dir, f's_{fold}_checkpoint.pt')
        torch.save(clam_model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint for Fold [{fold}] at {checkpoint_path}")

    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc,
        'test_recall': all_test_recall,
        'val_recall': all_val_recall,
        'test_f1': all_test_f1,
        'val_f1': all_val_f1,
        'test_specificity': all_test_specificity,
        'val_specificity': all_val_specificity
    })

    if len(folds) != args.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)

    logger.info("################# Final Results ###################")
    logger.info(final_df)
    logger.info("Finished!")
    logger.info("End script")
    
    if args.log_data and writer:
        writer.close()

def evaluate(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            patch_features = batch['features'].to(device)
            slide_labels = batch['labels'].to(device)
            attn_mask = batch['mask'].to(device)
            
            outputs, _, _, _, _ = model(patch_features, label=slide_labels, instance_eval=False, attn_mask=attn_mask)
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            labels = slide_labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.0
    
    binary_preds = np.round(all_preds)
    val_acc = accuracy_score(all_labels, binary_preds)
    val_recall = recall_score(all_labels, binary_preds)
    val_f1 = f1_score(all_labels, binary_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return val_auc, val_acc, val_recall, val_f1, val_specificity

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--task', type=str, choices=[
    'camelyon16_plip',
    'camelyon16_R50',
    'TCGA_BC_PLIP',
    'TCGA_BC_R50',
    'TCGA_NSCLC_PLIP',
    'TCGA_NSCLC_R50'
], required=True, help='Task type')
parser.add_argument('--data_root_dir', type=str, required=True, help='Data directory (project root)')
parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='Fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-2, help='Weight decay (default: 1e-2)')  # 用户自行调整为 1e-2
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (default: -1, last fold)')
parser.add_argument('--results_dir', default='./results', help='Results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, help='Manually specify the set of splits to use')
parser.add_argument('--log_data', action='store_true', default=False, help='Log data using TensorBoard')
parser.add_argument('--testing', action='store_true', default=False, help='Debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer (default: adam)')
parser.add_argument('--drop_out', type=float, default=0.25, help='Dropout rate (default: 0.25)')  # 用户要求保持不变
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce', help='Slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', help='Type of model (default: clam_sb)')
parser.add_argument('--exp_code', type=str, required=True, help='Experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='Size of model')
parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='Disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'None'], default='None', help='Instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, help='Subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7, help='CLAM: Weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='Number of positive/negative patches to sample for CLAM')
parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors in dynamic graph embedding')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {
    'num_splits': args.k, 
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs, 
    'results_dir': args.results_dir, 
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    "use_drop_out": args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt
}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B,
        'num_neighbors': args.num_neighbors
    })

print('\nLoad Dataset')

if args.task in ['TCGA_BC_PLIP', 'TCGA_BC_R50', 'TCGA_NSCLC_PLIP', 'TCGA_NSCLC_R50']:
    args.n_classes = 2
    if args.task in ['TCGA_BC_PLIP', 'TCGA_BC_R50']:
        label_dict = {'IDC': 0, 'ILC': 1}
    elif args.task in ['TCGA_NSCLC_PLIP', 'TCGA_NSCLC_R50']:
        label_dict = {'LUAD': 0, 'LUSC': 1}
    
    if args.task == 'TCGA_BC_PLIP':
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'tcga_brca_plip.csv')
        data_dir = os.path.join(args.data_root_dir, 'RRT_data', 'tcga-subtyping', 'TCGA-BRCA PLIP', 'pt_files')
    elif args.task == 'TCGA_BC_R50':
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'tcga_brca_r50.csv')
        data_dir = os.path.join(args.data_root_dir, 'RRT_data', 'tcga-subtyping', 'TCGA-BRCA R50', 'pt_files')
    elif args.task == 'TCGA_NSCLC_PLIP':
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'tcga_nsclc_plip.csv')
        data_dir = os.path.join(args.data_root_dir, 'RRT_data', 'tcga-subtyping', 'TCGA-NSCLC PLIP', 'pt_files')
    elif args.task == 'TCGA_NSCLC_R50':
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'tcga_nsclc_r50.csv')
        data_dir = os.path.join(args.data_root_dir, 'RRT_data', 'tcga-subtyping', 'TCGA-NSCLC R50', 'pt_files')

    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=data_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=label_dict,
        label_col='label',
        ignore=[]
    )
elif args.task == 'camelyon16_plip':
    args.n_classes = 2
    label_dict = {'0': 0, '1': 1}
    dataset = Generic_MIL_Dataset(
        csv_path=os.path.join(args.data_root_dir, 'dataset_csv', 'camelyon16_plip.csv'),
        data_dir=os.path.join(args.data_root_dir, 'RRT_data', 'camelyon16-diagnosis', 'CAMELYON16 PLIP', 'pt'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=label_dict,
        label_col='label',
        ignore=[]
    )
elif args.task == 'camelyon16_R50':
    args.n_classes = 2
    label_dict = {'0': 0, '1': 1}
    dataset = Generic_MIL_Dataset(
        csv_path=os.path.join(args.data_root_dir, 'dataset_csv', 'camelyon16_R50.csv'),
        data_dir=os.path.join(args.data_root_dir, 'RRT_data', 'camelyon16-diagnosis', 'CAMELYON16 R50', 'pt'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=label_dict,
        label_col='label',
        ignore=[]
    )
else:
    raise NotImplementedError(f"Task '{args.task}' is not implemented.")

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', f"{args.task}_{int(args.label_frac*100)}")
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir), f"Split directory does not exist: {args.split_dir}"

settings.update({'split_dir': args.split_dir})

experiment_settings_path = os.path.join(args.results_dir, f'experiment_{args.exp_code}.txt')
with open(experiment_settings_path, 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print(f"{key}:  {val}")        

if __name__ == "__main__":
    results = main(args, dataset) 
    print("Finished!")
    print("End script")