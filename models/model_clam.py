# models/model_clam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入动态图嵌入模块
from models.dynamic_graph import DynamicGraphEmbedding

class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        modules = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        if dropout:
            modules.append(nn.Dropout(0.25))
        modules.append(nn.Linear(D, n_classes))
        self.attention_net = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        x: (B*N, L)
        """
        A = self.attention_net(x)  # (B*N, n_classes)
        return A, x  # 返回 (B*N, n_classes), (B*N, L)

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        if dropout:
            self.attention_a.add_module('dropout', nn.Dropout(0.25))
            self.attention_b.add_module('dropout', nn.Dropout(0.25))
        self.attention_c = nn.Linear(D, n_classes)
    
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b
        A = self.attention_c(A)
        return A, x  # 返回 (B*N, n_classes), (B*N, L)

class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, num_neighbors=5):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        self.fc = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        if gate:
            self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        
        # 初始化动态图嵌入模块，支持可变的 num_neighbors
        self.dynamic_graph = DynamicGraphEmbedding(input_dim=size[1], hidden_dim=size[1], num_neighbors=num_neighbors)
    
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, attn_mask=None):
        """
        h: (B, N, D)
        attn_mask: (B, N)
        """
        # 先经过全连接层
        B, N, D = h.size()
        h = h.view(B*N, D)
        h = self.fc(h)  # (B*N, D_fc)
        
        # 使用动态图嵌入更新特征
        h = h.view(B, N, -1)
        graph_features = self.dynamic_graph(h)  # (B, N, hidden_dim)
        h = graph_features.view(B*N, -1)  # (B*N, hidden_dim)
        
        # 计算注意力权重
        A, h = self.attention_net(h)  # (B*N, 1), (B*N, D)
        A = A.view(B, N, -1)  # (B, N, 1)
        if attention_only:
            return A
        A_raw = A.clone()
        if attn_mask is not None:
            A = A.masked_fill(~attn_mask.unsqueeze(-1), float('-inf'))
        A = F.softmax(A, dim=1)  # (B, N, 1)
        
        # 计算Bag特征
        h = h.view(B, N, -1)  # (B, N, D)
        M = torch.bmm(A.transpose(1, 2), h).squeeze(1)  # (B, D)
        logits = self.classifiers(M)  # (B, n_classes)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, num_neighbors=5):
        super(CLAM_MB, self).__init__(gate, size_arg, dropout, k_sample, n_classes,
                                      instance_loss_fn, subtyping, embed_dim, num_neighbors)
        size = self.size_dict[size_arg]
        if gate:
            self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            self.attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])
    
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, attn_mask=None):
        """
        h: (B, N, D)
        attn_mask: (B, N)
        """
        # 先经过全连接层
        B, N, D = h.size()
        h = h.view(B*N, D)
        h = self.fc(h)  # (B*N, D_fc)
        
        # 使用动态图嵌入更新特征
        h = h.view(B, N, -1)
        graph_features = self.dynamic_graph(h)  # (B, N, hidden_dim)
        h = graph_features.view(B*N, -1)  # (B*N, hidden_dim)
        
        # 计算注意力权重
        A, h = self.attention_net(h)  # (B*N, n_classes), (B*N, D)
        A = A.view(B, N, -1)  # (B, N, n_classes)
        if attention_only:
            return A
        A_raw = A.clone()
        if attn_mask is not None:
            A = A.masked_fill(~attn_mask.unsqueeze(-1), float('-inf'))
        A = F.softmax(A, dim=1)  # (B, N, n_classes)
        
        # 计算Bag特征和预测
        h = h.view(B, N, -1)  # (B, N, D)
        M_list = []
        logits_list = []
        for i in range(self.n_classes):
            A_i = A[:, :, i].unsqueeze(-1)  # (B, N, 1)
            M_i = torch.bmm(A_i.transpose(1, 2), h).squeeze(1)  # (B, D)
            logits_i = self.classifiers[i](M_i)  # (B, 1)
            M_list.append(M_i)
            logits_list.append(logits_i)
        M = torch.stack(M_list, dim=1)  # (B, n_classes, D)
        logits = torch.cat(logits_list, dim=1)  # (B, n_classes)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict
