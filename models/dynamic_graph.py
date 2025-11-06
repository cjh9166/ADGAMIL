# models/dynamic_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors=5):
        super(DynamicGraphEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors

        # 定义线性层，用于特征变换
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，形状 (B, num_patches, input_dim)
        Returns:
            graph_features: 图嵌入后的特征，形状 (B, num_patches, hidden_dim)
        """
        B, N, D = x.size()  # B: 批大小, N: 节点数（patch数）, D: 输入维度

        # # 计算特征的余弦相似度
        # x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化，避免除以零
        # S = torch.bmm(x_norm, x_norm.transpose(1, 2))  # 相似度矩阵，形状 (B, N, N)

        # # 掩盖自环（节点与自身的相似度设为负无穷）
        # mask = torch.eye(N, device=x.device).bool().unsqueeze(0).expand(B, -1, -1)
        # S.masked_fill_(mask, float('-inf'))

        # # 选择每个节点的 top-k 邻居
        # values, indices = torch.topk(S, k=self.num_neighbors, dim=2)  # values: (B, N, num_neighbors), indices: (B, N, num_neighbors)

        # # 计算邻居的加权系数（基于相似度）
        # weights = F.softmax(values, dim=2)  # (B, N, num_neighbors)

        # # 收集邻居特征
        # neighbor_features = torch.stack([x[b, indices[b], :] for b in range(B)], dim=0)  # (B, N, num_neighbors, D)

        # # 加权聚合邻居特征
        # aggregate_neighbors = torch.sum(weights.unsqueeze(-1) * neighbor_features, dim=2)  # (B, N, D)

        # # 结合自身特征与邻居特征
        # h_combined = x + aggregate_neighbors  # (B, N, D)

# --- 开始修改 ---
        # 如果节点数量 N 小于或等于 1，或者 N-1（排除了自身）小于等于 num_neighbors，
        # 那么无法选择 k 个邻居，或者选择所有可用的邻居。
        # 针对 N=1 的情况：没有邻居可聚合。
        # 针对 N > 1 但 N-1 < self.num_neighbors 的情况：选择所有可用的 N-1 个邻居。
        if N <= 1:
            # 如果只有一个 patch，没有邻居可聚合。
            # 直接让特征通过线性层。
            h_combined = x
            print(f"DEBUG: N is {N}. 在 DynamicGraphEmbedding 中跳过邻居聚合。")
        else:
            # 对于 N > 1 的情况，执行邻居聚合
            # 确保 k 不会大于 N-1（因为我们排除了自身）
            actual_k = min(self.num_neighbors, N - 1)

            if actual_k <= 0: # 这种情况下，即使 N > 1，也没有实际的邻居可选
                h_combined = x
                print(f"DEBUG: N is {N}, actual_k is {actual_k}. 跳过邻居聚合。")
            else:
                # 计算特征的余弦相似度
                x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化，避免除以零
                S = torch.bmm(x_norm, x_norm.transpose(1, 2))  # 相似度矩阵，形状 (B, N, N)

                # 掩盖自环（节点与自身的相似度设为负无穷）
                mask = torch.eye(N, device=x.device).bool().unsqueeze(0).expand(B, -1, -1)
                S.masked_fill_(mask, float('-inf'))

                # 选择每个节点的 top-k 邻居
                values, indices = torch.topk(S, k=actual_k, dim=2)  # values: (B, N, actual_k), indices: (B, N, actual_k)

                # 计算邻居的加权系数（基于相似度）
                weights = F.softmax(values, dim=2)  # (B, N, actual_k)

                # 收集邻居特征
                neighbor_features = torch.stack([x[b, indices[b], :] for b in range(B)], dim=0)  # (B, N, actual_k, D)

                # 加权聚合邻居特征
                aggregate_neighbors = torch.sum(weights.unsqueeze(-1) * neighbor_features, dim=2)  # (B, N, D)

                # 结合自身特征与邻居特征
                h_combined = x + aggregate_neighbors  # (B, N, D)
        # --- 结束修改 ---



        # 通过线性层处理
        h_combined = h_combined.view(B * N, D)
        h_combined = F.relu(self.conv1(h_combined))  # (B*N, hidden_dim)
        h_combined = F.relu(self.conv2(h_combined))  # (B*N, hidden_dim)
        graph_features = h_combined.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        return graph_features
