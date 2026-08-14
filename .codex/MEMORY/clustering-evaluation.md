# 聚类算法评测

## 统一评测指标

每次聚类至少记录：

- Silhouette Score：越高越好；
- Calinski–Harabasz Index（CH）：越高越好，仅在同一数据集内比较；
- Davies–Bouldin Index（DBI）：越低越好；
- DBCV：越高越好，特别用于密度聚类。

还应记录算法、选中参数、cluster_count、noise_count、coverage、运行耗时。少于两个有效簇时，四项质量指标必须为 `null`。

## 2026-07-29 合成数据初测

条件：300 个固定随机向量，包含 4 个不同密度的潜在簇（264 条）和 36 条噪声；所有算法使用与生产一致的 `L2 → PCA(64) → L2` 预处理。

| 算法 | 选中参数 | 簇数 | 噪声 | Silhouette | CH | DBI | DBCV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HDBSCAN | min_cluster_size=5, min_samples=2 | 4 | 74 | 0.244 | 47.5 | 1.825 | 0.129 |
| K-Means | k=5 | 5 | 0 | 0.185 | 35.8 | 2.827 | 0.108 |
| Agglomerative (Ward) | k=7 | 7 | 0 | 0.182 | 25.0 | 3.017 | 0.104 |
| DBSCAN | eps=0.996, min_samples=3 | 6 | 84 | 0.239 | 29.4 | 1.735 | 0.142 |

结论：本合成场景下 HDBSCAN 的簇数与潜在簇结构一致，且轮廓系数与 CH 最好；DBSCAN 的 DBCV/DBI 略好但簇数和噪声更多。K-Means、Ward 会强制覆盖全部样本，且用轮廓系数选 K 时有过度切分倾向。

## 限制与下一步

当前 Docker PostgreSQL 中没有 `indexed` 状态的 Embedding，以上并非真实个人资产评测。真实数据导入后必须在同一 embedding_type、同一向量集合上复跑 HDBSCAN、K-Means、Agglomerative、DBSCAN，再结合代表资产内容人工检查，决定默认算法。
