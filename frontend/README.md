# Capsule Web

Capsule 完整 POC 前端工作台，覆盖飞书需求中的导入、处理任务、Asset、
Cluster、搜索与 Capsule 页面。搜索页直接调用 Capsule FastAPI 的
`POST /api/v1/search`；其余页面内置完整可操作的演示数据，便于后端接口逐步接入。

## 本地运行

```bash
cd ..
make setup
# 在根目录 .env 填写 CAPSULE_ARK_API_KEY
make dev
```

默认页面地址为 `http://localhost:3000`，默认搜索 API 为
`http://localhost:8010`。也可以通过环境变量修改：

```bash
NEXT_PUBLIC_API_BASE_URL=https://your-search-api.example.com
```

如只单独启动前端：

```bash
cp .env.example .env.local
npm ci
npm run dev
```

## 页面能力

- `/import`：文件/文件夹导入、格式校验、待导入清单与任务创建
- `/tasks`：处理阶段、实时日志、统计、失败原因与重试
- `/assets`：Asset 网格/列表、搜索过滤、状态与来源信息
- `/assets/:id`：原始定位、关联文字段落、Feature、Embedding 与 Cluster
- `/clusters`：Cluster Run 参数、2D 分布、代表素材、成员与噪声点
- `/search`：完整多模态搜索链路
- `/capsules`：Cluster/Search Capsule 列表、详情、快照、刷新与收藏
- 文字、图片上传/URL、图文组合检索
- 图片快速/精搜、Weighted RRF/Normalized Similarity、可选豆包重排
- Asset、Project、文件、Source File、模型版本、收藏和 Cluster Capsule 过滤
- Query Parser 拆解、维度权重和约束展示
- 加载、空结果、错误及部分降级状态
- 融合总分、命中通道、通道相似度、重排解释
- 图片、视频片段和 Markdown Block 结果
- 相邻视频/Markdown 折叠、同来源限制
- 来源文件、视频起始时间和 `source_contexts` 关联段落
- Search Capsule 最近记录、收藏、快照回放、刷新与删除
- 桌面端双栏工作台与移动端单栏布局

## 验证

```bash
npm run build
npm test
```
