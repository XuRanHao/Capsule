# Capsule Search Web

角色 B 的多模态检索工作台，直接调用 Capsule FastAPI 的
`POST /api/v1/search`。

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
