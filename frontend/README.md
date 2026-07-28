# Capsule Search Web

角色 B 的多模态检索工作台，直接调用 Capsule FastAPI 的
`POST /api/v1/search`。

## 本地运行

```bash
cp .env.example .env.local
npm install
npm run dev
```

默认页面地址为 `http://localhost:3000`，默认搜索 API 为
`http://localhost:8000`。也可以通过环境变量修改：

```bash
NEXT_PUBLIC_API_BASE_URL=https://your-search-api.example.com
```

## 页面能力

- 文字、图片 URL、图文组合检索
- Asset 类型过滤
- 加载、空结果、错误及部分降级状态
- RRF 总分、命中通道、通道相似度
- 图片、视频片段和 Markdown Block 结果
- 来源文件、视频起始时间和 `source_contexts` 关联段落
- 桌面端双栏工作台与移动端单栏布局

## 验证

```bash
npm run build
npm test
```
