# 前端开发环境

## 已验证环境（2026-07-29）

- Node.js `24.16.0`，项目要求 `>=22.13.0`；
- npm `11.13.0`；
- 依赖使用 `frontend/package-lock.json`，在 `frontend/` 执行 `npm ci`；
- `npm run lint` 与 `npm run build` 均已通过。

## 启动

```bash
cd frontend
npm run dev
```

默认地址为 `http://localhost:3000`，后端 API 默认地址为 `http://localhost:8010`，无需另建
`.env.local`。只有需要覆盖 API 地址时，才从 `.env.example` 复制并修改。

## 构建插件

`vite.config.ts` 依赖 `frontend/build/sites-vite-plugin.ts`，它在构建完成后将 Sites 的 hosting
metadata 与 drizzle 配置打包到 `dist/.openai`。该源文件曾被根目录 `build/` 忽略规则误排除；
`.gitignore` 已显式保留这个文件，避免新环境构建失败。

## 代理注意事项

若 shell 配置了 `ALL_PROXY`，本地健康检查必须绕过代理：

```bash
curl --noproxy '*' http://localhost:3000/
```

否则代理可能返回空响应，而开发服务器本身是正常的。
