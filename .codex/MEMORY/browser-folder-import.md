# 浏览器文件夹导入

## 已确认协议

前端优先调用 `showDirectoryPicker()`；不支持该 API 时退回 `<input type="file" webkitdirectory>`。
两种方式均使用浏览器可提供的相对路径，不能也不应传递用户机器上的绝对路径。

导入协议分三步，避免把整个文件夹封装成一个不可重试的大请求：

1. `POST /api/v1/import-jobs`：创建空的、`queued` 状态的任务；
2. `POST /api/v1/import-jobs/{job_id}/files`：一次上传一个文件及其 `relative_path`；相同路径重复上传会原子覆盖，前端每个文件自动重试三次；
3. `POST /api/v1/import-jobs/{job_id}/complete`：服务端确认至少一个支持的文件后，冻结任务并以 FastAPI 后台任务启动既有 `PipelineRunner`。

`GET /api/v1/import-jobs/{job_id}?workspace_id=...` 用于前端轮询处理状态。

## 存储与边界

- 浏览器文件先暂存于 `CAPSULE_IMPORT_ROOT/<job_id>/`；当前不上传原始文件至对象存储，因此暂存目录必须保留，供 `SourceFile.storage_uri` 使用。
- 服务端允许 `.md`、`.txt`、`.jpg`、`.jpeg`、`.png`、`.webp`、`.mp4`、`.mov`；当前页面按已确认的范围展示 Markdown、图片和视频。
- 单文件上限由 `CAPSULE_IMPORT_FILE_MAX_BYTES` 配置，默认 2 GiB。
- FastAPI `BackgroundTasks` 是进程内后台执行；服务重启不会自动续跑已开始的任务。将来需要跨重启可靠队列时，再接入持久化 Worker。
