# 全量数据清库

## 使用方式

Assets 页面提供“清空全部数据”按钮。浏览器弹出中文确认框，用户只需选择确认或取消；内部确认标记由前端自动发送，
不展示给用户。

## 清理范围

清空所有 Workspace 的 PostgreSQL Asset、Source File、处理任务、Embedding、Cluster、搜索历史与查询图片记录；
Workspace 本身也会删除，下次导入自动新建。随后清空配置的 Capsule MinIO bucket、Milvus collection 中的全部向量，
以及 `CAPSULE_IMPORT_ROOT` 下的全部暂存内容（保留根目录）。外部存储清理失败会返回警告，不会恢复已经删除的数据库记录。

## 并发约束

任意工作区存在 `queued`、`running` 或 `retrying` 的导入任务时，接口返回 409 并拒绝清空，防止后台处理在
清库后继续写入数据。
