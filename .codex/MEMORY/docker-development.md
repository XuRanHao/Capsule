# Docker 开发环境

## 组成

- `app`：Python 3.11、uv、FFmpeg 和项目依赖。
- `postgres`：业务元数据存储。
- `minio`：S3 兼容对象存储。
- `etcd`：Milvus 元数据依赖。
- `milvus`：向量存储。

应用源码通过 bind mount 映射到 `/workspace`，虚拟环境位于
`/opt/capsule-venv`，避免被源码挂载覆盖。

## 已验证结果

- Python 3.11.14
- uv 0.9.30
- FFmpeg 5.1.9
- pytest：9 passed
- ruff：通过
- mypy：通过

## ARM64 注意事项

`hdbscan 0.8.44` 在当前 ARM64/Python 3.11 组合下未获取到预编译 wheel，
需要通过 `build-essential` 在镜像构建阶段编译。首次构建耗时较长，后续构建会复用 Docker 缓存。

## 网络问题

首次拉取 Docker Hub 基础设施镜像时可能遇到 CloudFront TLS handshake timeout。
该问题属于外部镜像网络，不是 Compose 配置错误；重新执行 `docker compose up -d`
可以复用已下载层继续拉取。

若宿主机端口被其他项目占用，Compose 可能留下“容器已创建/运行，但没有挂入
`capsule_default` 网络”的半完成状态。即使容器自身显示 healthy，其他容器仍会因
Docker DNS 解析不到服务名而失败。释放端口后，对缺少网络地址的服务执行：

```bash
docker compose up -d --force-recreate <service>
```

本次实际冲突来自已停用的 Monsora 项目：`monsora-minio` 占用 9000/9001，
`monsora-postgres` 占用 5432；两个容器均只执行了 stop，没有删除容器或数据。
