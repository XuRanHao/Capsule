# 图片资产化方案

## 核心规则

- 每一张图片生成一个 `image` Asset。
- 图片按整个文件处理，不做区域检测或局部切分。
- 只提取文件和图片中客观存在的信息。
- EXIF 中不存在的信息不得推测或补全。
- 单张图片处理失败时，由统一资产化入口记录日志并返回失败结果，不向上抛异常。

## 提取字段

- 宽度；
- 高度；
- 宽高比；
- MIME Type；
- 文件大小；
- 色彩模式；
- EXIF；
- 拍摄时间；
- 软件信息；
- 缩略图；
- 文件夹上下文。

## 字段来源

- 宽度、高度、色彩模式和 EXIF：从图片文件读取。
- 宽高比：使用实际宽度除以高度计算；高度无效时不补值。
- MIME Type：根据实际图片格式识别，不只依赖扩展名。
- 文件大小：使用原始文件字节数。
- 拍摄时间、软件信息：仅从对应 EXIF 字段提取，不存在时为 null 或不写入。
- 文件夹上下文：根据 Source File 的相对路径生成，不调用模型推断。
- 缩略图：由原图生成派生文件，不覆盖原图。

## Asset 输出

```json
{
  "asset_type": "image",
  "source_locator": {
    "type": "whole_file"
  },
  "file_info": {
    "width": 2048,
    "height": 2048,
    "aspect_ratio": 1.0,
    "mime_type": "image/jpeg",
    "file_size_bytes": 102400,
    "color_mode": "RGB",
    "exif": {},
    "captured_at": null,
    "software": null,
    "folder_context": ["ProjectA", "Images"]
  },
  "preview_uri": "..."
}
```

字段名称会在实现前与现有 Source File、Asset 数据模型统一，避免同一信息重复存储。

## 尚待确定

- 缩略图最大尺寸、是否保持原宽高比。
- 缩略图输出格式和压缩质量。
- 缩略图在本地工作目录及对象存储中的路径规则。
- EXIF 是否完整保留，或仅保留允许字段，以避免位置等敏感信息无意传播。
- 是否根据 EXIF Orientation 自动旋转缩略图；原图始终不修改。
