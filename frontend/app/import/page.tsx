"use client";

import { DragEvent, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import DemoShell from "../components/DemoShell";

type QueuedFile = {
  name: string;
  path: string;
  type: "markdown" | "image" | "video";
  size: number;
};

const EXAMPLE_FILES: QueuedFile[] = [
  {
    name: "2026-夏日情绪板.md",
    path: "灵感库/视觉参考/2026-夏日情绪板.md",
    type: "markdown",
    size: 128_400,
  },
  {
    name: "street-twilight.jpg",
    path: "灵感库/视觉参考/images/street-twilight.jpg",
    type: "image",
    size: 4_820_000,
  },
  {
    name: "田野参考.mp4",
    path: "项目/短片A/参考/田野参考.mp4",
    type: "video",
    size: 186_400_000,
  },
  {
    name: "导演阐述.md",
    path: "项目/短片A/导演阐述.md",
    type: "markdown",
    size: 84_200,
  },
];

const folderInputProps = {
  webkitdirectory: "",
  directory: "",
} as React.InputHTMLAttributes<HTMLInputElement>;

function normalizeFiles(files: FileList | File[]): QueuedFile[] {
  return Array.from(files)
    .map((file) => {
      const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
      const type =
        ["md", "markdown"].includes(extension)
          ? "markdown"
          : ["mp4", "mov", "mkv", "webm"].includes(extension)
            ? "video"
            : ["jpg", "jpeg", "png", "webp", "gif"].includes(extension)
              ? "image"
              : null;
      if (!type) return null;
      return {
        name: file.name,
        path: file.webkitRelativePath || file.name,
        type,
        size: file.size,
      };
    })
    .filter((item): item is QueuedFile => item !== null);
}

function formatBytes(bytes: number) {
  if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export default function ImportPage() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<QueuedFile[]>(EXAMPLE_FILES);
  const [dragging, setDragging] = useState(false);
  const [starting, setStarting] = useState(false);

  const counts = useMemo(
    () => ({
      markdown: files.filter((item) => item.type === "markdown").length,
      image: files.filter((item) => item.type === "image").length,
      video: files.filter((item) => item.type === "video").length,
      bytes: files.reduce((total, item) => total + item.size, 0),
    }),
    [files],
  );

  const addFiles = (incoming: FileList | File[]) => {
    const normalized = normalizeFiles(incoming);
    setFiles((current) => {
      const seen = new Set(current.map((item) => item.path));
      return [...current, ...normalized.filter((item) => !seen.has(item.path))];
    });
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  };

  const startImport = () => {
    if (!files.length) return;
    setStarting(true);
    window.setTimeout(() => router.push("/tasks"), 650);
  };

  return (
    <DemoShell
      active="import"
      eyebrow="IMPORT / 44.1"
      title="把散落的素材，带回同一个地方。"
      description="保留文件夹层级和相对路径，自动识别 Markdown、图片与视频，提交后进入异步处理链路。"
      actions={
        <button className="secondary-action" onClick={() => setFiles([])}>
          清空队列
        </button>
      }
    >
      <div className="import-overview">
        <div>
          <small>待导入文件</small>
          <strong>{files.length}</strong>
          <span>{formatBytes(counts.bytes)}</span>
        </div>
        <div>
          <small>Markdown</small>
          <strong>{counts.markdown}</strong>
          <span>语义 Block</span>
        </div>
        <div>
          <small>图片</small>
          <strong>{counts.image}</strong>
          <span>原图 + 上下文</span>
        </div>
        <div>
          <small>视频</small>
          <strong>{counts.video}</strong>
          <span>场景切分</span>
        </div>
      </div>

      <div className="import-workspace">
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <div className="drop-orbit" aria-hidden="true">
            <span>＋</span>
          </div>
          <h2>拖入文件或整个文件夹</h2>
          <p>目录结构会写入 Source File，重复 SHA-256 文件会自动跳过。</p>
          <div>
            <button onClick={() => fileInput.current?.click()}>
              选择文件
            </button>
            <button onClick={() => folderInput.current?.click()}>
              选择文件夹
            </button>
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".md,.markdown,.jpg,.jpeg,.png,.webp,.gif,.mp4,.mov,.mkv,.webm"
            onChange={(event) => event.target.files && addFiles(event.target.files)}
            hidden
          />
          <input
            ref={folderInput}
            type="file"
            multiple
            {...folderInputProps}
            onChange={(event) => event.target.files && addFiles(event.target.files)}
            hidden
          />
        </div>

        <aside className="format-panel">
          <span className="eyebrow">SUPPORTED</span>
          <h3>支持格式</h3>
          <div>
            <span>MD</span>
            <p>
              <strong>Markdown</strong>
              <small>.md · .markdown</small>
            </p>
          </div>
          <div>
            <span>IMG</span>
            <p>
              <strong>图片</strong>
              <small>.jpg · .png · .webp · .gif</small>
            </p>
          </div>
          <div>
            <span>VID</span>
            <p>
              <strong>视频</strong>
              <small>.mp4 · .mov · .mkv · .webm</small>
            </p>
          </div>
          <footer>
            单文件建议不超过 2 GB
            <br />
            UTF-8 Markdown 获得最佳效果
          </footer>
        </aside>
      </div>

      <section className="file-queue">
        <header>
          <div>
            <span className="eyebrow">QUEUE / RELATIVE PATH</span>
            <h2>导入清单</h2>
          </div>
          <button onClick={() => setFiles(EXAMPLE_FILES)}>载入示例批次</button>
        </header>
        <div className="data-table">
          <div className="data-row data-head">
            <span>类型</span>
            <span>文件</span>
            <span>相对路径</span>
            <span>大小</span>
            <span />
          </div>
          {files.map((file) => (
            <div className="data-row" key={file.path}>
              <span className={`file-type-mark file-${file.type}`}>
                {file.type.slice(0, 3).toUpperCase()}
              </span>
              <strong>{file.name}</strong>
              <span className="path-cell">{file.path}</span>
              <span>{formatBytes(file.size)}</span>
              <button
                aria-label={`移除 ${file.name}`}
                onClick={() =>
                  setFiles((current) =>
                    current.filter((item) => item.path !== file.path),
                  )
                }
              >
                ×
              </button>
            </div>
          ))}
          {!files.length && <div className="table-empty">尚未选择文件</div>}
        </div>
      </section>

      <div className="import-submit-bar">
        <div>
          <small>WORKSPACE</small>
          <strong>workspace_demo</strong>
          <span>{files.length} 个 Source File 将进入处理队列</span>
        </div>
        <button disabled={!files.length || starting} onClick={startImport}>
          {starting ? "正在创建任务…" : "开始处理"}
          <b>↗</b>
        </button>
      </div>
    </DemoShell>
  );
}
