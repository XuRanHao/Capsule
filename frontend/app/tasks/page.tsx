"use client";

import { useMemo, useState } from "react";
import DemoShell, { StatusBadge } from "../components/DemoShell";
import { DEMO_TASKS } from "../lib/demo-data";

const PIPELINE_STAGES = [
  "Discovery",
  "Segmentation",
  "Understanding",
  "Embedding",
  "Milvus",
  "Completed",
];

const ERRORS = [
  {
    asset: "asset_reference_06",
    stage: "Understanding",
    code: "MODEL_TIMEOUT",
    message: "豆包多模态理解请求超过 180 秒",
    retry: 2,
  },
  {
    asset: "asset_video_044",
    stage: "Segmentation",
    code: "FFMPEG_DECODE",
    message: "源视频第 00:28.400 帧无法解码，已保留其他片段",
    retry: 1,
  },
  {
    asset: "asset_image_088",
    stage: "Embedding",
    code: "RATE_LIMITED",
    message: "Embedding 请求触发限流，等待重试",
    retry: 3,
  },
];

export default function TasksPage() {
  const [selectedId, setSelectedId] = useState(DEMO_TASKS[0].id);
  const [retrying, setRetrying] = useState<string | null>(null);
  const selected =
    DEMO_TASKS.find((task) => task.id === selectedId) ?? DEMO_TASKS[0];
  const currentStage = PIPELINE_STAGES.indexOf(selected.stage);
  const elapsed = useMemo(
    () => (selected.status === "processing" ? "08:42" : "12:18"),
    [selected.status],
  );

  const retry = (assetId: string) => {
    setRetrying(assetId);
    window.setTimeout(() => setRetrying(null), 900);
  };

  return (
    <DemoShell
      active="tasks"
      eyebrow="PROCESSING / 44.2"
      title="每一步，都看得见。"
      description="跟踪从 Source File 到 Asset、理解、Embedding 和入库的完整异步任务。"
      actions={
        <>
          <button className="secondary-action">仅看失败</button>
          <button className="primary-action">新建导入</button>
        </>
      }
    >
      <div className="task-layout">
        <aside className="task-list">
          <header>
            <span>PROCESSING JOBS</span>
            <b>{DEMO_TASKS.length}</b>
          </header>
          {DEMO_TASKS.map((task) => (
            <button
              className={selectedId === task.id ? "active" : ""}
              onClick={() => setSelectedId(task.id)}
              key={task.id}
            >
              <div>
                <StatusBadge status={task.status} />
                <small>{task.createdAt}</small>
              </div>
              <strong>{task.name}</strong>
              <span>{task.stage}</span>
              <div className="task-progress">
                <i style={{ width: `${task.progress}%` }} />
              </div>
              <footer>
                <span>{task.progress}%</span>
                <span>{task.assets} Assets</span>
              </footer>
            </button>
          ))}
        </aside>

        <section className="task-detail">
          <header>
            <div>
              <small>{selected.id}</small>
              <h2>{selected.name}</h2>
              <StatusBadge status={selected.status} />
            </div>
            <div className="task-clock">
              <small>ELAPSED</small>
              <strong>{elapsed}</strong>
            </div>
          </header>

          <div className="pipeline-timeline">
            {PIPELINE_STAGES.map((stage, index) => {
              const completed =
                selected.status === "completed" || index < currentStage;
              const active = index === currentStage;
              return (
                <div
                  className={`${completed ? "completed" : ""} ${
                    active ? "active" : ""
                  }`}
                  key={stage}
                >
                  <span>{completed ? "✓" : String(index + 1)}</span>
                  <strong>{stage}</strong>
                  <small>
                    {active
                      ? `${selected.progress}%`
                      : completed
                        ? "完成"
                        : "等待"}
                  </small>
                </div>
              );
            })}
          </div>

          <div className="task-metrics">
            <article>
              <small>SOURCE FILE</small>
              <strong>{selected.sourceFiles}</strong>
              <span>已发现文件</span>
            </article>
            <article>
              <small>ALL ASSETS</small>
              <strong>{selected.assets}</strong>
              <span>语义资产</span>
            </article>
            <article>
              <small>MARKDOWN</small>
              <strong>{selected.markdownBlocks}</strong>
              <span>Blocks</span>
            </article>
            <article>
              <small>VIDEO</small>
              <strong>{selected.videoSegments}</strong>
              <span>Segments</span>
            </article>
            <article className="metric-accent">
              <small>MODEL CALLS</small>
              <strong>{selected.modelCalls}</strong>
              <span>{selected.succeeded} 成功</span>
            </article>
            <article className={selected.failed ? "metric-error" : ""}>
              <small>FAILED</small>
              <strong>{selected.failed}</strong>
              <span>需要检查</span>
            </article>
          </div>

          <section className="stage-console">
            <header>
              <div>
                <span className="eyebrow">CURRENT STAGE</span>
                <h3>{selected.stage}</h3>
              </div>
              <span className="live-pulse">
                <i />
                {selected.status === "processing" ? "LIVE" : "SNAPSHOT"}
              </span>
            </header>
            <div className="console-lines">
              <p>
                <span>16:49:02</span>
                embedding batch 12/18 dispatched · concurrency=16
              </p>
              <p>
                <span>16:49:03</span>
                indexed 32 vectors into asset_embeddings_seed16_1024
              </p>
              <p>
                <span>16:49:04</span>
                asset_image_067 · visual_style · revision 1 completed
              </p>
              <p className="console-active">
                <span>16:49:05</span>
                waiting for 8 in-flight model calls…
              </p>
            </div>
          </section>

          <section className="error-table">
            <header>
              <div>
                <span className="eyebrow">ERROR DETAILS</span>
                <h3>失败与重试</h3>
              </div>
              <button>全部重试</button>
            </header>
            <div className="data-table">
              <div className="data-row error-row data-head">
                <span>Asset</span>
                <span>阶段</span>
                <span>错误</span>
                <span>重试</span>
              </div>
              {ERRORS.slice(0, Math.max(1, selected.failed)).map((error) => (
                <div className="data-row error-row" key={error.asset}>
                  <strong>{error.asset}</strong>
                  <span>{error.stage}</span>
                  <span>
                    <b>{error.code}</b>
                    <small>{error.message}</small>
                  </span>
                  <button
                    disabled={retrying === error.asset}
                    onClick={() => retry(error.asset)}
                  >
                    {retrying === error.asset
                      ? "重试中…"
                      : `重试 · ${error.retry}`}
                  </button>
                </div>
              ))}
            </div>
          </section>
        </section>
      </div>
    </DemoShell>
  );
}
