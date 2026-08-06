"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import DemoShell, { StatusBadge } from "../components/DemoShell";
import {
  type ProcessingJob,
  WORKSPACE_ID,
  apiFetch,
} from "../lib/api";

const PIPELINE_STAGES = [
  "discovering",
  "parsing",
  "segmenting",
  "asset_stored",
  "understanding",
  "feature_ready",
  "embedding",
  "indexing",
  "completed",
];

const STAGE_LABELS: Record<string, string> = {
  discovering: "Discovery",
  parsing: "Parsing",
  segmenting: "Segmentation",
  asset_stored: "Asset Stored",
  understanding: "Understanding",
  feature_ready: "Feature Ready",
  embedding: "Embedding",
  indexing: "Milvus",
  completed: "Completed",
  failed: "Failed",
};

function progress(job: ProcessingJob) {
  if (job.status === "completed") return 100;
  if (job.status === "failed") return 100;
  const stageIndex = Math.max(0, PIPELINE_STAGES.indexOf(job.current_stage));
  const stageProgress = (stageIndex / (PIPELINE_STAGES.length - 1)) * 100;
  const fileProgress = job.total_count
    ? (job.completed_count / job.total_count) * 40
    : 0;
  return Math.min(99, Math.round(Math.max(stageProgress, fileProgress)));
}

function elapsedMilliseconds(job: ProcessingJob) {
  if (!job.started_at) return null;
  const end = job.completed_at ? new Date(job.completed_at) : new Date();
  return Math.max(0, end.getTime() - new Date(job.started_at).getTime());
}

function formatDuration(durationMs: number | null | undefined) {
  if (durationMs == null) return "未记录";
  if (durationMs < 10) return "< 0.01s";
  if (durationMs < 1000) return `${Math.round(durationMs)}ms`;
  const seconds = durationMs / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60);
  return `${minutes}m ${String(remaining).padStart(2, "0")}s`;
}

export default function TasksPage() {
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clearMessage, setClearMessage] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  const load = useCallback(async () => {
    try {
      const payload = await apiFetch<{ items: ProcessingJob[] }>(
        `/api/v1/import-jobs?workspace_id=${WORKSPACE_ID}&limit=100`,
      );
      setJobs(payload.items);
      const requested =
        typeof window !== "undefined"
          ? new URLSearchParams(window.location.search).get("job_id")
          : null;
      setSelectedId((current) => {
        if (current && payload.items.some((job) => job.job_id === current)) {
          return current;
        }
        if (requested && payload.items.some((job) => job.job_id === requested)) {
          return requested;
        }
        return payload.items[0]?.job_id ?? null;
      });
      setError(null);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "任务加载失败",
      );
    }
  }, []);

  const clearJobs = async () => {
    if (
      !window.confirm(
        "强制清空全部处理任务？正在上传或运行的任务会立即停止，但已生成的素材与聚类结果不会删除。",
      )
    ) {
      return;
    }
    setClearing(true);
    setClearMessage(null);
    try {
      const result = await apiFetch<{
        deleted_count: number;
        cancelled_count: number;
      }>(`/api/v1/import-jobs?workspace_id=${WORKSPACE_ID}`, {
        method: "DELETE",
      });
      setClearMessage(
        result.cancelled_count
          ? `已停止 ${result.cancelled_count} 个运行任务，并清空 ${result.deleted_count} 条任务记录。`
          : `已清空 ${result.deleted_count} 个处理任务。`,
      );
      await load();
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "清空任务失败",
      );
    } finally {
      setClearing(false);
    }
  };

  useEffect(() => {
    const initial = window.setTimeout(() => void load(), 0);
    const polling = window.setInterval(() => void load(), 2000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(polling);
    };
  }, [load]);

  const selected = jobs.find((job) => job.job_id === selectedId) ?? jobs[0];
  const effectiveStage =
    selected?.status === "completed" ? "completed" : selected?.current_stage;
  const currentStage = effectiveStage
    ? PIPELINE_STAGES.indexOf(effectiveStage)
    : -1;
  const elapsed = useMemo(() => {
    if (!selected) return "—";
    const durationMs = elapsedMilliseconds(selected);
    if (durationMs == null) return "—";
    const seconds = Math.round(durationMs / 1000);
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(
      seconds % 60,
    ).padStart(2, "0")}`;
  }, [selected]);

  return (
    <DemoShell
      active="tasks"
      eyebrow="PROCESSING / LIVE"
      title="每一步，都看得见。"
      description="实时读取导入 Job，展示解析、理解、Embedding 和 Milvus 入库阶段。"
      actions={
        <>
          <button
            className="secondary-action"
            disabled={clearing || jobs.length === 0}
            onClick={() => void clearJobs()}
          >
            {clearing ? "正在清空…" : "清空处理任务"}
          </button>
          <Link className="primary-action button-link" href="/import">
            新建导入
          </Link>
        </>
      }
    >
      {clearMessage && <div className="task-clear-message">{clearMessage}</div>}
      {error && (
        <div className="asset-empty">
          <strong>无法读取任务</strong>
          <span>{error}</span>
        </div>
      )}
      {!error && !selected && (
        <div className="asset-empty">
          <strong>还没有处理任务</strong>
          <span>从导入页选择一个文件夹开始。</span>
        </div>
      )}
      {!error && selected && (
        <div className="task-layout">
          <aside className="task-list">
            <header>
              <span>PROCESSING JOBS</span>
              <b>{jobs.length}</b>
            </header>
            {jobs.map((job) => (
              <button
                className={selected.job_id === job.job_id ? "active" : ""}
                onClick={() => setSelectedId(job.job_id)}
                key={job.job_id}
              >
                <div>
                  <StatusBadge status={job.status} />
                  <small>
                    {job.started_at
                      ? new Date(job.started_at).toLocaleString("zh-CN")
                      : "等待上传"}
                  </small>
                </div>
                <strong>{job.input_path.split("/").at(-1)}</strong>
                <span>{STAGE_LABELS[job.current_stage] || job.current_stage}</span>
                <div className="task-progress">
                  <i style={{ width: `${progress(job)}%` }} />
                </div>
                <footer>
                  <span>{progress(job)}%</span>
                  <span>
                    {job.completed_count}/{job.total_count} FILES
                  </span>
                </footer>
              </button>
            ))}
          </aside>

          <section className="task-detail">
            <header>
              <div>
                <small>{selected.job_id}</small>
                <h2>{selected.input_path.split("/").at(-1)}</h2>
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
                const active = selected.status === "running" && index === currentStage;
                const duration =
                  stage === "completed"
                    ? elapsedMilliseconds(selected)
                    : selected.stage_durations_ms?.[stage];
                return (
                  <div
                    className={`${completed ? "completed" : ""} ${
                      active ? "active" : ""
                    }`}
                    key={stage}
                  >
                    <span>{completed ? "✓" : String(index + 1)}</span>
                    <strong>{STAGE_LABELS[stage]}</strong>
                    <small>
                      {active && selected.status === "running"
                        ? "计时中…"
                        : completed
                          ? duration == null
                            ? "已跳过"
                            : formatDuration(duration)
                          : "等待"}
                    </small>
                  </div>
                );
              })}
            </div>

            <div className="task-metrics">
              <article>
                <small>SOURCE FILE</small>
                <strong>{selected.total_count}</strong>
                <span>已发现文件</span>
              </article>
              <article>
                <small>PROCESSED</small>
                <strong>{selected.completed_count}</strong>
                <span>完成解析</span>
              </article>
              <article>
                <small>STAGE</small>
                <strong>{currentStage + 1}</strong>
                <span>{STAGE_LABELS[effectiveStage || selected.current_stage]}</span>
              </article>
              <article>
                <small>ERRORS</small>
                <strong>{selected.error_info.length}</strong>
                <span>阶段错误</span>
              </article>
              <article className="metric-accent">
                <small>PIPELINE</small>
                <strong>{progress(selected)}%</strong>
                <span>真实进度</span>
              </article>
              <article className={selected.failed_count ? "metric-error" : ""}>
                <small>FAILED</small>
                <strong>{selected.failed_count}</strong>
                <span>文件失败</span>
              </article>
            </div>

            <section className="stage-console">
              <header>
                <div>
                  <span className="eyebrow">CURRENT STAGE</span>
                  <h3>
                    {STAGE_LABELS[effectiveStage || selected.current_stage] ||
                      effectiveStage ||
                      selected.current_stage}
                  </h3>
                </div>
                <span className="live-pulse">
                  <i />
                  {selected.status === "running" ? "LIVE" : "SNAPSHOT"}
                </span>
              </header>
              <div className="console-lines">
                <p>
                  <span>JOB</span>
                  {selected.job_id}
                </p>
                <p>
                  <span>FILES</span>
                  {selected.completed_count} completed · {selected.failed_count} failed
                </p>
                <p className="console-active">
                  <span>STAGE</span>
                  {STAGE_LABELS[effectiveStage || selected.current_stage] ||
                    effectiveStage ||
                    selected.current_stage}
                </p>
                <p>
                  <span>DURATION</span>
                  {selected.status === "running"
                    ? "计时中…"
                    : formatDuration(
                        effectiveStage === "completed"
                          ? elapsedMilliseconds(selected)
                          : selected.stage_durations_ms?.[
                              effectiveStage || selected.current_stage
                            ],
                      )}
                </p>
              </div>
            </section>

            <section className="error-table">
              <header>
                <div>
                  <span className="eyebrow">ERROR DETAILS</span>
                  <h3>失败记录</h3>
                </div>
              </header>
              <div className="data-table">
                <div className="data-row error-row data-head">
                  <span>Asset / File</span>
                  <span>阶段</span>
                  <span>错误</span>
                  <span>状态</span>
                </div>
                {selected.error_info.map((item, index) => (
                  <div className="data-row error-row" key={`${item.asset_id}-${index}`}>
                    <strong>{item.asset_id || item.relative_path || "Job"}</strong>
                    <span>{item.stage || "assetization"}</span>
                    <span>
                      <small>{item.error || "未知错误"}</small>
                    </span>
                    <StatusBadge status="failed" />
                  </div>
                ))}
                {!selected.error_info.length && (
                  <div className="asset-empty">
                    <strong>没有错误</strong>
                    <span>当前任务各阶段运行正常。</span>
                  </div>
                )}
              </div>
            </section>
          </section>
        </div>
      )}
    </DemoShell>
  );
}
