import Link from "next/link";
import type { ReactNode } from "react";

export type DemoSection =
  | "import"
  | "tasks"
  | "assets"
  | "clusters"
  | "search"
  | "capsules";

const NAV_ITEMS: Array<{
  id: DemoSection;
  href: string;
  label: string;
  marker: string;
}> = [
  { id: "import", href: "/import", label: "导入", marker: "01" },
  { id: "tasks", href: "/tasks", label: "处理任务", marker: "02" },
  { id: "assets", href: "/assets", label: "Assets", marker: "03" },
  { id: "clusters", href: "/clusters", label: "Cluster", marker: "04" },
  { id: "search", href: "/search", label: "搜索", marker: "05" },
  { id: "capsules", href: "/capsules", label: "Capsule", marker: "06" },
];

export function AppNavigation({
  active,
  compact = false,
}: {
  active: DemoSection;
  compact?: boolean;
}) {
  return (
    <nav
      className={compact ? "product-nav product-nav-compact" : "product-nav"}
      aria-label="Capsule 功能导航"
    >
      {NAV_ITEMS.map((item) => (
        <Link
          href={item.href}
          className={active === item.id ? "active" : ""}
          key={item.id}
        >
          <small>{item.marker}</small>
          <span>{item.label}</span>
        </Link>
      ))}
    </nav>
  );
}

export default function DemoShell({
  active,
  eyebrow,
  title,
  description,
  actions,
  children,
}: {
  active: DemoSection;
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <main className="demo-app-shell">
      <header className="demo-topbar">
        <Link className="brand" href="/search" aria-label="Capsule 搜索">
          <span className="brand-orbit" aria-hidden="true">
            <i />
          </span>
          <strong>CAPSULE</strong>
          <em>个人多模态素材库</em>
        </Link>
        <div className="demo-topbar-meta">
          <span className="connection-dot live" />
          <span>Workspace Demo</span>
          <b>许然豪</b>
        </div>
      </header>
      <div className="demo-layout">
        <aside className="demo-sidebar">
          <AppNavigation active={active} />
          <div className="sidebar-footnote">
            <span>POC BUILD</span>
            <strong>2026.07 / V2</strong>
            <small>PostgreSQL · Milvus · Seed</small>
          </div>
        </aside>
        <section className="demo-main">
          <header className="demo-page-header">
            <div>
              <span className="eyebrow">{eyebrow}</span>
              <h1>{title}</h1>
              <p>{description}</p>
            </div>
            {actions && <div className="page-actions">{actions}</div>}
          </header>
          {children}
        </section>
      </div>
    </main>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const labels: Record<string, string> = {
    completed: "已完成",
    processing: "处理中",
    partial_failed: "部分失败",
    pending: "等待中",
    indexed: "已入库",
    observed: "观察值",
    inferred: "模型推断",
    user_supplied: "用户修改",
    metadata: "元数据",
  };
  return (
    <span className={`status-badge status-${status}`}>
      <i />
      {labels[status] ?? status}
    </span>
  );
}

export function AssetThumb({
  preview,
  name,
  type,
}: {
  preview: string | null;
  name: string;
  type: string;
}) {
  return (
    <div className="asset-thumb">
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt={name} />
      ) : (
        <div className="asset-thumb-placeholder">
          <span>{type === "markdown_block" ? "¶" : "C"}</span>
        </div>
      )}
      <small>{type.replace("_", " ")}</small>
    </div>
  );
}
