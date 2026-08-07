"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, WORKSPACE_ID } from "./api";

export const CURRENT_WORKSPACE_STORAGE_KEY = "capsule.currentWorkspaceId";

export type WorkspaceRecord = {
  workspace_id: string;
  name: string;
  created_at: string;
  updated_at: string;
};

function storeWorkspaceId(workspaceId: string) {
  window.localStorage.setItem(CURRENT_WORKSPACE_STORAGE_KEY, workspaceId);
}

export function useWorkspaceSelection() {
  const [workspaceId, setWorkspaceIdState] = useState(WORKSPACE_ID);
  const [workspaces, setWorkspaces] = useState<WorkspaceRecord[]>([]);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const setWorkspaceId = useCallback((nextWorkspaceId: string) => {
    if (!nextWorkspaceId) return;
    setWorkspaceIdState(nextWorkspaceId);
    storeWorkspaceId(nextWorkspaceId);
  }, []);

  const refreshWorkspaces = useCallback(async (preferredWorkspaceId?: string) => {
    setLoading(true);
    try {
      const payload = await apiFetch<{ items: WorkspaceRecord[] }>(
        "/api/v1/workspaces",
      );
      setWorkspaces(payload.items);
      setWorkspaceIdState((current) => {
        const preferred = preferredWorkspaceId?.trim() || current;
        const next = payload.items.some(
          (workspace) => workspace.workspace_id === preferred,
        )
          ? preferred
          : payload.items[0]?.workspace_id ?? WORKSPACE_ID;
        storeWorkspaceId(next);
        return next;
      });
      setError(null);
      return payload.items;
    } catch (requestError) {
      if (preferredWorkspaceId?.trim()) {
        setWorkspaceIdState(preferredWorkspaceId.trim());
      }
      setError(
        requestError instanceof Error
          ? requestError.message
          : "工作空间加载失败",
      );
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const stored = window.localStorage
      .getItem(CURRENT_WORKSPACE_STORAGE_KEY)
      ?.trim();
    const timer = window.setTimeout(() => {
      void refreshWorkspaces(stored).then(() => setReady(true));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refreshWorkspaces]);

  return {
    workspaceId,
    workspaces,
    ready,
    loading,
    error,
    setWorkspaceId,
    refreshWorkspaces,
  };
}

export function WorkspaceSelect({
  workspaceId,
  workspaces,
  loading,
  onChange,
}: {
  workspaceId: string;
  workspaces: WorkspaceRecord[];
  loading: boolean;
  onChange: (workspaceId: string) => void;
}) {
  const selectedExists = workspaces.some(
    (workspace) => workspace.workspace_id === workspaceId,
  );

  return (
    <label className="workspace-switcher">
      <span className="sr-only">切换工作空间</span>
      <select
        aria-label="切换工作空间"
        value={workspaceId}
        disabled={loading && workspaces.length === 0}
        onChange={(event) => onChange(event.target.value)}
      >
        {!selectedExists && (
          <option value={workspaceId}>
            {loading ? "正在读取工作空间…" : workspaceId}
          </option>
        )}
        {workspaces.map((workspace) => (
          <option value={workspace.workspace_id} key={workspace.workspace_id}>
            {workspace.name && workspace.name !== workspace.workspace_id
              ? `${workspace.name} · ${workspace.workspace_id}`
              : workspace.workspace_id}
          </option>
        ))}
      </select>
    </label>
  );
}
