export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export const WORKSPACE_ID = "workspace_demo";
export const CREATED_BY = "user_demo";

export type AssetEmbeddingState = {
  embedding_type: string;
  status: string;
  model_name: string;
  embedding_revision: number | null;
};

export type MediaPlayback = {
  mode: "derived_clip" | "source_range" | "transcoded_stream";
  url: string;
  mime_type: string;
  start_ms: number | null;
  end_ms: number | null;
  duration_ms: number | null;
  browser_compatible: boolean;
  fallback_url: string | null;
};

export type AssetRecord = {
  asset_id: string;
  workspace_id: string;
  project_id: string;
  source_file_id: string;
  asset_type: "image" | "video_segment" | "audio_segment" | "markdown_block" | "text_block";
  file_name: string;
  file_type: string;
  asset_name: string | null;
  asset_description: string | null;
  asset_features: Record<
    string,
    | string
    | {
        applicability?: "applicable" | "unknown" | "not_applicable";
        items?: Array<{
          subject?: string;
          description: string;
          salience: number | "high" | "medium" | "low";
          status: "observed" | "inferred" | "metadata" | "user_supplied";
          evidence: string[];
          ocr_confidence?: number | null;
        }>;
        // Historical records are normalized by the UI until Understanding reruns.
        value?: string | null;
        status?: string;
        confidence?: number;
        evidence?: string[];
        description?: string | null;
        source_path?: string | null;
      }
  >;
  file_tree_context: string[];
  source_contexts: Array<{
    text?: string;
    relation_type?: string;
    text_block_index?: number | null;
  }>;
  file_info: Record<string, unknown>;
  source_locator: Record<string, unknown>;
  raw_content: string | null;
  processing_status: string;
  feature_revision: number;
  embedding_revision: number;
  error_message: string | null;
  preview_url: string | null;
  content_url: string | null;
  playback: MediaPlayback | null;
  source_file: {
    source_file_id: string;
    original_file_name: string;
    relative_path: string;
    file_type: string;
    mime_type: string;
    file_size_bytes: number;
    processing_status: string;
    error_message: string | null;
  };
  embeddings: AssetEmbeddingState[];
  created_at: string;
  updated_at: string;
};

export type ProcessingJob = {
  job_id: string;
  workspace_id: string;
  input_path: string;
  total_count: number;
  completed_count: number;
  failed_count: number;
  status: string;
  current_stage: string;
  error_info: Array<{
    asset_id?: string;
    relative_path?: string;
    stage?: string;
    error?: string;
  }>;
  stage_durations_ms: Record<string, number>;
  started_at: string | null;
  completed_at: string | null;
};

export type SearchCapsule = {
  capsule_id: string;
  workspace_id: string;
  created_by: string;
  query_type: string;
  query_text: string | null;
  query_image_uri: string | null;
  fusion_method: string;
  is_favorite: boolean;
  result_count: number;
  last_used_at: string;
  created_at: string;
};

export function endpoint(path: string) {
  return `${API_BASE_URL.replace(/\/$/, "")}${path}`;
}

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(endpoint(path), {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: string | { code?: string; message?: string } }
      | null;
    const message =
      typeof payload?.detail === "string"
        ? payload.detail
        : payload?.detail?.message;
    const code =
      typeof payload?.detail === "object" ? payload.detail?.code : undefined;
    throw new ApiRequestError(
      message || `请求失败（${response.status}）`,
      response.status,
      code,
    );
  }
  return (await response.json()) as T;
}

export async function loadAssets(params: URLSearchParams) {
  return apiFetch<{
    items: AssetRecord[];
    total: number;
    limit: number;
    offset: number;
  }>(`/api/v1/assets?${params}`);
}
