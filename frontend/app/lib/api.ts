export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";

export const WORKSPACE_ID = "workspace_demo";
export const CREATED_BY = "user_demo";

export type AssetEmbeddingState = {
  embedding_type: string;
  status: string;
  model_name: string;
  embedding_revision: number | null;
};

export type AssetRecord = {
  asset_id: string;
  workspace_id: string;
  project_id: string;
  source_file_id: string;
  asset_type: "image" | "video_segment" | "markdown_block" | "text_block";
  file_name: string;
  file_type: string;
  asset_name: string | null;
  asset_description: string | null;
  asset_features: Record<
    string,
    | string
    | {
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

export type ClusterRun = {
  cluster_run_id: string;
  workspace_id: string;
  embedding_type: string;
  input_embedding_ids: string[];
  dataset_hash: string;
  sample_count: number;
  preprocessing: Record<string, unknown>;
  parameters: Record<string, unknown>;
  cluster_count: number | null;
  noise_count: number | null;
  noise_ratio: number | null;
  status: string;
  started_at: string | null;
  completed_at: string | null;
};

export type ClusterCapsule = {
  cluster_capsule_id: string;
  cluster_run_id: string;
  workspace_id: string;
  embedding_type: string;
  cluster_label: number;
  model_generated_name: string;
  user_override_name: string | null;
  effective_name: string;
  model_generated_description: string;
  user_override_description: string | null;
  effective_description: string;
  keywords: string[];
  common_features: string[];
  internal_variance: string | null;
  member_count: number;
  average_membership_probability: number;
  medoid_asset_id: string | null;
  representative_asset_ids: string[];
  is_favorite: boolean;
};

export type ClusterMember = {
  asset_id: string;
  asset_type: AssetRecord["asset_type"];
  file_name: string;
  asset_name: string | null;
  asset_description: string | null;
  source_file_id: string;
  relative_path: string;
  hdbscan_label: number;
  membership_probability: number;
  is_noise: boolean;
  distance_to_representative: number | null;
  preview_url: string | null;
};

export type CurrentClusterMode =
  | "dynamic"
  | "resident_open"
  | "resident_manual";

export type CurrentCluster = {
  cluster_id: string;
  workspace_id: string;
  embedding_type: string;
  mode: CurrentClusterMode;
  name: string;
  description: string;
  representative_asset_id: string | null;
  source_run_id: string | null;
  created_at: string;
  updated_at: string;
};

export type CurrentClusterMember = {
  cluster_id: string;
  asset_id: string;
  embedding_type: string;
  source: "full_cluster" | "incremental" | "user";
  score: number | null;
  created_at: string;
};

export type ClusterAssetStatusItem = {
  asset_id: string;
  asset_type: AssetRecord["asset_type"];
  file_name: string;
  asset_name: string | null;
  status: "incrementally_clustered" | "pending" | "manual_management";
  cluster_id: string | null;
  cluster_name: string | null;
  cluster_mode: CurrentClusterMode | null;
  member_source: CurrentClusterMember["source"] | null;
  score: number | null;
  created_at: string;
};

export type ClusterAssetStatus = {
  workspace_id: string;
  embedding_type: string;
  initialized: boolean;
  bootstrap_minimum_count: number;
  baseline_cluster_run_id: string | null;
  baseline_sample_count: number | null;
  eligible_asset_count: number;
  new_asset_count: number;
  incrementally_clustered_count: number;
  pending_count: number;
  manual_management_count: number;
  items: ClusterAssetStatusItem[];
};

export type SearchCapsule = {
  capsule_id: string;
  workspace_id: string;
  created_by: string;
  query_type: string;
  query_text: string | null;
  query_image_uri: string | null;
  fusion_method: string;
  rerank_method: string;
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
