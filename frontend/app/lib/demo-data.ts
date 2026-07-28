export type DemoAssetType = "image" | "video_segment" | "markdown_block";

export type DemoFeature = {
  key: string;
  label: string;
  value: string;
  status: "observed" | "inferred" | "user_supplied" | "metadata";
  confidence: number;
  evidence: string[];
};

export type DemoAsset = {
  id: string;
  type: DemoAssetType;
  name: string;
  description: string;
  preview: string | null;
  sourceFile: string;
  sourcePath: string;
  fileType: string;
  status: "completed" | "processing" | "partial_failed";
  locator: string;
  sourceContext: string;
  cluster: string | null;
  favorite: boolean;
  features: DemoFeature[];
  embeddings: Array<{ type: string; status: "indexed" | "pending"; revision: number }>;
};

const sharedFeatures: DemoFeature[] = [
  {
    key: "subject_content",
    label: "主体内容",
    value: "人物、城市街道、自行车与树影",
    status: "observed",
    confidence: 0.96,
    evidence: ["画面中心存在人物", "右侧可见自行车轮廓"],
  },
  {
    key: "scene_theme",
    label: "场景主题",
    value: "日常生活中的黄昏停留",
    status: "inferred",
    confidence: 0.88,
    evidence: ["低角度暖光", "人物姿态静止"],
  },
  {
    key: "visual_style",
    label: "视觉风格",
    value: "日系动画电影感，柔和颗粒",
    status: "user_supplied",
    confidence: 1,
    evidence: ["用户于 2026-07-28 修改"],
  },
  {
    key: "color_composition",
    label: "色彩构图",
    value: "暖金与蓝紫互补，人物位于右侧三分线",
    status: "observed",
    confidence: 0.91,
    evidence: ["主色为金橙", "天空区域呈低饱和蓝紫"],
  },
  {
    key: "mood_atmosphere",
    label: "情绪氛围",
    value: "安静、怀旧，略带离别感",
    status: "inferred",
    confidence: 0.84,
    evidence: ["长阴影", "空旷街道", "人物背向镜头"],
  },
];

export const DEMO_ASSETS: DemoAsset[] = [
  {
    id: "asset_twilight_01",
    type: "image",
    name: "午后，黄昏将至",
    description:
      "暖金色斜阳穿过街道树冠，女孩在自行车旁停留。画面具有动画电影般的叙事感。",
    preview:
      "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=82",
    sourceFile: "2026-夏日情绪板.md",
    sourcePath: "灵感库/视觉参考/2026-夏日情绪板.md",
    fileType: "markdown",
    status: "completed",
    locator: "Block 13 · 图片 01",
    sourceContext:
      "午后-黄昏：想收集一些日常与旅行交界处的光线，安静，但有故事正在发生。",
    cluster: "cluster_twilight_story",
    favorite: true,
    features: sharedFeatures,
    embeddings: [
      { type: "native_multimodal", status: "indexed", revision: 2 },
      { type: "asset_description", status: "indexed", revision: 2 },
      { type: "subject_content", status: "indexed", revision: 2 },
      { type: "visual_style", status: "indexed", revision: 3 },
    ],
  },
  {
    id: "asset_field_02",
    type: "video_segment",
    name: "麦田里的归途",
    description:
      "远山与麦田被低角度阳光切开，人物牵马缓慢穿过画面，色彩偏青绿与金黄。",
    preview:
      "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=82",
    sourceFile: "田野参考.mp4",
    sourcePath: "项目/短片A/参考/田野参考.mp4",
    fileType: "video",
    status: "completed",
    locator: "00:42.300 — 00:49.800",
    sourceContext: "这一组的核心不是落日，而是傍晚时人物和环境之间的尺度关系。",
    cluster: "cluster_twilight_story",
    favorite: false,
    features: [
      { ...sharedFeatures[0], value: "人物、马匹、金色田野" },
      { ...sharedFeatures[1], value: "田野归途与远山" },
      { ...sharedFeatures[2], value: "宽银幕动画电影感" },
      { ...sharedFeatures[3], value: "青绿与金黄，大面积横向留白" },
      { ...sharedFeatures[4], value: "平静、辽阔、温暖" },
    ],
    embeddings: [
      { type: "native_multimodal", status: "indexed", revision: 1 },
      { type: "asset_description", status: "indexed", revision: 1 },
      { type: "scene_theme", status: "indexed", revision: 1 },
    ],
  },
  {
    id: "asset_city_03",
    type: "image",
    name: "城市最后一束光",
    description:
      "高架桥上的人物望向被夕光覆盖的城市，宽银幕构图与大面积天空强化孤独感。",
    preview:
      "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=1200&q=82",
    sourceFile: "2026-夏日情绪板.md",
    sourcePath: "灵感库/视觉参考/2026-夏日情绪板.md",
    fileType: "markdown",
    status: "completed",
    locator: "Block 32 · 图片 04",
    sourceContext: "城市部分需要留出足够天空，不要拥挤，人物只是画面中的小锚点。",
    cluster: "cluster_city_solitude",
    favorite: true,
    features: [
      { ...sharedFeatures[0], value: "城市、高架桥、远处人物" },
      { ...sharedFeatures[1], value: "城市边缘与暮色" },
      { ...sharedFeatures[2], value: "宽银幕、逆光、城市远景" },
      { ...sharedFeatures[3], value: "大面积天空，中心透视" },
      { ...sharedFeatures[4], value: "孤独、克制、等待" },
    ],
    embeddings: [
      { type: "native_multimodal", status: "indexed", revision: 1 },
      { type: "asset_description", status: "indexed", revision: 1 },
      { type: "mood_atmosphere", status: "indexed", revision: 1 },
    ],
  },
  {
    id: "asset_notes_04",
    type: "markdown_block",
    name: "黄昏氛围关键词",
    description: "关于暖色暮光、蓝调时刻、长阴影和克制叙事的创作笔记。",
    preview: null,
    sourceFile: "导演阐述.md",
    sourcePath: "项目/短片A/导演阐述.md",
    fileType: "markdown",
    status: "completed",
    locator: "光线与时间 / Block 8",
    sourceContext:
      "黄昏不是橙色滤镜。它更像两个时间系统短暂重叠：街灯开始亮，天空还没有完全暗。",
    cluster: "cluster_twilight_notes",
    favorite: false,
    features: [
      { ...sharedFeatures[0], value: "黄昏光线与叙事关键词" },
      { ...sharedFeatures[1], value: "导演阐述与视觉原则" },
      { ...sharedFeatures[2], value: "克制的文字笔记" },
      { ...sharedFeatures[3], value: "暖橙与蓝调时刻" },
      { ...sharedFeatures[4], value: "克制、短暂、时间交叠" },
    ],
    embeddings: [
      { type: "native_multimodal", status: "indexed", revision: 1 },
      { type: "asset_description", status: "indexed", revision: 1 },
    ],
  },
  {
    id: "asset_seaside_05",
    type: "video_segment",
    name: "潮汐与飞鸟",
    description: "人物沿潮间带奔跑，夕阳在水面形成长条高光，飞鸟从镜头前掠过。",
    preview:
      "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=82",
    sourceFile: "海边采风.mov",
    sourcePath: "项目/短片A/采风/海边采风.mov",
    fileType: "video",
    status: "processing",
    locator: "01:18.000 — 01:25.600",
    sourceContext: "海边段落需要速度感，但整体情绪仍然柔和。",
    cluster: null,
    favorite: false,
    features: [
      { ...sharedFeatures[0], value: "奔跑人物、潮汐、飞鸟" },
      { ...sharedFeatures[1], value: "海边采风" },
      { ...sharedFeatures[2], value: "自然光写实电影感" },
      { ...sharedFeatures[3], value: "橙金反光与深蓝水面" },
      { ...sharedFeatures[4], value: "自由、轻盈、柔和" },
    ],
    embeddings: [
      { type: "native_multimodal", status: "pending", revision: 1 },
      { type: "asset_description", status: "indexed", revision: 1 },
    ],
  },
  {
    id: "asset_reference_06",
    type: "image",
    name: "逆光人像参考",
    description: "人物侧脸处于强逆光中，轮廓边缘呈现金色光晕。",
    preview:
      "https://images.unsplash.com/photo-1496440737103-cd596325d314?auto=format&fit=crop&w=1200&q=82",
    sourceFile: "角色光线参考.zip",
    sourcePath: "参考库/角色/角色光线参考.zip",
    fileType: "image",
    status: "partial_failed",
    locator: "portrait/07.jpg",
    sourceContext: "角色面部可以暗一些，但轮廓光必须清晰。",
    cluster: "cluster_portrait_light",
    favorite: false,
    features: [
      { ...sharedFeatures[0], value: "人物侧脸与轮廓光" },
      { ...sharedFeatures[1], value: "人像光线参考" },
      { ...sharedFeatures[2], value: "写实摄影" },
      { ...sharedFeatures[3], value: "暗部占比高，金色边缘光" },
      { ...sharedFeatures[4], value: "私密、坚定" },
    ],
    embeddings: [
      { type: "native_multimodal", status: "indexed", revision: 1 },
      { type: "rights_version_authorship", status: "pending", revision: 1 },
    ],
  },
];

export const DEMO_TASKS = [
  {
    id: "job_20260728_1642",
    name: "夏日情绪板与采风素材",
    status: "processing",
    stage: "Embedding",
    progress: 72,
    sourceFiles: 18,
    assets: 126,
    markdownBlocks: 48,
    videoSegments: 31,
    modelCalls: 184,
    succeeded: 171,
    failed: 3,
    createdAt: "今天 16:42",
  },
  {
    id: "job_20260727_1018",
    name: "短片 A 参考库",
    status: "completed",
    stage: "Completed",
    progress: 100,
    sourceFiles: 42,
    assets: 309,
    markdownBlocks: 87,
    videoSegments: 96,
    modelCalls: 511,
    succeeded: 511,
    failed: 0,
    createdAt: "昨天 10:18",
  },
  {
    id: "job_20260726_2130",
    name: "角色光线参考",
    status: "partial_failed",
    stage: "Understanding",
    progress: 88,
    sourceFiles: 12,
    assets: 74,
    markdownBlocks: 0,
    videoSegments: 0,
    modelCalls: 96,
    succeeded: 91,
    failed: 5,
    createdAt: "7 月 26 日 21:30",
  },
];

export const DEMO_CLUSTERS = [
  {
    id: "cluster_twilight_story",
    name: "黄昏中的叙事瞬间",
    description: "人物与环境在日落前后的短暂交汇，具有安静而明确的故事感。",
    embeddingType: "native_multimodal",
    members: 26,
    probability: 0.89,
    favorite: true,
    assetIds: ["asset_twilight_01", "asset_field_02", "asset_city_03"],
  },
  {
    id: "cluster_city_solitude",
    name: "城市边缘与独处",
    description: "大尺度城市空间中的小人物，强调留白、逆光与克制情绪。",
    embeddingType: "mood_atmosphere",
    members: 18,
    probability: 0.84,
    favorite: false,
    assetIds: ["asset_city_03", "asset_reference_06"],
  },
  {
    id: "cluster_twilight_notes",
    name: "蓝调时刻创作笔记",
    description: "围绕暮光、街灯、长阴影与时间交叠的文字资产。",
    embeddingType: "asset_description",
    members: 14,
    probability: 0.93,
    favorite: false,
    assetIds: ["asset_notes_04"],
  },
  {
    id: "cluster_portrait_light",
    name: "金色轮廓光人像",
    description: "强逆光、暗部人脸与金色边缘光构成的角色参考。",
    embeddingType: "visual_style",
    members: 11,
    probability: 0.81,
    favorite: true,
    assetIds: ["asset_reference_06"],
  },
];
