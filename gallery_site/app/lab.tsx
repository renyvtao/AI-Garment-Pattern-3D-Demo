"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type GarmentType = "upper" | "lower" | "wholebody";
type FilterType = "all" | "separates" | "wholebody";
type BodyGender = "female" | "male" | "neutral";
type GarmentMode = "general" | "mens_suit";
type SemanticProfile = "female" | "male";
type BodyActionId =
  | "none"
  | "official_showcase"
  | "standing_turn"
  | "wave"
  | "walk_in_place";

type CaseItem = {
  id: string;
  garments: GarmentType[];
};

const CASES: CaseItem[] = [
  {
    id: "valid_garment_1aee14a8c7b4d56b4e8b6ddd575d1f561a72fdc75c43a4b6926f1655152193c6",
    garments: ["upper", "lower"],
  },
  {
    id: "valid_garment_1dde6afed43187fe927089a615e3f744724ef3defdf3f2ae4a6cede5ad71dcea",
    garments: ["upper", "lower"],
  },
  {
    id: "valid_garment_62bb809fc2dcd50409cb36163a0eb222f9aa1af0f256a3233b67b3ed4081dc71",
    garments: ["upper", "lower"],
  },
  {
    id: "valid_garment_6fe14e1f646513ee93714fbe8026a84c6a2897be4df2f3c936cb2be8dd2d1762",
    garments: ["wholebody"],
  },
  {
    id: "valid_garment_72b086429d2dfe2a8de6f4403a024b2bb17446021c9e8f9ebacfc7a990ac8434",
    garments: ["wholebody"],
  },
  {
    id: "valid_garment_80141ce740f489f1d2f57a03f32c7577a28b62a6ac790a0d9ed8a18d961c2918",
    garments: ["wholebody"],
  },
  {
    id: "valid_garment_8e3c458da20c290c216813ec07f1a2e8f9cfb4ee7e412a783a238ec353b346a0",
    garments: ["upper", "lower"],
  },
  {
    id: "valid_garment_c2b582eb318455abaf8ed8e3126c1b423ade2704d810f7cd24428febda5632fa",
    garments: ["upper", "lower"],
  },
  {
    id: "valid_garment_d77c6f5d4856831878eadb7fe3c8b180bfa9e9ad4a14936ac10a1697bb3c054f",
    garments: ["upper", "lower"],
  },
  {
    id: "valid_garment_e918651cc154a7570e47d8b8f6c0f0f93cfbb7d5129103a1bacd8299ba945f91",
    garments: ["wholebody"],
  },
];

const garmentLabels: Record<GarmentType, string> = {
  upper: "上装",
  lower: "下装",
  wholebody: "连体装",
};

const SEMANTIC_ATTRIBUTES: Record<
  SemanticProfile,
  { key: string; label: string }[]
> = {
  female: [
    { key: "big", label: "体型较大" },
    { key: "broad_shoulders", label: "宽肩" },
    { key: "feminine", label: "女性化" },
    { key: "large_breasts", label: "胸部丰满" },
    { key: "long_legs", label: "腿长" },
    { key: "long_neck", label: "颈长" },
    { key: "long_torso", label: "躯干长" },
    { key: "muscular", label: "肌肉感" },
    { key: "pear_shaped", label: "梨形" },
    { key: "petite", label: "娇小" },
    { key: "short", label: "整体偏矮" },
    { key: "short_arms", label: "手臂偏短" },
    { key: "skinny_legs", label: "腿部纤细" },
    { key: "slim_waist", label: "腰部纤细" },
    { key: "tall", label: "整体偏高" },
  ],
  male: [
    { key: "average", label: "体型平均" },
    { key: "big", label: "体型较大" },
    { key: "broad_shoulders", label: "宽肩" },
    { key: "delicate_build", label: "纤细骨架" },
    { key: "long_legs", label: "腿长" },
    { key: "long_neck", label: "颈长" },
    { key: "long_torso", label: "躯干长" },
    { key: "masculine", label: "男性化" },
    { key: "muscular", label: "肌肉感" },
    { key: "rectangular", label: "直筒形" },
    { key: "short", label: "整体偏矮" },
    { key: "short_arms", label: "手臂偏短" },
    { key: "skinny_arms", label: "手臂纤细" },
    { key: "soft_body", label: "体型柔和" },
    { key: "tall", label: "整体偏高" },
  ],
};

type BodySchema = {
  status: "ready" | "waiting_for_weights";
  actions: {
    id: BodyActionId;
    label_zh: string;
    kind: string;
    available: boolean;
  }[];
  checkpoint_routes: {
    model_gender: BodyGender;
    semantic_profile: SemanticProfile;
    variant: string;
    available: boolean;
  }[];
};

type BodyGenerationResult = {
  request_id: string;
  gender: BodyGender;
  action?: {
    id: BodyActionId;
    label_zh: string;
    kind: string;
    enabled: boolean;
  };
  topology: { vertices: number; faces: number; joints: number };
  method: {
    method: string;
    variant?: string;
    measurement_refinement?: {
      enabled: boolean;
      targets: Record<string, number>;
      final: Record<string, number>;
      final_error: Record<string, number>;
      function_evaluations: number;
    };
  };
  downloads: Record<string, string>;
};

type GBTGender = "female" | "male";
type GBTBoundaryMode = "extrapolate" | "clamp";

type GBTSchema = {
  status?: string;
  genders?: GBTGender[];
  required?: string[];
  optional?: string[];
  boundary_policies?: GBTBoundaryMode[];
  out_of_range_modes?: GBTBoundaryMode[];
  templates?: unknown[];
};

type GBTGenerationResult = {
  request_id?: string;
  gender?: GBTGender;
  body_type?: {
    inferred?: string;
    mapped?: string;
    mapping_applied?: boolean;
  };
  base_source_id?: string;
  boundary_policy?: GBTBoundaryMode;
  out_of_range_mode?: GBTBoundaryMode;
  target?: {
    height_cm?: number;
    chest_cm?: number;
    waist_cm?: number;
    hips_cm?: number;
    hips_source?: string;
  };
  source_counts?: Record<string, number>;
  field_source_counts?: Record<string, number>;
  warnings?: string[];
  downloads?: Record<string, string>;
  files?: Record<string, string>;
};

type JobArtifact = {
  name: string;
  path: string;
  category: string;
  bytes: number;
  url: string;
};

type JobRecord = {
  id: string;
  name: string;
  state: "queued" | "running" | "completed" | "failed" | "cancelled" | "trashed";
  step: string;
  progress: number;
  message: string;
  runtime?: {
    kind: "idle" | "cpu" | "gpu" | "mixed" | "finished";
    label: string;
    detail: string;
    gpu_name?: string | null;
    verified: boolean;
  };
  created_at: string;
  updated_at: string;
  error?: string | null;
  cancel_requested: boolean;
  size_bytes: number;
  image_count: number;
  config: {
    garment_mode?: GarmentMode;
    body_mode: "preset" | "custom";
    gender: BodyGender;
    action_id: BodyActionId;
    input_files: string[];
    height_cm: number;
    weight_kg?: number | null;
    chest_cm: number;
    waist_cm: number;
    hips_cm?: number | null;
  };
  body_summary?: {
    inferred_type?: string;
    mapped_type?: string;
    base_source_id?: string;
    hips_cm?: number;
    hips_source?: string;
    body_field_count?: number;
    warnings?: string[];
  } | null;
  simulation_quality?: {
    case_count: number;
    completed_count: number;
    body_collisions: number;
    self_collisions: number;
    min_frames?: number | null;
    max_frames?: number | null;
    warnings: string[];
  } | null;
  suit_closure_summary?: {
    mode: "model_button_count";
    case_count: number;
    counts: Record<string, number>;
    closure_stitch_count: number;
  } | null;
  official_lower_summary?: {
    expected_image_count: number;
    detected_lower_count: number;
    static_completed_count: number;
    garment_counts: Record<string, number>;
    dynamic_included: boolean;
    dynamic_case_count?: number;
    simulation_quality?: {
      body_collisions: number;
      self_collisions: number;
      min_frames?: number | null;
      max_frames?: number | null;
      warnings: string[];
    } | null;
  } | null;
  artifacts: JobArtifact[];
  bundle_url?: string | null;
};

const JOB_STAGES = [
  { key: "queued", progress: 0, label: "等待执行", detail: "任务已进入队列，等待 GPU 空闲" },
  { key: "preparing", progress: 3, label: "准备人体数据", detail: "补全人体尺寸并生成本次人体文件" },
  { key: "chatgarment", progress: 10, label: "生成二维板片", detail: "ChatGarment 正在理解图片并生成参数化板片" },
  { key: "static_3d", progress: 38, label: "静态缝合与垂坠", detail: "GarmentCode 正在缝合、垂坠并渲染正反面" },
  { key: "dynamic_preparation", progress: 55, label: "准备动态网格", detail: "正在整理缝合网格和动作输入" },
  { key: "dynamic_3d", progress: 65, label: "动态布料与动作", detail: "正在进行动态布料仿真和视频渲染" },
  { key: "collecting", progress: 93, label: "收集全部产物", detail: "正在整理文件并生成一键下载包" },
  { key: "completed", progress: 100, label: "处理完成", detail: "全部制版与三维产物已经生成" },
] as const;

const SUIT_JOB_STAGES = [
  { key: "queued", progress: 0, label: "等待执行", detail: "任务已进入队列，等待 GPU 空闲" },
  { key: "preparing", progress: 3, label: "补全制版尺寸", detail: "正在匹配基础人体并生成 26 字段人体 YAML" },
  { key: "chatgarment", progress: 10, label: "识别男西装", detail: "正在识别西装上衣、纽扣和下装" },
  { key: "static_3d", progress: 42, label: "男西装制版与静态预览", detail: "正在生成上下装板片、完成扣合并生成静态三维服装" },
  { key: "dynamic_preparation", progress: 62, label: "合并上下装动态网格", detail: "正在把西装上衣与下装组合为动态服装" },
  { key: "dynamic_3d", progress: 68, label: "男西装动态布料与动作", detail: "正在驱动上下装生成动态视频" },
  { key: "collecting", progress: 93, label: "收集全部产物", detail: "正在整理模型参数、二维版片、静态三维和下载包" },
  { key: "completed", progress: 100, label: "处理完成", detail: "男西装二维、静态与所选动态三维产物已经生成" },
] as const;

const JOB_STATE_LABELS: Record<JobRecord["state"], string> = {
  queued: "排队中",
  running: "处理中",
  completed: "已完成",
  failed: "处理失败",
  cancelled: "已取消",
  trashed: "回收站",
};

function jobStages(job: JobRecord) {
  if (job.config.garment_mode === "mens_suit") {
    return job.config.action_id === "none"
      ? SUIT_JOB_STAGES.filter(
          (stage) => stage.key !== "dynamic_preparation" && stage.key !== "dynamic_3d",
        )
      : SUIT_JOB_STAGES;
  }
  return job.config.action_id === "none"
    ? JOB_STAGES.filter(
        (stage) => stage.key !== "dynamic_preparation" && stage.key !== "dynamic_3d",
      )
    : JOB_STAGES;
}

function currentJobStage(job: JobRecord) {
  const stages = jobStages(job);
  const stage = stages.find((item) => item.key === job.step);
  if (stage) {
    return {
      ...stage,
      detail: job.message || stage.detail,
    };
  }
  return {
    key: job.step,
    progress: job.progress,
    label:
      job.state === "failed"
        ? "处理失败"
        : job.state === "cancelled"
          ? "任务已取消"
          : "正在处理",
    detail: job.message,
  };
}

const BODY_GENDER_LABELS: Record<BodyGender, string> = {
  female: "女性",
  male: "男性",
  neutral: "中性",
};

const LOWER_GARMENT_LABELS: Record<string, string> = {
  Pants: "裤装",
  SkirtCircle: "圆裙",
  AsymmSkirtCircle: "不对称圆裙",
  GodetSkirt: "插片裙",
  Skirt2: "基础裙",
  SkirtManyPanels: "多片裙",
  PencilSkirt: "铅笔裙",
  SkirtLevels: "分层裙",
};

function artifactPresentation(artifact: JobArtifact, job?: JobRecord) {
  const value = `${artifact.path}/${artifact.name}`.toLowerCase();
  const isOfficialLower = value.includes("outputs/official_lower/");
  if (artifact.category === "inputs" || value.startsWith("inputs/")) {
    return { order: 0, step: "01 · 用户输入", label: "用户上传的服装图片" };
  }
  if (artifact.name === "body_preview.png") {
    const gender = job ? BODY_GENDER_LABELS[job.config.gender] : "本次任务";
    const label =
      job?.config.body_mode === "custom"
        ? `${gender}按本次尺寸定制的 SMPL-X 人体预览`
        : `${gender}预设标准 SMPL-X 人体预览`;
    return { order: 10, step: "02 · 人体准备", label };
  }
  if (/pattern_dxf_preview\.svg$/i.test(artifact.name)) {
    return {
      order: 23,
      step: "03 · DXF 导出",
      label: "1:1 毫米 DXF 样片预览",
    };
  }
  if (/pattern\.png$/i.test(artifact.name)) {
    const isK62Assembly = value.includes("k62_3d_pattern.png");
    return {
      order: isK62Assembly ? 22 : isOfficialLower ? 21 : 20,
      step: "03 · 二维制版",
      label:
        isK62Assembly
          ? "迁移到 K62 11 片装配拓扑的板片图"
          : isOfficialLower
          ? "男西装下装二维板片"
          : job?.config.garment_mode === "mens_suit"
          ? "男西装上衣二维板片"
          : "ChatGarment 生成的二维板片",
    };
  }
  if (/render_front\.png$/i.test(artifact.name)) {
    const isCombinedOutfit = value.includes("combined_outfit");
    return {
      order: isCombinedOutfit ? 34 : isOfficialLower ? 32 : 30,
      step: "04 · 静态三维",
      label: isCombinedOutfit
        ? "男西装上下装同穿静态三维 · 正面"
        : isOfficialLower
        ? "男西装下装静态三维 · 正面"
        : job?.config.garment_mode === "mens_suit"
        ? "K62 装配与 Warp 垂坠渲染 · 正面"
        : "GarmentCode 垂坠渲染 · 正面",
    };
  }
  if (/render_back\.png$/i.test(artifact.name)) {
    const isCombinedOutfit = value.includes("combined_outfit");
    return {
      order: isCombinedOutfit ? 34 : isOfficialLower ? 32 : 30,
      step: "04 · 静态三维",
      label: isCombinedOutfit
        ? "男西装上下装同穿静态三维 · 背面"
        : isOfficialLower
        ? "男西装下装静态三维 · 背面"
        : job?.config.garment_mode === "mens_suit"
        ? "K62 装配与 Warp 垂坠渲染 · 背面"
        : "GarmentCode 垂坠渲染 · 背面",
    };
  }
  if (/\.mp4$/i.test(artifact.name)) {
    return {
      order: 40,
      step: "05 · 动态三维",
      label: job?.config.garment_mode === "mens_suit"
        ? "男西装上下装动态布料视频"
        : "动态布料与动作渲染视频",
    };
  }
  if (value.includes("body")) {
    return { order: 11, step: "02 · 人体准备", label: "人体生成中间预览" };
  }
  return { order: 25, step: "03 · 模型产物", label: "模型生成的中间图像" };
}

function artifactPairKey(artifact: JobArtifact) {
  return artifact.path
    .toLowerCase()
    .replace(/_render_(front|back)\.png$/i, "_render");
}

function artifactViewOrder(artifact: JobArtifact) {
  if (/render_front\.png$/i.test(artifact.name)) return 0;
  if (/render_back\.png$/i.test(artifact.name)) return 1;
  return 0;
}

const MEASUREMENT_DISPLAY: Record<
  string,
  { label: string; unit: string; scale: number; digits: number }
> = {
  height_m: { label: "身高", unit: "cm", scale: 100, digits: 2 },
  weight_kg: { label: "体重", unit: "kg", scale: 1, digits: 2 },
  chest_m: { label: "胸围", unit: "cm", scale: 100, digits: 2 },
  waist_m: { label: "腰围", unit: "cm", scale: 100, digits: 2 },
  hips_m: { label: "臀围", unit: "cm", scale: 100, digits: 2 },
};

function MeasurementAudit({
  data,
}: {
  data: NonNullable<
    BodyGenerationResult["method"]["measurement_refinement"]
  >;
}) {
  if (!data.enabled) return null;
  return (
    <section className="measurement-audit">
      <div className="measurement-audit__heading">
        <span>VIRTUAL MEASUREMENT AUDIT</span>
        <small>{data.function_evaluations} 次求解评估</small>
      </div>
      <div className="measurement-audit__grid">
        {Object.entries(data.targets).map(([key, target]) => {
          const display = MEASUREMENT_DISPLAY[key];
          const measured = data.final[key];
          const error = data.final_error[key];
          if (!display || measured === undefined || error === undefined) return null;
          return (
            <div key={key}>
              <span>{display.label}</span>
              <strong>
                {(measured * display.scale).toFixed(display.digits)}
                <small>{display.unit}</small>
              </strong>
              <p>
                输入 {(target * display.scale).toFixed(display.digits)}
                {" · "}
                误差 {(error * display.scale).toFixed(display.digits)}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

const SIM_METRICS: Record<
  string,
  { seconds: number; frames: number; selfCollisions: number; bodyCollisions: number }
> = {
  "1aee14a8:lower": { seconds: 6.81, frames: 414, selfCollisions: 0, bodyCollisions: 0 },
  "1aee14a8:upper": { seconds: 11.7, frames: 405, selfCollisions: 0, bodyCollisions: 0 },
  "1dde6afe:lower": { seconds: 5.79, frames: 736, selfCollisions: 0, bodyCollisions: 0 },
  "1dde6afe:upper": { seconds: 12.1, frames: 405, selfCollisions: 0, bodyCollisions: 0 },
  "62bb809f:lower": { seconds: 12.4, frames: 1087, selfCollisions: 0, bodyCollisions: 0 },
  "62bb809f:upper": { seconds: 5.3, frames: 405, selfCollisions: 0, bodyCollisions: 0 },
  "6fe14e1f:wholebody": { seconds: 7.52, frames: 731, selfCollisions: 0, bodyCollisions: 0 },
  "72b08642:wholebody": { seconds: 19.5, frames: 783, selfCollisions: 38, bodyCollisions: 0 },
  "80141ce7:wholebody": { seconds: 10.2, frames: 1226, selfCollisions: 41, bodyCollisions: 0 },
  "8e3c458d:lower": { seconds: 2.62, frames: 415, selfCollisions: 0, bodyCollisions: 0 },
  "8e3c458d:upper": { seconds: 11.5, frames: 408, selfCollisions: 3, bodyCollisions: 0 },
  "c2b582eb:lower": { seconds: 2.95, frames: 545, selfCollisions: 0, bodyCollisions: 0 },
  "c2b582eb:upper": { seconds: 13.3, frames: 409, selfCollisions: 0, bodyCollisions: 0 },
  "d77c6f5d:lower": { seconds: 7.61, frames: 658, selfCollisions: 0, bodyCollisions: 0 },
  "d77c6f5d:upper": { seconds: 7.14, frames: 408, selfCollisions: 0, bodyCollisions: 0 },
  "e918651c:wholebody": { seconds: 9.08, frames: 638, selfCollisions: 112, bodyCollisions: 0 },
};

const DYNAMIC_RESULTS: Record<
  string,
  {
    frames: number;
    fps: number;
    duration: number;
    maxCollisions: number;
    finalCollisions: number;
    zeroCollisionFrames: number;
  }
> = {
  valid_garment_1aee14a8c7b4d56b4e8b6ddd575d1f561a72fdc75c43a4b6926f1655152193c6: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 11,
    finalCollisions: 0,
    zeroCollisionFrames: 330,
  },
  valid_garment_1dde6afed43187fe927089a615e3f744724ef3defdf3f2ae4a6cede5ad71dcea: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 0,
    finalCollisions: 0,
    zeroCollisionFrames: 331,
  },
  valid_garment_62bb809fc2dcd50409cb36163a0eb222f9aa1af0f256a3233b67b3ed4081dc71: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 69,
    finalCollisions: 40,
    zeroCollisionFrames: 324,
  },
  valid_garment_6fe14e1f646513ee93714fbe8026a84c6a2897be4df2f3c936cb2be8dd2d1762: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 4,
    finalCollisions: 1,
    zeroCollisionFrames: 324,
  },
  valid_garment_72b086429d2dfe2a8de6f4403a024b2bb17446021c9e8f9ebacfc7a990ac8434: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 71,
    finalCollisions: 12,
    zeroCollisionFrames: 293,
  },
  valid_garment_80141ce740f489f1d2f57a03f32c7577a28b62a6ac790a0d9ed8a18d961c2918: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 31,
    finalCollisions: 1,
    zeroCollisionFrames: 312,
  },
  valid_garment_8e3c458da20c290c216813ec07f1a2e8f9cfb4ee7e412a783a238ec353b346a0: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 44,
    finalCollisions: 0,
    zeroCollisionFrames: 327,
  },
  valid_garment_c2b582eb318455abaf8ed8e3126c1b423ade2704d810f7cd24428febda5632fa: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 19,
    finalCollisions: 0,
    zeroCollisionFrames: 330,
  },
  valid_garment_d77c6f5d4856831878eadb7fe3c8b180bfa9e9ad4a14936ac10a1697bb3c054f: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 49,
    finalCollisions: 0,
    zeroCollisionFrames: 312,
  },
  valid_garment_e918651cc154a7570e47d8b8f6c0f0f93cfbb7d5129103a1bacd8299ba945f91: {
    frames: 331,
    fps: 30,
    duration: 11.03,
    maxCollisions: 63,
    finalCollisions: 0,
    zeroCollisionFrames: 316,
  },
};

function assetPaths(caseId: string, garment: GarmentType) {
  const name = `valid_garment_${garment}`;
  const root = `/cases/${caseId}/${name}`;
  const resultRoot = `${root}/${name}`;
  return {
    pattern: `${root}/${name}_pattern.png`,
    front: `${resultRoot}/${name}_render_front.png`,
    back: `${resultRoot}/${name}_render_back.png`,
    spec: `${root}/${name}_specification.json`,
    mesh: `${resultRoot}/${name}_sim.obj`,
  };
}

function ResultImage({
  src,
  alt,
  kind,
  onOpen,
}: {
  src: string;
  alt: string;
  kind: "input" | "pattern" | "render";
  onOpen: () => void;
}) {
  const [failed, setFailed] = useState(false);

  return (
    <button
      className={`image-stage image-stage--${kind}`}
      type="button"
      onClick={onOpen}
      aria-label={`放大查看：${alt}`}
      disabled={failed}
    >
      {!failed ? (
        <img src={src} alt={alt} onError={() => setFailed(true)} />
      ) : (
        <span className="missing-state">
          <span className="missing-icon">···</span>
          结果同步中
        </span>
      )}
      {!failed && <span className="zoom-hint">点击放大</span>}
    </button>
  );
}

function DynamicPreview({
  caseId,
  caseNumber,
}: {
  caseId: string;
  caseNumber: number;
}) {
  const [missing, setMissing] = useState(false);
  const source = `/cases/${caseId}/dynamic/contourcraft.mp4`;
  const result = DYNAMIC_RESULTS[caseId];

  return (
    <section className="dynamic-result" aria-label={`示例 ${caseNumber} 动态布料结果`}>
      <div className="dynamic-copy">
        <span className="dynamic-kicker">DYNAMIC 3D · CONTOURCRAFT</span>
        <strong>动作驱动的缝合、垂坠与碰撞</strong>
        {result ? (
          <p>
            正式 SMPL-X v1.1 路线：{result.frames} 帧 · {result.fps} FPS ·{" "}
            {result.duration.toFixed(2)} 秒；碰撞峰值 {result.maxCollisions}、末帧{" "}
            {result.finalCollisions}，其中 {result.zeroCollisionFrames} 帧碰撞数为 0。
          </p>
        ) : (
          <p>静态结果已完成；该案例尚未进入 ContourCraft 批量动态仿真。</p>
        )}
      </div>
      {result && !missing ? (
        <video
          className="dynamic-video"
          controls
          loop
          muted
          playsInline
          preload="metadata"
          onError={() => setMissing(true)}
        >
          <source src={source} type="video/mp4" />
        </video>
      ) : (
        <div className="dynamic-waiting">
          <span>{result ? "VIDEO LOAD ERROR" : "PENDING BATCH RUN"}</span>
          <strong>{result ? "动态视频加载失败" : "动态案例尚待生成"}</strong>
          <small>
            {result
              ? "请检查服务器上的 MP4 文件与静态资源路径。"
              : "已有案例可查看动态结果，其余案例可继续批量生成。"}
          </small>
        </div>
      )}
    </section>
  );
}

const GBT_SOURCE_LABELS: Record<string, string> = {
  user_input: "用户输入",
  derived: "规则推导",
  base_inherited: "基础人体继承",
  template_inherited: "基础人体继承",
};

function GBTPatternBodyAdapter() {
  const [schema, setSchema] = useState<GBTSchema | null>(null);
  const [schemaError, setSchemaError] = useState("");
  const [gender, setGender] = useState<GBTGender>("female");
  const [values, setValues] = useState({
    height_cm: "168",
    chest_cm: "86",
    waist_cm: "68",
    hips_cm: "",
  });
  const [outOfRangeMode, setOutOfRangeMode] =
    useState<GBTBoundaryMode>("extrapolate");
  const [submitState, setSubmitState] = useState<
    "idle" | "submitting" | "success" | "error"
  >("idle");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<GBTGenerationResult | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/api/body/gbt1335/schema")
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return (await response.json()) as GBTSchema;
      })
      .then((payload) => {
        if (!active) return;
        setSchema(payload);
        setSchemaError("");
      })
      .catch((error: Error) => {
        if (!active) return;
        setSchemaError(`规则服务未连接：${error.message}`);
      });
    return () => {
      active = false;
    };
  }, []);

  const updateValue = (key: keyof typeof values, value: string) => {
    setValues((current) => ({ ...current, [key]: value }));
    setResult(null);
    setSubmitState("idle");
    setMessage("");
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitState("submitting");
    setMessage("正在匹配基础人体并生成 GarmentCode 人体 YAML…");
    setResult(null);
    try {
      const hips = values.hips_cm.trim();
      const response = await fetch("/api/body/gbt1335/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          gender,
          height_cm: Number(values.height_cm),
          chest_cm: Number(values.chest_cm),
          waist_cm: Number(values.waist_cm),
          hips_cm: hips ? Number(hips) : null,
          out_of_range_mode: outOfRangeMode,
        }),
      });
      const payload = (await response.json()) as GBTGenerationResult & {
        error?: string;
        message?: string;
      };
      if (!response.ok) {
        throw new Error(payload.message || payload.error || `HTTP ${response.status}`);
      }
      setResult(payload);
      setSubmitState("success");
      setMessage("26 项人体字段已生成，可下载并输入 GarmentCode 解码与制版流程。");
    } catch (error) {
      setSubmitState("error");
      setMessage(error instanceof Error ? error.message : "生成失败");
    }
  };

  const sourceCounts = result?.source_counts ?? result?.field_source_counts ?? {};
  const downloads = result?.downloads ?? {};
  const inferredType = result?.body_type?.inferred ?? "—";
  const mappedType = result?.body_type?.mapped ?? inferredType;
  const hipsSource = result?.target?.hips_source;

  return (
    <section className="body-customizer body-customizer--gbt" id="body-gbt1335">
      <div className="customizer-heading">
        <div>
          <p className="eyebrow">方案 B · GB/T 1335 → GARMENTCODE</p>
          <h2>用少量量体数据补全制版人体尺寸</h2>
          <p>
            根据胸腰差判断 Y / A / B / C 体型，再匹配男女各三套固定基础人体。C
            体型统一映射到 B；用户实测值优先，缺失臀围由规则估算，其余字段按规则联动或继承。
          </p>
        </div>
        <div className={`deployment-state ${schema ? "deployment-state--ready" : ""}`}>
          <span>{schema ? "RULE ROUTE READY" : "CONNECTING"}</span>
          <strong>{schema ? "六套基础人体可用" : "正在连接规则服务"}</strong>
          <small>
            {schemaError ||
              "无需 GPU、无需模型训练；输出完整 26 字段 GarmentCode 人体 YAML。"}
          </small>
        </div>
      </div>

      <form className="body-form gbt-form" onSubmit={submit}>
        <div className="gbt-route-strip" aria-label="方案 B 处理流程">
          <span>01 输入尺寸</span>
          <i>→</i>
          <span>02 判断体型</span>
          <i>→</i>
          <span>03 选择六模板之一</span>
          <i>→</i>
          <span>04 补全 26 字段</span>
        </div>

        <fieldset className="model-choice">
          <legend>01 · 性别与基础模板组</legend>
          <div className="choice-grid choice-grid--two">
            {(
              [
                ["female", "女性", "Y / A / B 三套基础人体"],
                ["male", "男性", "Y / A / B 三套基础人体"],
              ] as [GBTGender, string, string][]
            ).map(([value, label, note]) => (
              <label key={value} className={gender === value ? "selected" : ""}>
                <input
                  type="radio"
                  name="gbt_gender"
                  value={value}
                  checked={gender === value}
                  onChange={() => {
                    setGender(value);
                    setResult(null);
                    setSubmitState("idle");
                  }}
                />
                <strong>{label}</strong>
                <small>{note}</small>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="measurement-fields">
          <legend>02 · 用户量体数据</legend>
          <div className="measurement-grid measurement-grid--gbt">
            {(
              [
                ["height_cm", "身高", true],
                ["chest_cm", "胸围", true],
                ["waist_cm", "腰围", true],
                ["hips_cm", "臀围（可留空）", false],
              ] as [keyof typeof values, string, boolean][]
            ).map(([key, label, required]) => (
              <label key={key}>
                <span>{label}</span>
                <div>
                  <input
                    type="number"
                    min="1"
                    step="0.1"
                    required={required}
                    value={values[key]}
                    onChange={(event) => updateValue(key, event.target.value)}
                  />
                  <small>cm</small>
                </div>
              </label>
            ))}
          </div>
          <p className="route-note">
            臀围留空时自动估算；身高、胸围、腰围、臀围若由用户填写，将始终以用户值为准。
          </p>
        </fieldset>

        <fieldset className="gbt-boundary-field">
          <legend>03 · 超出国标范围时</legend>
          <div className="boundary-choice">
            {(
              [
                ["extrapolate", "允许外推", "保留用户输入，按现有比例规则继续计算"],
                [
                  "clamp",
                  "推导差值夹到边界",
                  "用户实测值不改，仅限制体型分类值及缺失、联动字段的推导差值",
                ],
              ] as [GBTBoundaryMode, string, string][]
            ).map(([value, label, note]) => (
              <label key={value} className={outOfRangeMode === value ? "selected" : ""}>
                <input
                  type="radio"
                  name="out_of_range_mode"
                  value={value}
                  checked={outOfRangeMode === value}
                  onChange={() => {
                    setOutOfRangeMode(value);
                    setResult(null);
                    setSubmitState("idle");
                  }}
                />
                <strong>{label}</strong>
                <small>{note}</small>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="model-choice">
          <legend>{"04 · 动态展示动作"}</legend>
          <div className="choice-grid action-choice-grid">
            {(schema?.actions ?? []).map((action) => (
              <label
                key={action.id}
                className={actionId === action.id ? "selected" : ""}
              >
                <input
                  type="radio"
                  name="action_id"
                  value={action.id}
                  checked={actionId === action.id}
                  disabled={!action.available}
                  onChange={() => {
                    setActionId(action.id);
                    setSubmitState("idle");
                    setResult(null);
                  }}
                />
                <strong>{action.label_zh}</strong>
                <small>
                  {action.available ? action.id : `${action.id} · 未安装`}
                </small>
              </label>
            ))}
          </div>
          <p className="route-note">
            {
              "动作采用原始动作内容；系统只做 SMPL-X 格式适配。选择“不生成动态视频”时仍会生成人体静态文件。"
            }
          </p>
        </fieldset>

        <div className="generation-bar">
          <div>
            <span>OUTPUT</span>
            <strong>garmentcode_body.yaml · audit.json · 26 项制版人体尺寸</strong>
          </div>
          <button type="submit" disabled={!schema || submitState === "submitting"}>
            {submitState === "submitting" ? "正在补全…" : "生成制版人体 YAML"}
          </button>
        </div>

        {(message || result) && (
          <div className={`generation-result generation-result--${submitState} gbt-result`}>
            <strong>{message}</strong>
            {result && (
              <>
                <div className="gbt-result-grid">
                  <div>
                    <span>原始体型判断</span>
                    <strong>{inferredType}</strong>
                  </div>
                  <div>
                    <span>实际映射体型</span>
                    <strong>{mappedType}</strong>
                    {inferredType === "C" && mappedType === "B" && <small>C → B</small>}
                  </div>
                  <div>
                    <span>基础人体 ID</span>
                    <strong>{result.base_source_id ?? "—"}</strong>
                  </div>
                  <div>
                    <span>臀围来源</span>
                    <strong>
                      {hipsSource === "user_input"
                        ? "用户输入"
                        : hipsSource === "estimated_from_waist"
                          ? "由腰围估算"
                          : hipsSource ?? "—"}
                    </strong>
                  </div>
                </div>

                <section className="gbt-source-audit">
                  <div className="gbt-source-heading">
                    <span>26 字段来源统计</span>
                    <small>
                      合计 {Object.values(sourceCounts).reduce((sum, count) => sum + count, 0)} 项
                    </small>
                  </div>
                  <div className="gbt-source-counts">
                    {Object.entries(sourceCounts).map(([source, count]) => (
                      <div key={source}>
                        <span>{GBT_SOURCE_LABELS[source] ?? source}</span>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </div>
                </section>

                {(result.warnings?.length ?? 0) > 0 && (
                  <section className="gbt-warnings">
                    <strong>规则提示</strong>
                    <ul>
                      {result.warnings?.map((warning, index) => (
                        <li key={`${warning}-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  </section>
                )}

                <div className="gbt-downloads">
                  {Object.entries(downloads).map(([name, href]) => (
                    <a href={href} key={name} download>
                      下载 {result.files?.[name] ?? name}
                    </a>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </form>
    </section>
  );
}

function BodyCustomizer() {
  const [gender, setGender] = useState<BodyGender>("female");
  const [neutralProfile, setNeutralProfile] =
    useState<SemanticProfile>("female");
  const [actionId, setActionId] =
    useState<BodyActionId>("official_showcase");
  const [measurements, setMeasurements] = useState({
    height_cm: "168",
    weight_kg: "58",
    chest_cm: "86",
    waist_cm: "68",
    hips_cm: "92",
  });
  const [ratings, setRatings] = useState<Record<string, number>>({});
  const [schema, setSchema] = useState<BodySchema | null>(null);
  const [serviceError, setServiceError] = useState("");
  const [submitState, setSubmitState] = useState<
    "idle" | "submitting" | "success" | "error"
  >("idle");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<BodyGenerationResult | null>(null);

  const semanticProfile: SemanticProfile =
    gender === "neutral" ? neutralProfile : gender;
  const attributes = SEMANTIC_ATTRIBUTES[semanticProfile];
  const variant = measurements.weight_kg.trim()
    ? "05b_ahwcwh2s"
    : "04b_ahcwh2s";
  const routeReady =
    schema?.checkpoint_routes.some(
      (route) =>
        route.model_gender === gender &&
        route.semantic_profile === semanticProfile &&
        route.variant === variant &&
        route.available,
    ) ?? false;

  useEffect(() => {
    let active = true;
    fetch("/api/body/schema")
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return (await response.json()) as BodySchema;
      })
      .then((payload) => {
        if (active) {
          setSchema(payload);
          setServiceError("");
        }
      })
      .catch((error: Error) => {
        if (active) setServiceError(`人体服务未连接：${error.message}`);
      });
    return () => {
      active = false;
    };
  }, []);

  const updateMeasurement = (key: keyof typeof measurements, value: string) => {
    setMeasurements((current) => ({ ...current, [key]: value }));
    setResult(null);
    setSubmitState("idle");
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitState("submitting");
    setMessage("正在运行 SHAPY 映射并生成 SMPL-X 文件…");
    setResult(null);
    try {
      const response = await fetch("/api/body/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          gender,
          semantic_profile: semanticProfile,
          height_cm: Number(measurements.height_cm),
          weight_kg: measurements.weight_kg.trim()
            ? Number(measurements.weight_kg)
            : null,
          chest_cm: Number(measurements.chest_cm),
          waist_cm: Number(measurements.waist_cm),
          hips_cm: Number(measurements.hips_cm),
          attributes: Object.fromEntries(
            attributes.map(({ key }) => [key, ratings[key] ?? 3]),
          ),
          action_id: actionId,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.message || payload.error || `HTTP ${response.status}`);
      }
      setResult(payload as BodyGenerationResult);
      setSubmitState("success");
      setMessage("人体文件已生成，可直接下载并接入静态注册与动态布料流程。");
    } catch (error) {
      setSubmitState("error");
      setMessage(error instanceof Error ? error.message : "生成失败");
    }
  };

  return (
    <section className="body-customizer" id="body-customizer">
      <div className="solution-label">方案 A · SHAPY 测量映射 · 生成 SMPL-X 三维人体</div>
      <div className="customizer-heading">
        <div>
          <p className="eyebrow">CUSTOM BODY · SHAPY → SMPL-X</p>
          <h2>用人体尺寸和语义描述生成定制体型</h2>
          <p>
            选择人体模型后输入真实测量值，并用 1–5 分描述体型。系统输出与现有工程兼容的
            PKL、OBJ、NPZ 和动作文件；不会在本实例重新训练模型。
          </p>
        </div>
        <div
          className={`deployment-state ${
            routeReady ? "deployment-state--ready" : ""
          }`}
        >
          <span>{routeReady ? "ROUTE READY" : "WEIGHT REQUIRED"}</span>
          <strong>
            {serviceError
              ? "服务未连接"
              : routeReady
                ? "当前路线可生成"
                : "等待 SHAPY 官方权重"}
          </strong>
          <small>
            {serviceError ||
              (routeReady
                ? `${variant} · ${gender.toUpperCase()}`
                : "人体模型、接口和网页已部署；不使用第三方权重。")}
          </small>
        </div>
      </div>

      <form className="body-form" onSubmit={submit}>
        <fieldset className="model-choice">
          <legend>01 · 人体模型</legend>
          <div className="choice-grid">
            {(
              [
                ["female", "女性", "SMPLX_FEMALE"],
                ["male", "男性", "SMPLX_MALE"],
                ["neutral", "中性", "SMPLX_NEUTRAL"],
              ] as [BodyGender, string, string][]
            ).map(([value, label, model]) => (
              <label key={value} className={gender === value ? "selected" : ""}>
                <input
                  type="radio"
                  name="gender"
                  value={value}
                  checked={gender === value}
                  onChange={() => {
                    setGender(value);
                    setSubmitState("idle");
                    setResult(null);
                  }}
                />
                <strong>{label}</strong>
                <small>{model}</small>
              </label>
            ))}
          </div>
          {gender === "neutral" && (
            <div className="neutral-profile">
              <span>中性模型使用哪套语义词表：</span>
              {(["female", "male"] as SemanticProfile[]).map((profile) => (
                <button
                  type="button"
                  key={profile}
                  className={neutralProfile === profile ? "active" : ""}
                  onClick={() => setNeutralProfile(profile)}
                >
                  {profile === "female" ? "女性词表" : "男性词表"}
                </button>
              ))}
            </div>
          )}
        </fieldset>

        <fieldset className="measurement-fields">
          <legend>02 · 人体测量</legend>
          <div className="measurement-grid">
            {(
              [
                ["height_cm", "身高", "cm", 130, 220],
                ["weight_kg", "体重（可留空）", "kg", 30, 220],
                ["chest_cm", "胸围", "cm", 50, 180],
                ["waist_cm", "腰围", "cm", 45, 180],
                ["hips_cm", "臀围", "cm", 50, 190],
              ] as [
                keyof typeof measurements,
                string,
                string,
                number,
                number,
              ][]
            ).map(([key, label, unit, min, max]) => (
              <label key={key}>
                <span>{label}</span>
                <div>
                  <input
                    type="number"
                    min={min}
                    max={max}
                    step="0.1"
                    required={key !== "weight_kg"}
                    value={measurements[key]}
                    onChange={(event) => updateMeasurement(key, event.target.value)}
                  />
                  <small>{unit}</small>
                </div>
              </label>
            ))}
          </div>
          <p className="route-note">
            当前选择：{variant === "05b_ahwcwh2s" ? "05b（含体重）" : "04b（不含体重）"} ·
            量体数据请使用贴身测量值。
          </p>
        </fieldset>

        <fieldset className="semantic-fields">
          <legend>03 · 语义体型评分</legend>
          <p>1 表示完全不符合，3 表示中性，5 表示非常符合；默认均为 3。</p>
          <div className="semantic-grid">
            {attributes.map(({ key, label }) => {
              const value = ratings[key] ?? 3;
              return (
                <label key={`${semanticProfile}-${key}`}>
                  <span>{label}</span>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="0.5"
                    value={value}
                    onChange={(event) =>
                      setRatings((current) => ({
                        ...current,
                        [key]: Number(event.target.value),
                      }))
                    }
                  />
                  <strong>{value.toFixed(1)}</strong>
                </label>
              );
            })}
          </div>
        </fieldset>

        <div className="generation-bar">
          <div>
            <span>OUTPUT</span>
            <strong>
              {actionId === "none"
                ? "registered_params.pkl · body.obj · body_params.npz"
                : "registered_params.pkl · body.obj · body_params.npz · motion.npz"}
            </strong>
          </div>
          <button type="submit" disabled={!routeReady || submitState === "submitting"}>
            {submitState === "submitting"
              ? "正在生成…"
              : routeReady
                ? "生成人体文件"
                : "等待官方权重"}
          </button>
        </div>

        {(message || result) && (
          <div className={`generation-result generation-result--${submitState}`}>
            <strong>{message}</strong>
            {result && (
              <>
                {result.downloads.preview && (
                  <img
                    className="generated-body-preview"
                    src={result.downloads.preview}
                    alt={`${result.gender} SMPL-X 定制人体预览`}
                  />
                )}
                <p>
                  {result.gender.toUpperCase()} · {result.topology.vertices} 顶点 ·{" "}
                  {result.topology.faces} 面 · {result.topology.joints} 关节
                </p>
                {result.action && (
                  <p>{`动作：${result.action.label_zh} · ${result.action.id}`}</p>
                )}
                {result.method.measurement_refinement && (
                  <MeasurementAudit
                    data={result.method.measurement_refinement}
                  />
                )}
                <div>
                  {Object.entries(result.downloads).map(([name, href]) => (
                    <a href={href} key={name} download>
                      {name}
                    </a>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </form>
    </section>
  );
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function JobProgressPanel({ job, prominent = false }: { job: JobRecord; prominent?: boolean }) {
  const stages = jobStages(job);
  const current = currentJobStage(job);
  const currentIndex = stages.findIndex((stage) => stage.key === job.step);
  const finished = job.state === "completed";

  return (
    <section
      className={`task-progress-panel${prominent ? " task-progress-panel--prominent" : ""}`}
      aria-live="polite"
      aria-label={`任务进度 ${job.progress}%`}
    >
      <div className="task-progress-heading">
        <div>
          <span>{finished ? "处理完成" : "当前处理步骤"}</span>
          <strong>{current.label}</strong>
          <small>{current.detail}</small>
        </div>
        <strong className="task-progress-number">{job.progress}%</strong>
      </div>
      <div className="job-progress job-progress--large">
        <span style={{ width: `${Math.max(0, Math.min(100, job.progress))}%` }} />
      </div>
      {job.runtime && (
        <div className={`task-runtime task-runtime--${job.runtime.kind}`}>
          <span aria-hidden="true" className="task-runtime-dot" />
          <div>
            <small>当前计算资源</small>
            <strong>{job.runtime.label}</strong>
            <span>{job.runtime.detail}</span>
          </div>
          <em>{job.runtime.verified ? "可用" : "准备中"}</em>
        </div>
      )}
      <ol className="task-stage-list">
        {stages.map((stage, index) => {
          const complete = finished || index < currentIndex;
          const active = index === currentIndex && !finished;
          return (
            <li
              key={stage.key}
              className={complete ? "complete" : active ? "active" : "pending"}
            >
              <span>{complete ? "✓" : index + 1}</span>
              <small>{stage.label}</small>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function WorkflowStudio() {
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("服装1");
  const [garmentMode, setGarmentMode] = useState<GarmentMode>("mens_suit");
  const [bodyMode, setBodyMode] = useState<"preset" | "custom">("preset");
  const [gender, setGender] = useState<BodyGender>("male");
  const [actionId, setActionId] = useState<BodyActionId>("official_showcase");
  const [measurements, setMeasurements] = useState({
    height_cm: "180",
    weight_kg: "75",
    chest_cm: "100",
    waist_cm: "84",
    hips_cm: "98",
  });
  const [actions, setActions] = useState<BodySchema["actions"]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [storage, setStorage] = useState<{
    total: number;
    used: number;
    free: number;
    runs: number;
    trash: number;
    accepting_jobs: boolean;
  } | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const [focusedJobId, setFocusedJobId] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshFeedback, setRefreshFeedback] = useState("");

  const refresh = async (showFeedback = false) => {
    if (showFeedback) {
      setRefreshing(true);
      setRefreshFeedback("正在从服务器读取最新状态…");
    }
    try {
      const [jobResponse, storageResponse] = await Promise.all([
        fetch("/api/jobs", { cache: "no-store" }),
        fetch("/api/jobs/storage", { cache: "no-store" }),
      ]);
      if (!jobResponse.ok || !storageResponse.ok) {
        throw new Error("服务器状态读取失败");
      }
      const payload = await jobResponse.json();
      setJobs(payload.jobs ?? []);
      setStorage(await storageResponse.json());
      if (showFeedback) {
        const time = new Date().toLocaleTimeString("zh-CN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
        setRefreshFeedback(`已更新 · ${time}`);
      }
    } catch (error) {
      if (showFeedback) {
        setRefreshFeedback(error instanceof Error ? error.message : "刷新失败，请重试");
      }
      throw error;
    } finally {
      if (showFeedback) setRefreshing(false);
    }
  };

  useEffect(() => {
    let active = true;
    fetch("/api/body/schema")
      .then((response) => response.json())
      .then((payload: BodySchema) => {
        if (active) setActions(payload.actions ?? []);
      })
      .catch(() => undefined);
    refresh().catch(() => undefined);
    const timer = window.setInterval(() => refresh().catch(() => undefined), 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!files.length) {
      setMessage("请至少选择一张服装图片。");
      return;
    }
    setSubmitting(true);
    setMessage("正在上传并创建任务…");
    try {
      const config = {
        name,
        garment_mode: garmentMode,
        body_mode: bodyMode,
        gender,
        semantic_profile: gender === "neutral" ? "female" : gender,
        action_id: actionId,
        height_cm: Number(measurements.height_cm),
        weight_kg: measurements.weight_kg ? Number(measurements.weight_kg) : null,
        chest_cm: Number(measurements.chest_cm),
        waist_cm: Number(measurements.waist_cm),
        hips_cm: measurements.hips_cm ? Number(measurements.hips_cm) : null,
        boundary_policy: "extrapolate",
        attributes: {},
      };
      const form = new FormData();
      form.append("config", JSON.stringify(config));
      files.forEach((file) => form.append("images", file, file.name));
      const response = await fetch("/api/jobs", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message || payload.error || `HTTP ${response.status}`);
      setMessage(`任务 ${payload.job.id.slice(0, 8)} 已进入队列。`);
      setFocusedJobId(payload.job.id);
      setFiles([]);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "任务创建失败");
    } finally {
      setSubmitting(false);
    }
  };

  const mutate = async (
    job: JobRecord,
    action: "cancel" | "delete",
    mode?: "trash" | "permanent" | "cache",
  ) => {
    if (mode === "permanent" && !window.confirm("永久删除后无法恢复，确定继续吗？")) return;
    const response = await fetch(`/api/jobs/${job.id}/${action}`, {
      method: "POST",
      headers: mode ? { "content-type": "application/json" } : undefined,
      body: mode ? JSON.stringify({ mode }) : undefined,
    });
    const payload = await response.json();
    setMessage(response.ok ? "任务记录已更新。" : payload.message || payload.error);
    await refresh();
  };

  const updateMeasurement = (key: keyof typeof measurements, value: string) =>
    setMeasurements((current) => ({ ...current, [key]: value }));

  const updateGarmentMode = (value: GarmentMode) => {
    setGarmentMode(value);
    if (value === "mens_suit") {
      setBodyMode("preset");
      setGender("male");
      setActionId("official_showcase");
      setMeasurements({
        height_cm: "180",
        weight_kg: "75",
        chest_cm: "100",
        waist_cm: "84",
        hips_cm: "98",
      });
    } else {
      setBodyMode("preset");
      setGender("female");
      setActionId("official_showcase");
      setMeasurements({
        height_cm: "168",
        weight_kg: "55",
        chest_cm: "84",
        waist_cm: "68",
        hips_cm: "90",
      });
    }
    setMessage("");
  };

  const activeJob =
    jobs.find((job) => job.id === focusedJobId) ??
    jobs.find((job) => job.state === "running" || job.state === "queued");

  return (
    <section className="workflow-studio" id="workflow-studio">
      <div className="studio-heading">
        <div>
          <p className="eyebrow">END-TO-END DEMO</p>
          <h2>上传图片和人体尺寸，一次查看全部处理产物</h2>
          <p>单张或批量图片共用同一套人体参数；任务在单张 GPU 上依次排队执行。</p>
        </div>
        <div className="storage-card">
          <span>SERVER STORAGE</span>
          <strong>{storage ? `${formatBytes(storage.free)} 可用` : "正在读取"}</strong>
          <small>
            {storage
              ? `任务 ${formatBytes(storage.runs)} · 回收站 ${formatBytes(storage.trash)}`
              : "任务记录与文件均保存在服务器"}
          </small>
        </div>
      </div>

      <form className="studio-form" onSubmit={submit}>
        <div className="studio-grid">
          <label className="upload-zone">
            <span>01 · 服装图片</span>
            <strong>{files.length ? `已选择 ${files.length} 张` : "点击选择单张或多张图片"}</strong>
            <small>PNG / JPG / WEBP · 每批最多 20 张 · 批量共用人体参数</small>
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              multiple
              onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
            />
          </label>

          <div className="studio-settings">
            <label>
              <span>任务名称</span>
              <input value={name} maxLength={100} onChange={(event) => setName(event.target.value)} />
            </label>
            <label>
              <span>服装类型</span>
              <select value={garmentMode} onChange={(event) => updateGarmentMode(event.target.value as GarmentMode)}>
                <option value="general">通用女装</option>
                <option value="mens_suit">男西装</option>
              </select>
            </label>
            <label>
              <span>三维人体</span>
              <select disabled={!SHOW_EXPERIMENTAL_BODY_TOOLS || garmentMode === "mens_suit"} value={bodyMode} onChange={(event) => setBodyMode(event.target.value as "preset" | "custom")}>
                <option value="preset">预设标准人体</option>
                {SHOW_EXPERIMENTAL_BODY_TOOLS && (
                  <option value="custom">{garmentMode === "mens_suit" ? "本次尺寸用于制版与板片适配" : "按本次尺寸定制人体"}</option>
                )}
              </select>
            </label>
            <label>
              <span>性别模型</span>
              <select disabled={garmentMode === "mens_suit"} value={gender} onChange={(event) => setGender(event.target.value as BodyGender)}>
                <option value="female">女性标准人体</option>
                <option value="male">男性标准人体</option>
                <option value="neutral">中性标准人体</option>
              </select>
            </label>
            <label>
              <span>动态动作</span>
              <select value={actionId} onChange={(event) => setActionId(event.target.value as BodyActionId)}>
                {actions.map((action) => (
                  <option
                    key={action.id}
                    value={action.id}
                    disabled={
                      !action.available ||
                      (garmentMode === "mens_suit" && !["none", "official_showcase"].includes(action.id))
                    }
                  >
                    {action.label_zh}{action.available ? "" : "（未安装）"}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <div className="studio-measurements">
          {(
            [
              ["height_cm", "身高", "cm"],
              ["weight_kg", "体重", "kg"],
              ["chest_cm", "胸围", "cm"],
              ["waist_cm", "腰围", "cm"],
              ["hips_cm", "臀围", "cm"],
            ] as [keyof typeof measurements, string, string][]
          ).map(([key, label, unit]) => (
            <label key={key}>
              <span>{label}</span>
              <div>
                <input
                  type="number"
                  step="0.1"
                  value={measurements[key]}
                  required={key !== "weight_kg" && key !== "hips_cm"}
                  onChange={(event) => updateMeasurement(key, event.target.value)}
                />
                <small>{unit}</small>
              </div>
            </label>
          ))}
        </div>

        <div className="studio-submit">
          <p>{message || "上传后可关闭此页面；任务状态和产物记录会保留在服务器。"}</p>
          <button disabled={submitting || !files.length || storage?.accepting_jobs === false}>
            {submitting
              ? "正在创建…"
              : "开始生成"}
          </button>
        </div>
      </form>

      {submitting && (
        <section className="task-progress-panel task-progress-panel--prominent" aria-live="polite">
          <div className="task-progress-heading">
            <div>
              <span>正在提交</span>
              <strong>上传图片并创建任务</strong>
              <small>上传完成后会立即显示服务器端的逐步处理进度。</small>
            </div>
            <span className="refresh-spinner" aria-hidden="true" />
          </div>
          <div className="job-progress job-progress--large job-progress--indeterminate">
            <span />
          </div>
        </section>
      )}

      {!submitting && activeJob && (
        <JobProgressPanel job={activeJob} prominent />
      )}

      <div className="job-section">
        <div className="job-section-heading">
          <div>
            <h3>生成记录与任务状态</h3>
            <small aria-live="polite">{refreshFeedback || "页面每 3 秒自动同步一次"}</small>
          </div>
          <button
            className={refreshing ? "is-refreshing" : ""}
            type="button"
            disabled={refreshing}
            onClick={() => refresh(true).catch(() => undefined)}
          >
            {refreshing && <span className="refresh-spinner" aria-hidden="true" />}
            {refreshing ? "刷新中…" : "立即刷新"}
          </button>
        </div>
        {!jobs.length && <p className="empty-jobs">还没有任务。上传一张图片即可开始。</p>}
        <div className="job-list">
          {jobs.map((job) => {
            const media = job.artifacts
              .filter(
                (artifact) =>
                  /\.(png|jpe?g|webp|mp4)$/i.test(artifact.name) ||
                  /pattern_dxf_preview\.svg$/i.test(artifact.name),
              )
              .sort((left, right) => {
                const orderDifference =
                  artifactPresentation(left, job).order - artifactPresentation(right, job).order;
                if (orderDifference) return orderDifference;
                const pairDifference = artifactPairKey(left).localeCompare(
                  artifactPairKey(right),
                  "zh-CN",
                  { numeric: true },
                );
                if (pairDifference) return pairDifference;
                return (
                  artifactViewOrder(left) - artifactViewOrder(right) ||
                  left.path.localeCompare(right.path, "zh-CN", { numeric: true })
                );
              });
            const dxfArtifacts = job.artifacts.filter((artifact) =>
              artifact.name.toLowerCase().endsWith(".dxf"),
            );
            return (
              <article className="job-card" key={job.id}>
                <header>
                  <div>
                    <span className={`job-state job-state--${job.state}`}>
                      {JOB_STATE_LABELS[job.state]}
                    </span>
                    <h4>{job.name}</h4>
                    <code>{job.id.slice(0, 12)}</code>
                  </div>
                  <div className="job-actions">
                    {job.bundle_url && <a href={job.bundle_url}>一键下载 ZIP</a>}
                    {(job.state === "queued" || job.state === "running") && (
                      <button type="button" onClick={() => mutate(job, "cancel")}>取消</button>
                    )}
                    {job.state !== "queued" && job.state !== "running" && (
                      <button type="button" onClick={() => mutate(job, "delete", "cache")}>清中间缓存</button>
                    )}
                    <button type="button" onClick={() => mutate(job, "delete", "trash")}>移到回收站</button>
                    <button className="danger" type="button" onClick={() => mutate(job, "delete", "permanent")}>永久删除</button>
                  </div>
                </header>
                <JobProgressPanel job={job} />
                <div className="job-meta">
                  <span>{job.image_count} 张图片</span>
                  <span>{job.config.garment_mode === "mens_suit" ? "男西装" : "通用女装"}</span>
                  <span>{job.config.body_mode === "custom" ? "本次尺寸" : "预设人体"}</span>
                  <span>{BODY_GENDER_LABELS[job.config.gender]}</span>
                  <span>{job.config.action_id}</span>
                  {job.suit_closure_summary && (
                    <span>
                      模型扣合 · {Object.entries(job.suit_closure_summary.counts)
                        .sort(([left], [right]) => Number(left) - Number(right))
                        .map(([count, cases]) => `${count}扣 × ${cases}`)
                        .join("、")}
                    </span>
                  )}
                  {job.official_lower_summary && (
                    <span>
                      下装 · {job.official_lower_summary.detected_lower_count
                        ? Object.entries(job.official_lower_summary.garment_counts)
                            .sort(([left], [right]) => left.localeCompare(right))
                            .map(([type, cases]) => `${LOWER_GARMENT_LABELS[type] || type} × ${cases}`)
                            .join("、") + ` · 静态 ${job.official_lower_summary.static_completed_count}/${job.official_lower_summary.detected_lower_count}` +
                              (job.official_lower_summary.dynamic_included
                                ? ` · 动态 ${job.official_lower_summary.dynamic_case_count ?? job.official_lower_summary.detected_lower_count}/${job.official_lower_summary.detected_lower_count}`
                                : "")
                        : "未检测到"}
                    </span>
                  )}
                  <span>{formatBytes(job.size_bytes)}</span>
                </div>
                <div className="job-inputs">
                  <span>身高 {job.config.height_cm} cm</span>
                  <span>体重 {job.config.weight_kg ?? "未填"}{job.config.weight_kg ? " kg" : ""}</span>
                  <span>胸围 {job.config.chest_cm} cm</span>
                  <span>腰围 {job.config.waist_cm} cm</span>
                  <span>臀围 {job.config.hips_cm ?? "自动估算"}{job.config.hips_cm ? " cm" : ""}</span>
                </div>
                {job.simulation_quality && (
                  <div className={`job-simulation-quality ${job.simulation_quality.warnings.length ? "has-warning" : "is-clean"}`}>
                    <strong>静态三维质量</strong>
                    <span>完成 {job.simulation_quality.completed_count}/{job.simulation_quality.case_count}</span>
                    <span>人体穿插 {job.simulation_quality.body_collisions}</span>
                    <span>布料自交 {job.simulation_quality.self_collisions}</span>
                    <span>
                      收敛帧 {job.simulation_quality.min_frames ?? "—"}
                      {job.simulation_quality.max_frames !== job.simulation_quality.min_frames
                        ? `–${job.simulation_quality.max_frames ?? "—"}`
                        : ""}
                    </span>
                    {!!job.simulation_quality.warnings.length && (
                      <small>质量警告：{job.simulation_quality.warnings.join("、")}</small>
                    )}
                  </div>
                )}
                {job.official_lower_summary?.simulation_quality && (
                  <div className={`job-simulation-quality ${job.official_lower_summary.simulation_quality.warnings.length ? "has-warning" : "is-clean"}`}>
                    <strong>下装静态预览质量</strong>
                    <span>人体穿插 {job.official_lower_summary.simulation_quality.body_collisions}</span>
                    <span>布料自交 {job.official_lower_summary.simulation_quality.self_collisions}</span>
                    <span>
                      收敛帧 {job.official_lower_summary.simulation_quality.min_frames ?? "—"}
                      {job.official_lower_summary.simulation_quality.max_frames !== job.official_lower_summary.simulation_quality.min_frames
                        ? `–${job.official_lower_summary.simulation_quality.max_frames ?? "—"}`
                        : ""}
                    </span>
                    {!!job.official_lower_summary.simulation_quality.warnings.length && (
                      <small>质量警告：{job.official_lower_summary.simulation_quality.warnings.join("、")}</small>
                    )}
                  </div>
                )}
                {job.error && <pre className="job-error">{job.error}</pre>}
                {!!dxfArtifacts.length && (
                  <div className="job-dxf-downloads">
                    <div>
                      <strong>DXF 样片</strong>
                      <small>1:1 毫米制板片，可分别下载或随结果 ZIP 一并下载</small>
                    </div>
                    <div>
                      {dxfArtifacts.map((artifact, index) => (
                        <a key={artifact.path} href={artifact.url} download>
                          下载 DXF {dxfArtifacts.length > 1 ? index + 1 : ""}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
                {!!media.length && (
                  <div className="job-media-grid">
                    {media.map((artifact, index) => {
                      const presentation = artifactPresentation(artifact, job);
                      const version = encodeURIComponent(`${job.updated_at}-${artifact.bytes}`);
                      const source = `${artifact.url}?v=${version}`;
                      const isVideo = artifact.name.toLowerCase().endsWith(".mp4");
                      const videoReady = job.state === "completed" || job.step === "collecting";
                      return (
                        <figure className="job-media-card" key={`${artifact.path}-${version}`}>
                          <div className="job-media-label">
                            <span>{String(index + 1).padStart(2, "0")}</span>
                            <div>
                              <small>{presentation.step}</small>
                              <strong>{presentation.label}</strong>
                            </div>
                          </div>
                          {isVideo && videoReady ? (
                            <video controls playsInline preload="metadata" src={source}>
                              当前浏览器无法播放该视频。
                            </video>
                          ) : isVideo ? (
                            <div className="video-processing">
                              <span className="refresh-spinner" aria-hidden="true" />
                              <strong>视频正在写入</strong>
                              <small>仿真完成后播放器会自动出现</small>
                            </div>
                          ) : (
                            <a href={source} target="_blank" rel="noreferrer">
                              <img src={source} alt={presentation.label} loading="lazy" />
                            </a>
                          )}
                          <figcaption>
                            <span title={artifact.path}>{artifact.name}</span>
                            <span>{formatBytes(artifact.bytes)}</span>
                            {isVideo && videoReady && (
                              <a href={source} target="_blank" rel="noreferrer">单独打开</a>
                            )}
                          </figcaption>
                        </figure>
                      );
                    })}
                  </div>
                )}
                {!!job.artifacts.length && (
                  <details className="technical-artifacts">
                    <summary>按需查看中间模型数据与全部单项下载（{job.artifacts.length}）</summary>
                    {job.body_summary && (
                      <div className="artifact-summary-grid">
                        <span>命中体型 <strong>{job.body_summary.inferred_type}</strong></span>
                        <span>实际基础体型 <strong>{job.body_summary.mapped_type}</strong></span>
                        <span>基础人体 ID <strong>{job.body_summary.base_source_id}</strong></span>
                        <span>人体字段 <strong>{job.body_summary.body_field_count} 项</strong></span>
                        <span>臀围来源 <strong>{job.body_summary.hips_source}</strong></span>
                      </div>
                    )}
                    <div>
                      {job.artifacts.map((artifact) => (
                        <a key={artifact.path} href={artifact.url} download>
                          <span>{artifact.path}</span>
                          <small>{formatBytes(artifact.bytes)}</small>
                        </a>
                      ))}
                    </div>
                  </details>
                )}
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function HomeWorkflowSummary() {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const response = await fetch("/api/jobs", { cache: "no-store" });
        if (!response.ok) return;
        const payload = (await response.json()) as { jobs?: JobRecord[] };
        if (active) {
          setJobs(payload.jobs ?? []);
          setLoaded(true);
        }
      } catch {
        if (active) setLoaded(true);
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const job =
    jobs.find((item) => item.state === "running" || item.state === "queued") ??
    jobs.find((item) => item.state === "completed") ??
    jobs[0];
  const stage = job ? currentJobStage(job) : null;
  const activeJob = job?.state === "running" || job?.state === "queued";
  const resource =
    (activeJob ? job?.runtime?.label : job?.runtime?.gpu_name) ??
    (loaded ? "等待任务" : "正在读取");

  return (
    <section className="metrics current-workflow-summary" aria-label="当前完整处理概览" aria-live="polite">
      <div>
        <strong className="metric-text">{job?.name ?? (loaded ? "暂无任务" : "读取中")}</strong>
        <span>{activeJob ? "当前任务" : "最近完整处理"}</span>
      </div>
      <div>
        <strong>{job ? `${job.progress}%` : "—"}</strong>
        <span>处理进度</span>
      </div>
      <div>
        <strong className="metric-text">{stage?.label ?? "等待提交"}</strong>
        <span>当前步骤</span>
      </div>
      <div>
        <strong>{job ? `${job.image_count} 张` : "—"}</strong>
        <span>本次输入</span>
      </div>
      <div className="metric-wide">
        <strong>{resource}</strong>
        <span>当前计算资源</span>
      </div>
    </section>
  );
}

type LabView = "landing" | "workflow" | "body-customizer" | "body-pattern" | "results";

const SHOW_EXPERIMENTAL_BODY_TOOLS = false;

const VIEW_COPY: Record<Exclude<LabView, "landing">, { eyebrow: string; title: string; detail: string }> = {
  workflow: {
    eyebrow: "END-TO-END WORKFLOW",
    title: "完整处理与生成记录",
    detail: "上传图片、填写人体尺寸，持续查看任务步骤并按生成顺序检查全部产物。",
  },
  "body-customizer": {
    eyebrow: "BODY ROUTE A",
    title: "三维定制人体",
    detail: "用人体尺寸和语义描述生成可注册、可驱动动作的 SMPL-X 人体。",
  },
  "body-pattern": {
    eyebrow: "BODY ROUTE B",
    title: "制版人体尺寸补全",
    detail: "按 GB/T 基础人体规则补全 GarmentCode 解码器所需的 26 项尺寸。",
  },
  results: {
    eyebrow: "RESULT ATLAS",
    title: "官方示例",
    detail: "集中核对输入图、二维板片、静态垂坠结果与动态布料视频。",
  },
};

function LabPage({ view }: { view: LabView }) {
  const [filter, setFilter] = useState<FilterType>("all");
  const [query, setQuery] = useState("");
  const [lightbox, setLightbox] = useState<{ src: string; label: string } | null>(null);

  const visibleCases = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return CASES.filter((item, index) => {
      const isWholebody = item.garments.includes("wholebody");
      const filterMatches =
        filter === "all" ||
        (filter === "wholebody" && isWholebody) ||
        (filter === "separates" && !isWholebody);
      const searchMatches =
        !normalizedQuery ||
        item.id.toLowerCase().includes(normalizedQuery) ||
        String(index + 1).padStart(2, "0").includes(normalizedQuery);
      return filterMatches && searchMatches;
    });
  }, [filter, query]);

  return (
    <main>
      <header className="hero">
        <nav className="nav-shell" aria-label="页面导航">
          <a className="brand" href="/" aria-label="返回功能首页">
            <span className="brand-mark" aria-hidden="true">
              CG
            </span>
            <span>
              <strong>ChatGarment</strong>
              <small>AI GARMENT SYSTEM</small>
            </span>
          </a>
          <div className="page-nav">
            {(
              [
                ["/", "首页", "landing"],
                ["/workflow", "完整处理", "workflow"],
                ...(SHOW_EXPERIMENTAL_BODY_TOOLS
                  ? [
                      ["/body-customizer", "定制人体", "body-customizer"],
                      ["/body-pattern", "尺寸补全", "body-pattern"],
                    ]
                  : []),
                ["/results", "官方示例", "results"],
              ] as [string, string, LabView][]
            ).map(([href, label, itemView]) => (
              <a key={href} href={href} className={view === itemView ? "active" : ""}>
                {label}
              </a>
            ))}
          </div>
        </nav>

        {view === "landing" ? (
        <section className="hero-content" id="top">
          <div className="hero-copy">
            <p className="eyebrow">2D PATTERN → 3D GARMENT</p>
            <h1>
              从一张服装图，
              <span>走到可缝合的三维布料。</span>
            </h1>
            <p className="hero-description">
              使用 ChatGarment 生成二维板片，再由 GarmentCode 完成缝合、物理垂坠与正反面渲染。
            </p>
            <div className="hero-actions">
              <a className="primary-action" href="/workflow">
                开始完整处理
                <span aria-hidden="true">→</span>
              </a>
              <a className="secondary-action" href="/results">
                查看官方示例
              </a>
            </div>
          </div>

          <div className="pipeline-panel" aria-label="服装生成流程">
            <div className="pipeline-header">
              <span>GARMENT GENERATION PIPELINE</span>
              <span className="pipeline-status">已贯通</span>
            </div>
            <ol className="pipeline-list">
              <li>
                <span className="step-number">01</span>
                <div>
                  <strong>图像理解</strong>
                  <small>服装语义与结构解析</small>
                </div>
                <span className="step-state">DONE</span>
              </li>
              <li>
                <span className="step-number">02</span>
                <div>
                  <strong>二维制版</strong>
                  <small>参数化板片与缝合线</small>
                </div>
                <span className="step-state">DONE</span>
              </li>
              <li>
                <span className="step-number">03</span>
                <div>
                  <strong>静态缝合与垂坠</strong>
                  <small>GarmentCode · NVIDIA Warp</small>
                </div>
                <span className="step-state">DONE</span>
              </li>
              <li>
                <span className="step-number">04</span>
                <div>
                  <strong>动态布料与动作</strong>
                  <small>ContourCraft-CG · SMPL-X</small>
                </div>
                <span className="step-state">DONE</span>
              </li>
            </ol>
          </div>
        </section>
        ) : (
          <section className="subpage-hero-content" id="top">
            <a href="/" className="back-home">← 返回功能首页</a>
            <p className="eyebrow">{VIEW_COPY[view].eyebrow}</p>
            <h1>{VIEW_COPY[view].title}</h1>
            <p>{VIEW_COPY[view].detail}</p>
          </section>
        )}
      </header>

      {view === "workflow" && <WorkflowStudio />}

      {view === "landing" && (
      <>
      <HomeWorkflowSummary />

      <section className="body-solutions product-routes" id="product-routes">
        <div>
          <p className="eyebrow">FUNCTION PAGES</p>
          <h2>选择要使用的功能，每项工作在独立页面完成</h2>
        </div>
        <div className="solution-cards">
          <a href="/workflow">
            <span>服装生成</span>
            <strong>图片 → 制版 → 三维产物</strong>
            <small>上传图片和人体尺寸，查看进度与全部生成记录</small>
          </a>
          {SHOW_EXPERIMENTAL_BODY_TOOLS && (
            <>
              <a href="/body-customizer">
                <span>方案 A</span>
                <strong>SHAPY → SMPL-X</strong>
                <small>生成可注册、驱动动作和布料仿真的三维人体</small>
              </a>
              <a href="/body-pattern">
                <span>方案 B</span>
                <strong>GB/T 六模板 → GarmentCode YAML</strong>
                <small>用身高与三围快速补全解码器需要的 26 项人体尺寸</small>
              </a>
            </>
          )}
          <a href="/results">
            <span>结果图册</span>
            <strong>十组官方示例</strong>
            <small>检查二维板片、静态垂坠和动态视频</small>
          </a>
        </div>
      </section>
      </>
      )}

      {view === "body-customizer" && <BodyCustomizer />}
      {view === "body-pattern" && <GBTPatternBodyAdapter />}

      {view === "results" && (
      <section className="results-section" id="results">
        <div className="section-heading">
          <div>
            <p className="eyebrow">RESULT ATLAS</p>
            <h2>十组官方示例，一页完成核对</h2>
            <p>点击任意图片可放大；每个服装结果均可直接下载制版 JSON 与模拟后的 OBJ 网格。</p>
          </div>
          <span className="result-count">显示 {visibleCases.length} / {CASES.length} 组</span>
        </div>

        <div className="toolbar" aria-label="结果筛选">
          <div className="filter-group">
            {(
              [
                ["all", "全部"],
                ["separates", "上下装"],
                ["wholebody", "连体装"],
              ] as [FilterType, string][]
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={filter === value ? "active" : ""}
                onClick={() => setFilter(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="search-box">
            <span aria-hidden="true">⌕</span>
            <input
              type="search"
              placeholder="搜索编号或哈希…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
        </div>

        <div className="case-list">
          {visibleCases.map((item) => {
            const caseNumber = CASES.indexOf(item) + 1;
            const hash = item.id.replace("valid_garment_", "");
            return (
              <article className="case-card" key={item.id}>
                <header className="case-header">
                  <div className="case-identity">
                    <span className="case-index">{String(caseNumber).padStart(2, "0")}</span>
                    <div>
                      <h3>示例 {String(caseNumber).padStart(2, "0")}</h3>
                      <code title={hash}>{hash.slice(0, 12)}…{hash.slice(-6)}</code>
                    </div>
                  </div>
                  <div className="case-tags">
                    {item.garments.map((garment) => (
                      <span key={garment}>{garmentLabels[garment]}</span>
                    ))}
                    <span className="complete-tag">3D 已完成</span>
                    <span className="dynamic-tag">
                      {DYNAMIC_RESULTS[item.id] ? "动态已完成" : "动态待生成"}
                    </span>
                  </div>
                </header>

                <div className="comparison-layout">
                  <div className="input-result">
                    <div className="media-label">
                      <span>INPUT</span>
                      输入图像
                    </div>
                    <ResultImage
                      src={`/cases/${item.id}/gt_image.png`}
                      alt={`示例 ${caseNumber} 的输入服装图`}
                      kind="input"
                      onOpen={() =>
                        setLightbox({
                          src: `/cases/${item.id}/gt_image.png`,
                          label: `示例 ${String(caseNumber).padStart(2, "0")} · 输入图像`,
                        })
                      }
                    />
                  </div>

                  <div className="garment-results">
                    {item.garments.map((garment) => {
                      const paths = assetPaths(item.id, garment);
                      const shortHash = item.id.replace("valid_garment_", "").slice(0, 8);
                      const metric = SIM_METRICS[`${shortHash}:${garment}`];
                      return (
                        <section className="garment-row" key={garment}>
                          <div className="garment-title">
                            <div>
                              <span className="garment-dot" />
                              <strong>{garmentLabels[garment]}</strong>
                            </div>
                            <div className="download-links">
                              <a href={paths.spec} download>
                                JSON
                              </a>
                              <a href={paths.mesh} download>
                                OBJ
                              </a>
                            </div>
                          </div>
                          <div className="simulation-meta" aria-label={`${garmentLabels[garment]}仿真统计`}>
                            <span>{metric.seconds.toFixed(2)} 秒</span>
                            <span>{metric.frames} 帧收敛</span>
                            <span className="clean-metric">身体穿插 {metric.bodyCollisions}</span>
                            <span className={metric.selfCollisions > 0 ? "notice-metric" : "clean-metric"}>
                              自碰撞 {metric.selfCollisions}
                            </span>
                          </div>
                          <div className="output-grid">
                            {[
                              ["2D PATTERN", "二维版片", paths.pattern, "pattern"],
                              ["3D FRONT", "垂坠正面", paths.front, "render"],
                              ["3D BACK", "垂坠背面", paths.back, "render"],
                            ].map(([english, chinese, src, kind]) => (
                              <div className="output-result" key={english}>
                                <div className="media-label">
                                  <span>{english}</span>
                                  {chinese}
                                </div>
                                <ResultImage
                                  src={src}
                                  alt={`示例 ${caseNumber} ${garmentLabels[garment]}${chinese}`}
                                  kind={kind as "pattern" | "render"}
                                  onOpen={() =>
                                    setLightbox({
                                      src,
                                      label: `示例 ${String(caseNumber).padStart(2, "0")} · ${garmentLabels[garment]} · ${chinese}`,
                                    })
                                  }
                                />
                              </div>
                            ))}
                          </div>
                        </section>
                      );
                    })}
                  </div>
                </div>
                <DynamicPreview caseId={item.id} caseNumber={caseNumber} />
              </article>
            );
          })}
        </div>

        {visibleCases.length === 0 && (
          <div className="empty-results">
            <span>⌕</span>
            <strong>没有找到对应结果</strong>
            <p>试试清空搜索，或切换服装类型。</p>
          </div>
        )}
      </section>
      )}

      {lightbox && (
        <div
          className="lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={lightbox.label}
          onClick={() => setLightbox(null)}
        >
          <button type="button" className="lightbox-close" onClick={() => setLightbox(null)}>
            ×
          </button>
          <figure onClick={(event) => event.stopPropagation()}>
            <img src={lightbox.src} alt={lightbox.label} />
            <figcaption>{lightbox.label}</figcaption>
          </figure>
        </div>
      )}
    </main>
  );
}

export function LandingPage() {
  return <LabPage view="landing" />;
}

export function WorkflowPage() {
  return <LabPage view="workflow" />;
}

export function BodyCustomizerPage() {
  return <LabPage view={SHOW_EXPERIMENTAL_BODY_TOOLS ? "body-customizer" : "landing"} />;
}

export function BodyPatternPage() {
  return <LabPage view={SHOW_EXPERIMENTAL_BODY_TOOLS ? "body-pattern" : "landing"} />;
}

export function ResultsPage() {
  return <LabPage view="results" />;
}
