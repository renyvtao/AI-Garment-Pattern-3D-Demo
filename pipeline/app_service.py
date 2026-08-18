#!/usr/bin/env python3
"""Persistent single-GPU job service for the ChatGarment demo application."""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from dxf_export import export_specification


JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGES = 20
MAX_FILE_BYTES = 25 * 1024 * 1024
MIN_FREE_BYTES = 8 * 1024 * 1024 * 1024

FINAL_STATES = {"completed", "failed", "cancelled", "trashed"}
VISIBLE_ARTIFACT_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".svg", ".mp4", ".obj",
    ".mtl", ".json", ".yaml", ".yml", ".txt", ".npz", ".pkl",
    ".pickle", ".zip", ".blend", ".pdf", ".dxf",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    return parser.parse_args()


def safe_job_id(value: str) -> str:
    if not JOB_ID_PATTERN.fullmatch(value):
        raise ValueError("invalid job id")
    return value


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """Return an inclusive byte range for one RFC 7233 range request."""
    if not value:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or size <= 0:
        raise ValueError("invalid byte range")
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise ValueError("invalid byte range")
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("invalid byte range")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
        if start >= size or end < start:
            raise ValueError("range outside file")
        end = min(end, size - 1)
    return start, end


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except FileNotFoundError:
            pass
    return total


def summarize_simulation_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in entries if item.get("status") == "completed"]
    body_collisions: list[int] = []
    self_collisions: list[int] = []
    frames: list[int] = []
    warnings: set[str] = set()
    for item in completed:
        stats = item.get("simulation_stats") or {}
        body_collisions.extend(
            int(value) for value in (stats.get("body_collisions") or {}).values()
        )
        self_collisions.extend(
            int(value) for value in (stats.get("self_collisions") or {}).values()
        )
        frames.extend(int(value) for value in (stats.get("fin_frame") or {}).values())
        for name, affected in (stats.get("fails") or {}).items():
            if affected:
                warnings.add(str(name))
    return {
        "case_count": len(entries),
        "completed_count": len(completed),
        "body_collisions": sum(body_collisions),
        "self_collisions": sum(self_collisions),
        "min_frames": min(frames) if frames else None,
        "max_frames": max(frames) if frames else None,
        "warnings": sorted(warnings),
    }


def build_result_bundle(root: Path) -> Path:
    bundle = root / "result_bundle.zip"
    temporary = root / ".result_bundle.zip.part"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path in {temporary, bundle}:
                continue
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] == "work":
                continue
            archive.write(path, relative.as_posix())
    temporary.replace(bundle)
    return bundle


def safe_child(root: Path, relative: str) -> Path:
    target = (root / unquote(relative)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path escapes storage root") from exc
    return target


def prepare_chatgarment_inputs(source_dir: Path, target_dir: Path) -> list[Path]:
    """Convert uploaded images to the PNG format accepted by official inference."""
    target_dir.mkdir(parents=True, exist_ok=False)
    prepared: list[Path] = []
    for source in sorted(path for path in source_dir.iterdir() if path.is_file()):
        if source.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            continue
        target = target_dir / f"{source.stem}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(target, format="PNG")
        prepared.append(target)
    if not prepared:
        raise ValueError("no supported images were prepared for ChatGarment inference")
    return prepared


def resolve_suit_button_count(result_path: Path) -> dict[str, Any]:
    """Resolve the model-predicted button count for K62's two closure slots."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    predicted = payload.get("predicted")
    if not isinstance(predicted, dict) or "button_count" not in predicted:
        raise ValueError(f"missing predicted.button_count: {result_path}")
    raw_value = predicted["button_count"]
    if isinstance(raw_value, bool):
        raise ValueError(f"invalid predicted.button_count: {raw_value!r}")
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid predicted.button_count: {raw_value!r}") from exc
    integer = int(numeric)
    if numeric != integer:
        raise ValueError(f"predicted.button_count must be an integer: {raw_value!r}")
    resolved = min(2, max(0, integer))
    return {
        "source": "ChatGarment suit result predicted.button_count",
        "result_path": str(result_path.resolve()),
        "model_button_count": integer,
        "resolved_button_count": resolved,
        "supported_k62_button_count": [0, 2],
        "clamped": resolved != integer,
    }


def load_lower_specifications(
    chatgarment_root: Path,
    vis_root: Path,
    spec_list_path: Path,
) -> list[Path]:
    """Load only official ChatGarment lower-body specs from a generated list."""
    payload = json.loads(spec_list_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"specification list must be an array: {spec_list_path}")
    vis_root = vis_root.resolve()
    selected: list[Path] = []
    for raw in payload:
        if not isinstance(raw, str):
            raise ValueError(f"invalid specification path: {raw!r}")
        path = Path(raw)
        if not path.is_absolute():
            path = chatgarment_root / path
        path = path.resolve()
        try:
            relative = path.relative_to(vis_root)
        except ValueError as exc:
            raise ValueError(f"specification escapes inference output: {path}") from exc
        is_lower = "_lower" in path.stem or any(
            part.endswith("_lower") for part in relative.parts
        )
        if is_lower:
            if not path.is_file():
                raise FileNotFoundError(path)
            selected.append(path)
    return sorted(set(selected))


def read_lower_garment_type(design_path: Path) -> str | None:
    """Read design.meta.bottom.v without adding a YAML dependency to the service."""
    keys: list[str] = []
    target = ("design", "meta", "bottom", "v")
    for raw_line in design_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = re.match(r"^(\s*)([A-Za-z0-9_-]+):(?:\s*(.*))?$", raw_line)
        if not match:
            continue
        indent, key, value = match.groups()
        level = len(indent.expandtabs(2)) // 2
        keys = keys[:level]
        keys.append(key)
        if tuple(keys) == target:
            scalar = (value or "").strip().strip("'\"")
            return None if scalar.lower() in {"", "null", "none", "~"} else scalar
    return None


class JobStore:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self.runs_root = self.data_root / "runs"
        self.trash_root = self.data_root / "trash"
        self.db_path = self.data_root / "jobs.sqlite3"
        self.lock = threading.RLock()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.trash_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    step TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    delete_after_cancel TEXT,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    trash_path TEXT
                )
                """
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS jobs_updated_idx ON jobs(updated_at DESC)"
            )
            db.execute(
                """
                UPDATE jobs
                SET state='failed', step='interrupted', progress=progress,
                    message='Service restarted while this task was running',
                    error='interrupted_by_service_restart', updated_at=?
                WHERE state='running'
                """,
                (utc_now(),),
            )

    def job_root(self, job_id: str) -> Path:
        return self.runs_root / safe_job_id(job_id)

    def create(self, name: str, config: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO jobs
                    (id, name, state, step, progress, message, created_at,
                     updated_at, config_json)
                VALUES (?, ?, 'queued', 'queued', 0, 'Waiting for GPU worker',
                        ?, ?, ?)
                """,
                (job_id, name[:100] or job_id[:8], now, now,
                 json.dumps(config, ensure_ascii=False)),
            )
        return job_id

    def update(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key}=?" for key in values)
        params = list(values.values()) + [safe_job_id(job_id)]
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {columns} WHERE id=?", params)

    def row(self, job_id: str) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute(
                "SELECT * FROM jobs WHERE id=?", (safe_job_id(job_id),)
            ).fetchone()

    def rows(self, include_trashed: bool = False) -> list[sqlite3.Row]:
        query = "SELECT * FROM jobs"
        if not include_trashed:
            query += " WHERE state!='trashed'"
        query += " ORDER BY created_at DESC LIMIT 200"
        with self.connect() as db:
            return list(db.execute(query).fetchall())

    def next_queued(self) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()


class Pipeline:
    def __init__(self, project_root: Path, store: JobStore):
        self.root = project_root.resolve()
        self.store = store
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.runtime_profile = self.empty_runtime_profile()

    def start(self) -> None:
        self.runtime_profile = self.detect_runtime_profile()
        self.thread.start()

    @staticmethod
    def empty_runtime_profile() -> dict[str, Any]:
        return {
            "gpu_name": None,
            "chatgarment_cuda": False,
            "static_cuda": False,
            "ccraft_cuda": False,
            "nvidia_egl": False,
            "egl_renderer": None,
        }

    @staticmethod
    def probe_torch_cuda(python: Path) -> bool:
        if not python.is_file():
            return False
        try:
            result = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import torch; print('1' if torch.cuda.is_available() else '0')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and result.stdout.strip().endswith("1")

    def probe_nvidia_egl(self, blender: Path) -> tuple[bool, str | None]:
        if not blender.is_file():
            return False, None
        vendor_files = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
        selected_nvidia_vendor = False
        for value in vendor_files.split(":"):
            if not value:
                continue
            try:
                vendor = json.loads(Path(value).read_text(encoding="utf-8"))
                library = str(vendor.get("ICD", {}).get("library_path", ""))
                if "nvidia" in library.lower():
                    selected_nvidia_vendor = True
                    break
            except (OSError, ValueError, TypeError):
                continue
        expression = (
            "import bpy;"
            "s=bpy.context.scene;"
            "s.render.engine='BLENDER_EEVEE';"
            "s.render.resolution_x=1;"
            "s.render.resolution_y=1;"
            "s.render.resolution_percentage=100;"
            "bpy.ops.render.render();"
            "print('[CHATGARMENT_EGL_OK]')"
        )
        try:
            result = subprocess.run(
                [str(blender), "--background", "--python-expr", expression],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False, None
        output = f"{result.stdout}\n{result.stderr}"
        render_succeeded = result.returncode == 0 and "[CHATGARMENT_EGL_OK]" in output
        verified = render_succeeded and selected_nvidia_vendor
        renderer = "NVIDIA EGL (GLVND)" if verified else None
        return verified, renderer

    def detect_runtime_profile(self) -> dict[str, Any]:
        profile = self.empty_runtime_profile()
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if result.returncode == 0 and names:
                profile["gpu_name"] = names[0]
        except (OSError, subprocess.SubprocessError):
            pass

        paths = self.paths()
        profile["ccraft_cuda"] = self.probe_torch_cuda(paths["ccraft_python"])
        profile["static_cuda"] = self.probe_torch_cuda(paths["venv_python"])
        chatgarment_python = Path(shutil.which("python") or sys.executable)
        profile["chatgarment_cuda"] = self.probe_torch_cuda(chatgarment_python)
        profile["nvidia_egl"], profile["egl_renderer"] = self.probe_nvidia_egl(
            paths["blender"]
        )
        print(
            json.dumps({"event": "runtime_profile", **profile}, ensure_ascii=False),
            flush=True,
        )
        return profile

    def require_accelerator(self, capability: str, stage: str) -> None:
        if not self.runtime_profile.get("gpu_name") or not self.runtime_profile.get(capability):
            raise RuntimeError(
                f"{stage} requires CUDA, but the configured runtime did not verify a CUDA GPU"
            )

    def require_nvidia_egl(self, stage: str) -> None:
        if not self.runtime_profile.get("nvidia_egl"):
            raise RuntimeError(
                f"{stage} requires NVIDIA EGL, but the startup render probe did not verify it"
            )

    def runtime_for_job(
        self,
        *,
        state: str,
        step: str,
        message: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        gpu_name = self.runtime_profile.get("gpu_name")
        gpu_label = f"GPU · {gpu_name}" if gpu_name else "GPU 不可用"
        is_suit = config.get("garment_mode") == "mens_suit"
        result: dict[str, Any] = {
            "kind": "idle",
            "label": "等待计算资源",
            "detail": "任务尚未开始",
            "gpu_name": gpu_name,
            "verified": False,
        }
        if state in FINAL_STATES:
            result.update(
                kind="finished",
                label="计算已结束",
                detail=(
                    f"运行设备：{gpu_name}"
                    if gpu_name
                    else "本实例当前未检测到 GPU"
                ),
                verified=bool(gpu_name),
            )
            return result
        if step == "queued":
            result.update(
                label=f"等待 {gpu_name}" if gpu_name else "等待 GPU",
                detail="单卡任务队列会依次执行",
                verified=bool(gpu_name),
            )
        elif step == "preparing":
            if is_suit:
                if config.get("action_id") == "none":
                    result.update(
                        kind="cpu",
                        label="CPU · 国标尺寸适配",
                        detail="匹配基础人体并生成 GarmentCode 26 字段人体 YAML",
                        verified=True,
                    )
                else:
                    result.update(
                        kind="mixed",
                        label=f"CPU + {gpu_label}",
                        detail="CPU 补全制版尺寸；CUDA 生成标准男性 SMPL-X 与官方动作人体",
                        verified=bool(
                            self.runtime_profile.get("ccraft_cuda")
                            and self.runtime_profile.get("nvidia_egl")
                        ),
                    )
            elif config.get("body_mode") == "custom":
                result.update(
                    kind="mixed",
                    label=f"CPU + {gpu_label}",
                    detail="CPU 优化测量值；CUDA 生成 SMPL-X；NVIDIA EGL 生成人体预览",
                    verified=bool(
                        self.runtime_profile.get("ccraft_cuda")
                        and self.runtime_profile.get("nvidia_egl")
                    ),
                )
            else:
                result.update(
                    kind="gpu",
                    label=gpu_label,
                    detail="CUDA 生成 SMPL-X；NVIDIA EGL 生成人体预览",
                    verified=bool(
                        self.runtime_profile.get("ccraft_cuda")
                        and self.runtime_profile.get("nvidia_egl")
                    ),
                )
        elif step == "chatgarment":
            result.update(
                kind="gpu",
                label=gpu_label,
                detail=(
                    (
                        "ChatGarment 官方原权重 · CUDA 下装识别"
                        if "下装" in message
                        else "ChatGarment 官方权重 + 男西装 LoRA · CUDA 上装推理"
                    )
                    if is_suit
                    else "ChatGarment 官方权重 · CUDA 推理"
                ),
                verified=bool(self.runtime_profile.get("chatgarment_cuda")),
            )
        elif step == "static_3d":
            result.update(
                kind="gpu",
                label=gpu_label,
                detail=(
                    (
                        "官方下装 GarmentCode + NVIDIA Warp + NVIDIA EGL 静态渲染"
                        if "下装" in message
                        else "K62 装配适配 + NVIDIA Warp 布料仿真 + NVIDIA EGL 静态渲染"
                    )
                    if is_suit
                    else "NVIDIA Warp 布料仿真 + NVIDIA EGL 静态渲染"
                ),
                verified=bool(
                    self.runtime_profile.get(
                        "ccraft_cuda" if is_suit else "static_cuda"
                    )
                    and self.runtime_profile.get("nvidia_egl")
                ),
            )
        elif step == "dynamic_preparation":
            result.update(
                kind="cpu",
                label="CPU · 网格与文件整理",
                detail="该步骤是轻量几何预处理，使用 CPU 更合适",
                verified=True,
            )
        elif step == "dynamic_3d":
            rendering = "逐帧渲染" in message
            result.update(
                kind="gpu",
                label=gpu_label,
                detail=(
                    "Blender EEVEE · NVIDIA EGL 视频渲染"
                    if rendering
                    else "ContourCraft · CUDA 动态布料仿真"
                ),
                verified=bool(
                    self.runtime_profile.get("ccraft_cuda")
                    and self.runtime_profile.get("nvidia_egl")
                ),
            )
        elif step == "collecting":
            result.update(
                kind="cpu",
                label="CPU + 数据盘",
                detail="整理产物并压缩下载包；无需占用 GPU",
                verified=True,
            )
        return result

    def paths(self) -> dict[str, Path]:
        dynamic = self.root / "dynamic3d"
        assets = dynamic / "assets" / "ccraft_data"
        return {
            "chatgarment": self.root / "ChatGarment",
            "venv_python": self.root / "venv" / "bin" / "python",
            "ccraft_python": dynamic / "envs" / "ccraft" / "bin" / "python",
            "garment_sim": self.root / "scripts" / "garment_sim_runner.py",
            "garmentcode": self.root / "GarmentCodeRC",
            "sim_config": self.root / "GarmentCodeRC" / "assets" / "Sim_props" / "default_sim_props.yaml",
            "sim_system": self.root / "GarmentCodeRC" / "system.json",
            "prepare_dynamic": dynamic / "scripts" / "prepare_dynamic_inputs.py",
            "dynamic_batch": dynamic / "scripts" / "run_dynamic_batch.py",
            "contourcraft": dynamic / "src" / "ContourCraft-CG",
            "hood_data": assets,
            "checkpoint": assets / "trained_models" / "contourcraft.pth",
            "blender": dynamic / "blender-3.6.14-linux-x64" / "blender",
            "body_generator": dynamic / "body_customization" / "body_generator.py",
            "shapy_data": self.root / "third_party" / "shapy" / "data",
            "smplx_models": assets / "aux_data" / "body_models" / "smplx",
            "suit_inference": self.root / "suit_finetune" / "run_suit_poc_inference.py",
            "suit_lora": (
                self.root / "ChatGarment" / "runs" / "suit_poc_lora_v1"
                / "suit_lora_state.bin"
            ),
            "suit_3d_adapter": self.root / "suit_finetune" / "build_suit_3d_spec.py",
            "suit_3d_garmentcode": self.root / "GarmentCodeRC_K62_3D",
            "suit_3d_golden_spec": (
                self.root / "incoming" / "K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
                / "01_GOLDEN_BASE" / "K62_specification.json"
            ),
            "suit_3d_sim_config": (
                self.root / "incoming" / "K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
                / "01_GOLDEN_BASE" / "sim_props.yaml"
            ),
            "suit_3d_system": (
                self.root / "incoming" / "K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
                / "REPLAY_OUTPUT" / "system.generated.json"
            ),
            "suit_dynamic_prepare": (
                dynamic / "scripts" / "prepare_suit_dynamic_inputs.py"
            ),
            "suit_static_outfit_render": (
                dynamic / "scripts" / "render_static_outfit.py"
            ),
        }

    def worker(self) -> None:
        while not self.stop_event.is_set():
            row = self.store.next_queued()
            if row is None:
                self.stop_event.wait(1.0)
                continue
            try:
                self.run_job(row["id"])
            except Exception:
                traceback.print_exc()

    def check_cancel(self, job_id: str) -> None:
        row = self.store.row(job_id)
        if row and row["cancel_requested"]:
            raise InterruptedError("cancel_requested")

    def set_stage(
        self,
        job_id: str,
        step: str,
        progress: int,
        message: str,
    ) -> None:
        self.store.update(
            job_id,
            state="running",
            step=step,
            progress=progress,
            message=message,
        )

    def run_command(
        self,
        job_id: str,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        frame_progress: dict[str, int] | None = None,
    ) -> None:
        job_root = self.store.job_root(job_id)
        log_path = job_root / "logs" / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("\n[COMMAND] " + " ".join(command) + "\n")
            log.flush()
            monitor_offset = log.tell()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            if frame_progress:
                frames_per_case = max(1, frame_progress["frames_per_case"])
                case_count = max(1, frame_progress["case_count"])
                progress_start = frame_progress["progress_start"]
                progress_end = frame_progress["progress_end"]
                simulation_end = frame_progress.get("simulation_end", progress_start)
                total_frames = frames_per_case * case_count
                completed_cases = 0
                last_frame = 0
                simulation_completed_cases = 0
                last_simulation_frame = 0
                last_simulation_total = frames_per_case
                last_reported = progress_start
                render_started = False
                while process.poll() is None:
                    time.sleep(2)
                    try:
                        with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                            reader.seek(monitor_offset)
                            chunk = reader.read()
                            values = [int(value) for value in re.findall(r"Fra:(\d+)", chunk)]
                            simulation_values = [
                                (int(current), int(total))
                                for current, total in re.findall(
                                    r"(\d+)/(\d+)\s+\[",
                                    chunk,
                                )
                                if int(total) > 0
                            ]
                        if simulation_values and not values and not render_started:
                            for current, reported_total in simulation_values:
                                if (
                                    current < last_simulation_frame
                                    and last_simulation_frame >= last_simulation_total // 2
                                ):
                                    simulation_completed_cases = min(
                                        case_count - 1,
                                        simulation_completed_cases + 1,
                                    )
                                last_simulation_frame = current
                                last_simulation_total = reported_total
                            simulation_total_frames = last_simulation_total * case_count
                            simulated_frames = min(
                                simulation_total_frames,
                                simulation_completed_cases * last_simulation_total
                                + last_simulation_frame,
                            )
                            progress = progress_start + int(
                                (simulation_end - progress_start)
                                * simulated_frames
                                / simulation_total_frames
                            )
                            if progress > last_reported:
                                self.store.update(
                                    job_id,
                                    progress=progress,
                                    message=(
                                        "ContourCraft GPU 动态仿真："
                                        f"{simulated_frames}/{simulation_total_frames} 帧"
                                    ),
                                )
                                last_reported = progress
                        monitor_offset = log_path.stat().st_size
                        if values or render_started:
                            for frame in values:
                                if frame < last_frame and last_frame >= frames_per_case // 2:
                                    completed_cases = min(case_count - 1, completed_cases + 1)
                                last_frame = frame
                            finished_frames = min(
                                total_frames,
                                completed_cases * frames_per_case + last_frame,
                            )
                            progress = simulation_end + int(
                                (progress_end - simulation_end)
                                * finished_frames
                                / total_frames
                            )
                            first_render_update = bool(values) and not render_started
                            if values:
                                render_started = True
                            if progress > last_reported or first_render_update:
                                self.store.update(
                                    job_id,
                                    progress=progress,
                                    message=(
                                        "动态视频逐帧渲染："
                                        f"{finished_frames}/{total_frames} 帧"
                                    ),
                                )
                                last_reported = progress
                    except (OSError, ValueError):
                        pass
                code = process.wait()
            else:
                code = process.wait()
        if code:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(
                f"command failed with exit code {code}: {' '.join(command[:3])}\n{tail}"
            )
        self.check_cancel(job_id)

    def write_body_inputs(
        self,
        job_id: str,
        config: dict[str, Any],
        job_root: Path,
    ) -> Path:
        body_root = job_root / "outputs" / "body"
        scheme_b_root = job_root / "outputs" / "body_measurements"
        body_request = {
            "name": config.get("name", ""),
            "gender": config["gender"],
            "semantic_profile": config.get("semantic_profile")
            or (config["gender"] if config["gender"] != "neutral" else "female"),
            "height_cm": config["height_cm"],
            "weight_kg": config.get("weight_kg"),
            "chest_cm": config["chest_cm"],
            "waist_cm": config["waist_cm"],
            "hips_cm": config.get("hips_cm"),
            "attributes": config.get("attributes", {}),
            "optimize_measurements": True,
            "action_id": config["action_id"],
        }
        # Scheme B is always produced for GarmentCode audit/display.  Neutral
        # uses the selected semantic profile to pick the standards table.
        sys.path.insert(0, str(self.root / "dynamic3d" / "body_customization"))
        from gbt1335_adapter import write_garmentcode_artifacts

        gbt_payload = {
            "gender": (
                body_request["gender"]
                if body_request["gender"] in {"female", "male"}
                else body_request["semantic_profile"]
            ),
            "height_cm": body_request["height_cm"],
            "chest_cm": body_request["chest_cm"],
            "waist_cm": body_request["waist_cm"],
            "hips_cm": body_request["hips_cm"],
            "boundary_policy": config.get("boundary_policy", "extrapolate"),
        }
        garmentcode_body, _, _ = write_garmentcode_artifacts(gbt_payload, scheme_b_root)
        if body_request["hips_cm"] is None:
            body_request["hips_cm"] = garmentcode_body["hips"]

        self.store.update(
            job_id,
            progress=5,
            message="人体尺寸已补全，正在生成 SMPL-X 人体与预览",
        )

        body_request_path = job_root / "body_request.json"
        body_request_path.write_text(
            json.dumps(body_request, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        debug_args: list[str] = []
        if config["body_mode"] == "preset":
            debug_path = job_root / "preset_zero_betas.json"
            debug_path.write_text(json.dumps({"betas": [0.0] * 10}), encoding="utf-8")
            debug_args = ["--debug-betas", str(debug_path)]
        paths = self.paths()
        command = [
            str(paths["ccraft_python"]),
            str(paths["body_generator"]),
            "--request", str(body_request_path),
            "--output-dir", str(body_root),
            "--project-root", str(self.root),
            "--shapy-data-root", str(paths["shapy_data"]),
            "--device", "cuda:0",
            *debug_args,
        ]
        self.run_command(job_id, command, cwd=self.root)
        return body_root

    def write_suit_body_measurements(
        self,
        job_id: str,
        config: dict[str, Any],
        job_root: Path,
    ) -> Path:
        """Generate the auditable 26-field body YAML without creating SMPL-X."""
        output_root = job_root / "outputs" / "body_measurements"
        semantic_profile = config.get("semantic_profile") or (
            config["gender"] if config["gender"] != "neutral" else "female"
        )
        request = {
            "gender": (
                config["gender"]
                if config["gender"] in {"female", "male"}
                else semantic_profile
            ),
            "height_cm": config["height_cm"],
            "chest_cm": config["chest_cm"],
            "waist_cm": config["waist_cm"],
            "hips_cm": config.get("hips_cm"),
            "boundary_policy": config.get("boundary_policy", "extrapolate"),
        }
        sys.path.insert(0, str(self.root / "dynamic3d" / "body_customization"))
        from gbt1335_adapter import write_garmentcode_artifacts

        garmentcode_body, _, _ = write_garmentcode_artifacts(request, output_root)
        request["resolved_hips_cm"] = garmentcode_body["hips"]
        (output_root / "body_request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.store.update(
            job_id,
            progress=7,
            message="人体尺寸已补全为 GarmentCode 26 字段，正在加载男西装模型",
        )
        return output_root / "garmentcode_body.yaml"

    def run_official_lower_branch(
        self,
        job_id: str,
        config: dict[str, Any],
        job_root: Path,
        body_path: Path,
        paths: dict[str, Path],
    ) -> dict[str, Any]:
        """Run the untouched official model and retain only its lower-body branch."""
        dataset_name = f"job_{job_id}_official_lower"
        transient_inputs = paths["chatgarment"] / "app_inputs" / dataset_name
        transient_run = (
            paths["chatgarment"]
            / "runs"
            / "try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final"
            / f"{dataset_name}_img_recon"
        )
        if transient_inputs.exists() or transient_run.exists():
            raise FileExistsError("official lower-body transient paths already exist")
        lower_output = job_root / "outputs" / "official_lower"
        try:
            prepare_chatgarment_inputs(job_root / "inputs", transient_inputs)
            self.set_stage(
                job_id,
                "chatgarment",
                32,
                "官方原权重正在识别同批图片中的下装",
            )
            inference_env = os.environ.copy()
            inference_env["CUDA_VISIBLE_DEVICES"] = "0"
            inference_env["CHATGARMENT_BODY_MEASUREMENT_PATH"] = str(body_path)
            self.run_command(
                job_id,
                [
                    "bash",
                    str(
                        paths["chatgarment"]
                        / "scripts"
                        / "v1_5"
                        / "evaluate_garment_v2_imggen_2step.sh"
                    ),
                    str(transient_inputs),
                ],
                cwd=paths["chatgarment"],
                env=inference_env,
            )
            vis_root = transient_run / "vis_new"
            spec_list = vis_root / "all_json_spec_files.json"
            if not spec_list.is_file():
                raise FileNotFoundError(spec_list)
            lower_specs = load_lower_specifications(
                paths["chatgarment"], vis_root, spec_list
            )

            static_summary = job_root / "outputs" / "official_lower_static_summary.json"
            if lower_specs:
                self.set_stage(
                    job_id,
                    "static_3d",
                    42,
                    f"官方原权重识别到 {len(lower_specs)} 件下装，正在生成独立板片与静态预览",
                )
                sim_env = os.environ.copy()
                sim_env.update(
                    {
                        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
                        "PYTHONPATH": str(paths["garmentcode"]),
                        "PYOPENGL_PLATFORM": "egl",
                        "CUDA_VISIBLE_DEVICES": "0",
                    }
                )
                sim_command = [
                    str(paths["venv_python"]),
                    str(paths["garment_sim"]),
                    "--garmentcode-root", str(paths["garmentcode"]),
                ]
                for lower_spec in lower_specs:
                    sim_command.extend(["--spec", str(lower_spec)])
                sim_command.extend(
                    [
                        "--config", str(paths["sim_config"]),
                        "--system", str(paths["sim_system"]),
                        "--body", "mean_all",
                        "--summary", str(static_summary),
                    ]
                )
                self.run_command(
                    job_id,
                    sim_command,
                    cwd=paths["garmentcode"],
                    env=sim_env,
                )

            lower_output.mkdir(parents=True, exist_ok=False)
            case_sources = sorted(path for path in vis_root.iterdir() if path.is_dir())
            for case_source in case_sources:
                case_target = lower_output / case_source.name
                case_target.mkdir(parents=True, exist_ok=True)
                output_text = case_source / "output.txt"
                if output_text.is_file():
                    shutil.copy2(output_text, case_target / "official_output.txt")

            cases: list[dict[str, Any]] = []
            garment_counts: dict[str, int] = {}
            for lower_spec in lower_specs:
                case_source = lower_spec.parent.parent
                lower_source = lower_spec.parent
                case_target = lower_output / case_source.name
                lower_target = case_target / lower_source.name
                shutil.copytree(lower_source, lower_target, dirs_exist_ok=True)
                lower_id = lower_spec.stem.removesuffix("_specification")
                design_path = lower_source / "design.yaml"
                garment_type = (
                    read_lower_garment_type(design_path)
                    if design_path.is_file()
                    else None
                )
                garment_label = garment_type or "UnknownLower"
                garment_counts[garment_label] = garment_counts.get(garment_label, 0) + 1
                render_root = lower_target / lower_id
                cases.append(
                    {
                        "case_id": case_source.name,
                        "input_key": case_source.name.removeprefix("valid_garment_"),
                        "lower_id": lower_id,
                        "garment_type": garment_type,
                        "specification": str(
                            (lower_target / lower_spec.name).relative_to(job_root)
                        ),
                        "pattern_png": str(
                            (lower_target / f"{lower_id}_pattern.png").relative_to(job_root)
                        ),
                        "render_front": str(
                            (render_root / f"{lower_id}_render_front.png").relative_to(job_root)
                        ),
                        "render_back": str(
                            (render_root / f"{lower_id}_render_back.png").relative_to(job_root)
                        ),
                        "sim_mesh": str(
                            (render_root / f"{lower_id}_sim.obj").relative_to(job_root)
                        ),
                    }
                )

            static_completed_count = 0
            lower_simulation_quality = None
            if static_summary.is_file():
                static_entries = json.loads(static_summary.read_text(encoding="utf-8"))
                completed_entries = [
                    item for item in static_entries if item.get("status") == "completed"
                ]
                static_completed_count = len(completed_entries)
                body_collisions: list[int] = []
                self_collisions: list[int] = []
                frames: list[int] = []
                warnings: set[str] = set()
                for item in completed_entries:
                    stats = item.get("simulation_stats") or {}
                    body_collisions.extend(
                        int(value)
                        for value in (stats.get("body_collisions") or {}).values()
                    )
                    self_collisions.extend(
                        int(value)
                        for value in (stats.get("self_collisions") or {}).values()
                    )
                    frames.extend(
                        int(value) for value in (stats.get("fin_frame") or {}).values()
                    )
                    for name, affected in (stats.get("fails") or {}).items():
                        if affected:
                            warnings.add(str(name))
                lower_simulation_quality = {
                    "body_collisions": sum(body_collisions),
                    "self_collisions": sum(self_collisions),
                    "min_frames": min(frames) if frames else None,
                    "max_frames": max(frames) if frames else None,
                    "warnings": sorted(warnings),
                }
            payload = {
                "schema": "chatgarment_official_lower_v1",
                "method": "official_base_weights_lower_only_suit_lora_disabled",
                "expected_image_count": len(config["input_files"]),
                "detected_lower_count": len(cases),
                "static_completed_count": static_completed_count,
                "simulation_quality": lower_simulation_quality,
                "garment_counts": garment_counts,
                "collision_body": "mean_all",
                "cases": cases,
                "static_summary": (
                    str(static_summary.relative_to(job_root))
                    if static_summary.is_file()
                    else None
                ),
                "dynamic_included": False,
                "dynamic_case_count": 0,
                "notes": [
                    "The suit upper branch is generated separately by the suit LoRA and K62.",
                    "Only the untouched official model's lower-body output is retained here.",
                    "Detected lower garments are available for merging with the K62 upper in ContourCraft.",
                ],
            }
            (lower_output / "manifest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return payload
        finally:
            if transient_inputs.is_dir():
                shutil.rmtree(transient_inputs)
            if transient_run.is_dir():
                shutil.rmtree(transient_run)

    def run_suit_job(
        self,
        job_id: str,
        config: dict[str, Any],
        job_root: Path,
        paths: dict[str, Path],
    ) -> None:
        self.require_accelerator("chatgarment_cuda", "ChatGarment suit inference")
        self.require_accelerator("ccraft_cuda", "K62 suit simulation")
        self.require_accelerator("static_cuda", "official lower garment simulation")
        self.require_nvidia_egl("K62 suit rendering")
        for required in (
            paths["suit_inference"],
            paths["suit_lora"],
            paths["suit_3d_adapter"],
            paths["suit_3d_golden_spec"],
            paths["suit_3d_sim_config"],
            paths["suit_3d_system"],
            paths["garment_sim"],
            paths["suit_dynamic_prepare"],
            paths["suit_static_outfit_render"],
            paths["sim_config"],
            paths["sim_system"],
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        if not paths["suit_3d_garmentcode"].is_dir():
            raise FileNotFoundError(paths["suit_3d_garmentcode"])
        if not paths["garmentcode"].is_dir():
            raise FileNotFoundError(paths["garmentcode"])

        self.set_stage(job_id, "preparing", 3, "正在匹配国标基础人体并补全制版尺寸")
        self.check_cancel(job_id)
        body_root = self.write_body_inputs(job_id, config, job_root)
        body_path = job_root / "outputs" / "body_measurements" / "garmentcode_body.yaml"
        output_root = job_root / "outputs" / "mens_suit"
        self.set_stage(
            job_id,
            "chatgarment",
            10,
            f"男西装 LoRA 正在依次处理 {len(config['input_files'])} 张图片并生成二维版片",
        )
        inference_env = os.environ.copy()
        inference_env["CUDA_VISIBLE_DEVICES"] = "0"
        self.run_command(
            job_id,
            [
                str(paths["venv_python"]),
                str(paths["suit_inference"]),
                "--project-root", str(self.root),
                "--image-dir", str(job_root / "inputs"),
                "--output-dir", str(output_root),
                "--suit-lora", str(paths["suit_lora"]),
                "--body", str(body_path),
                "--limit", str(len(config["input_files"])),
                "--max-new-tokens", "256",
            ],
            cwd=self.root / "suit_finetune",
            env=inference_env,
        )

        source_specs = sorted(output_root.rglob("*_specification.json"))
        if len(source_specs) != len(config["input_files"]):
            raise RuntimeError(
                "Suit inference generated an unexpected number of specifications: "
                f"expected {len(config['input_files'])}, got {len(source_specs)}"
            )

        lower_payload = self.run_official_lower_branch(
            job_id, config, job_root, body_path, paths
        )
        lower_mesh_by_key = {
            str(case["input_key"]): job_root / str(case["sim_mesh"])
            for case in lower_payload.get("cases", [])
            if case.get("input_key") and case.get("sim_mesh")
        }

        self.set_stage(
            job_id,
            "static_3d",
            52,
            f"K62 正在适配并使用 NVIDIA Warp 处理 {len(source_specs)} 套西装三维规格",
        )
        adapted_specs: list[Path] = []
        dynamic_cases: list[tuple[str, Path, Path | None, str | None]] = []
        closure_counts: list[int] = []
        for index, source_spec in enumerate(source_specs, start=1):
            self.check_cancel(job_id)
            case_root = source_spec.parents[2]
            static_root = case_root / "static_3d"
            static_root.mkdir(parents=True, exist_ok=True)
            garment_name = source_spec.stem.removesuffix("_specification")
            adapted_spec = static_root / f"{garment_name}_k62_3d_specification.json"
            audit_path = static_root / f"{garment_name}_k62_3d_adapter_audit.json"
            closure_input = resolve_suit_button_count(case_root / "result.json")
            button_count = int(closure_input["resolved_button_count"])
            closure_counts.append(button_count)
            (static_root / f"{garment_name}_model_closure_input.json").write_text(
                json.dumps(closure_input, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shutil.copy2(body_path, static_root / "body_measurements.yaml")
            self.run_command(
                job_id,
                [
                    str(paths["ccraft_python"]),
                    str(paths["suit_3d_adapter"]),
                    "--input-spec", str(source_spec),
                    "--golden-spec", str(paths["suit_3d_golden_spec"]),
                    "--output-spec", str(adapted_spec),
                    "--audit", str(audit_path),
                    "--button-count", str(button_count),
                ],
                cwd=self.root,
            )
            adapted_specs.append(adapted_spec)
            input_key = re.sub(r"^\d+_", "", case_root.name, count=1)
            lower_mesh = lower_mesh_by_key.get(input_key)
            dynamic_cases.append(
                (
                    case_root.name,
                    static_root / f"{garment_name}_k62_3d" / f"{garment_name}_k62_3d_sim.obj",
                    lower_mesh,
                    input_key if lower_mesh is not None else None,
                )
            )
            progress = 52 + round(8 * index / max(1, len(source_specs)))
            self.store.update(
                job_id,
                progress=progress,
                message=(
                    f"已完成 {index}/{len(source_specs)} 套 K62 三维装配规格；"
                    f"模型预测 {button_count} 扣"
                ),
            )

        closure_text = "、".join(f"{count} 扣" for count in sorted(set(closure_counts)))
        self.store.update(
            job_id,
            message=f"K62 已按模型扣合（{closure_text}），正在进行 Warp 垂坠与渲染",
        )

        sim_env = os.environ.copy()
        sim_env.update(
            {
                "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
                "PYTHONPATH": str(paths["suit_3d_garmentcode"]),
                "PYOPENGL_PLATFORM": "egl",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
        sim_command = [
            str(paths["ccraft_python"]),
            str(paths["garment_sim"]),
            "--garmentcode-root", str(paths["suit_3d_garmentcode"]),
        ]
        for adapted_spec in adapted_specs:
            sim_command.extend(["--spec", str(adapted_spec)])
        sim_command.extend(
            [
                "--config", str(paths["suit_3d_sim_config"]),
                "--system", str(paths["suit_3d_system"]),
                "--body", "mean_all",
                "--summary", str(output_root / "static_3d_simulation_summary.json"),
            ]
        )
        self.run_command(
            job_id,
            sim_command,
            cwd=paths["suit_3d_garmentcode"],
            env=sim_env,
        )

        combined_static_records: list[tuple[Path, Path, Path]] = []
        for case_id, upper_mesh, lower_mesh, _ in dynamic_cases:
            if lower_mesh is None:
                continue
            combined_static_root = (
                output_root / case_id / "static_3d" / "combined_outfit"
            )
            combined_static_root.mkdir(parents=True, exist_ok=True)
            combined_static_records.append(
                (
                    upper_mesh,
                    lower_mesh,
                    combined_static_root,
                )
            )

        if combined_static_records:
            self.set_stage(
                job_id,
                "static_3d",
                61,
                f"正在生成 {len(combined_static_records)} 套男西装上下装同穿静态预览",
            )

        for upper_mesh, lower_mesh, combined_static_root in combined_static_records:
            self.run_command(
                job_id,
                [
                    str(paths["blender"]),
                    "--background",
                    "--python", str(paths["suit_static_outfit_render"]),
                    "--",
                    "--upper", str(upper_mesh),
                    "--lower", str(lower_mesh),
                    "--body", str(
                        paths["suit_3d_garmentcode"]
                        / "assets"
                        / "bodies"
                        / "mean_all.obj"
                    ),
                    "--output-dir", str(combined_static_root),
                ],
                cwd=self.root,
            )

        if config["action_id"] != "none":
            for _, mesh_path, lower_mesh, _ in dynamic_cases:
                if not mesh_path.is_file():
                    raise FileNotFoundError(mesh_path)
                if lower_mesh is not None and not lower_mesh.is_file():
                    raise FileNotFoundError(lower_mesh)
            dynamic_inputs = job_root / "work" / "suit_dynamic_inputs"
            dynamic_manifest = dynamic_inputs / "dynamic_manifest.json"
            self.set_stage(
                job_id,
                "dynamic_preparation",
                62,
                "正在合并男西装上衣与下装动态网格",
            )
            prepare_command = [
                str(paths["ccraft_python"]),
                str(paths["suit_dynamic_prepare"]),
            ]
            for case_id, mesh_path, lower_mesh, _ in dynamic_cases:
                prepare_command.extend(["--case", f"{case_id}={mesh_path}"])
                if lower_mesh is not None:
                    prepare_command.extend(["--lower", f"{case_id}={lower_mesh}"])
            prepare_command.extend(
                [
                    "--output-root", str(dynamic_inputs),
                    "--manifest", str(dynamic_manifest),
                ]
            )
            self.run_command(job_id, prepare_command, cwd=self.root)

            self.set_stage(
                job_id,
                "dynamic_3d",
                68,
                "ContourCraft 正在使用 CUDA 驱动男西装上衣与下装运动",
            )
            with np.load(body_root / "motion.npz", allow_pickle=True) as motion_data:
                motion_frames = int(np.asarray(motion_data["poses"]).shape[0])
            self.run_command(
                job_id,
                [
                    str(paths["ccraft_python"]), str(paths["dynamic_batch"]),
                    "--project-root", str(paths["contourcraft"]),
                    "--hood-data", str(paths["hood_data"]),
                    "--manifest", str(dynamic_manifest),
                    "--checkpoint", str(paths["checkpoint"]),
                    "--smplx-model", str(paths["smplx_models"] / "SMPLX_MALE.pkl"),
                    "--motion", str(body_root / "motion.npz"),
                    "--rest-body-params", str(body_root / "registered_params.pkl"),
                    "--output-root", str(job_root / "outputs" / "dynamic"),
                    "--blender", str(paths["blender"]),
                    "--fps", "30",
                ],
                cwd=self.root,
                frame_progress={
                    "frames_per_case": max(1, motion_frames),
                    "case_count": len(dynamic_cases),
                    "progress_start": 68,
                    "simulation_end": 82,
                    "progress_end": 92,
                },
            )
            merged_input_keys = {
                input_key
                for _, _, lower_mesh, input_key in dynamic_cases
                if lower_mesh is not None and input_key is not None
            }
            lower_payload["dynamic_included"] = bool(merged_input_keys)
            lower_payload["dynamic_case_count"] = len(merged_input_keys)
            for case in lower_payload.get("cases", []):
                case["dynamic_included"] = case.get("input_key") in merged_input_keys
            lower_payload["notes"] = [
                "The suit upper branch is generated by the suit LoRA and K62.",
                "Only the untouched official model's lower-body output is retained.",
                "Detected lower garments are merged with the K62 upper for ContourCraft dynamic simulation.",
            ]
            (job_root / "outputs" / "official_lower" / "manifest.json").write_text(
                json.dumps(lower_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        self.set_stage(job_id, "collecting", 93, "正在导出 DXF 并整理男西装二维、静态与动态三维产物")
        self.export_job_dxfs(job_root)
        self.write_result_manifest(job_id, config)
        self.make_bundle(job_id)
        self.store.update(
            job_id,
            state="completed",
            step="completed",
            progress=100,
            message=(
                "男西装上衣与下装、按模型扣合及动态视频已生成"
                if config["action_id"] != "none"
                else "男西装上衣与下装、按模型扣合及静态渲染产物已生成"
            ),
            error=None,
            size_bytes=tree_size(job_root),
        )

    def run_job(self, job_id: str) -> None:
        row = self.store.row(job_id)
        if row is None:
            return
        config = json.loads(row["config_json"])
        job_root = self.store.job_root(job_id)
        paths = self.paths()
        try:
            if config.get("garment_mode") == "mens_suit":
                self.run_suit_job(job_id, config, job_root, paths)
                return
            self.require_accelerator("ccraft_cuda", "SMPL-X body generation")
            self.require_accelerator("chatgarment_cuda", "ChatGarment inference")
            self.require_accelerator("static_cuda", "GarmentCode simulation")
            self.require_nvidia_egl("Body, garment, and video rendering")
            self.set_stage(job_id, "preparing", 3, "正在补全人体尺寸与体型参数")
            self.check_cancel(job_id)
            body_root = self.write_body_inputs(job_id, config, job_root)

            self.set_stage(job_id, "chatgarment", 10, "ChatGarment 正在使用 CUDA 生成二维板片")
            input_dir = job_root / "inputs"
            dataset_name = f"job_{job_id}"
            transient_inputs = paths["chatgarment"] / "app_inputs" / dataset_name
            transient_run = (
                paths["chatgarment"]
                / "runs"
                / "try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final"
                / f"{dataset_name}_img_recon"
            )
            if transient_inputs.exists() or transient_run.exists():
                raise FileExistsError("transient job paths already exist")
            try:
                prepare_chatgarment_inputs(input_dir, transient_inputs)
                inference_env = os.environ.copy()
                inference_env["CUDA_VISIBLE_DEVICES"] = "0"
                inference_env["CHATGARMENT_BODY_MEASUREMENT_PATH"] = str(
                    job_root / "outputs" / "body_measurements" / "garmentcode_body.yaml"
                )
                self.run_command(
                    job_id,
                    [
                        "bash",
                        str(paths["chatgarment"] / "scripts" / "v1_5" / "evaluate_garment_v2_imggen_2step.sh"),
                        str(transient_inputs),
                    ],
                    cwd=paths["chatgarment"],
                    env=inference_env,
                )
                vis_root = transient_run / "vis_new"
                spec_list = vis_root / "all_json_spec_files.json"
                if not spec_list.is_file():
                    raise FileNotFoundError(spec_list)

                self.set_stage(
                    job_id,
                    "static_3d",
                    38,
                    "GarmentCode 正在使用 NVIDIA Warp 缝合、垂坠并渲染",
                )
                sim_env = os.environ.copy()
                sim_env.update(
                    {
                        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
                        "PYTHONPATH": str(paths["garmentcode"]),
                        "PYOPENGL_PLATFORM": "egl",
                        "CUDA_VISIBLE_DEVICES": "0",
                    }
                )
                self.run_command(
                    job_id,
                    [
                        str(paths["venv_python"]), str(paths["garment_sim"]),
                        "--garmentcode-root", str(paths["garmentcode"]),
                        "--spec-list", str(spec_list),
                        "--config", str(paths["sim_config"]),
                        "--system", str(paths["sim_system"]),
                        "--summary", str(job_root / "outputs" / "static_sim_summary.json"),
                        "--skip-completed",
                    ],
                    cwd=paths["garmentcode"],
                    env=sim_env,
                )

                if config["action_id"] != "none":
                    dynamic_inputs = job_root / "work" / "dynamic_inputs"
                    dynamic_manifest = dynamic_inputs / "dynamic_manifest.json"
                    self.set_stage(
                        job_id,
                        "dynamic_preparation",
                        55,
                        "正在使用 CPU 整理缝合网格和动作输入",
                    )
                    self.run_command(
                        job_id,
                        [
                            str(paths["ccraft_python"]), str(paths["prepare_dynamic"]),
                            "--vis-root", str(vis_root),
                            "--output-root", str(dynamic_inputs),
                            "--manifest", str(dynamic_manifest),
                            "--expected-cases", str(len(config["input_files"])),
                        ],
                        cwd=self.root,
                    )
                    self.set_stage(
                        job_id,
                        "dynamic_3d",
                        65,
                        "ContourCraft 正在使用 CUDA 进行动态布料仿真",
                    )
                    gender = config["gender"].upper()
                    with np.load(body_root / "motion.npz", allow_pickle=True) as motion_data:
                        motion_frames = int(np.asarray(motion_data["poses"]).shape[0])
                    self.run_command(
                        job_id,
                        [
                            str(paths["ccraft_python"]), str(paths["dynamic_batch"]),
                            "--project-root", str(paths["contourcraft"]),
                            "--hood-data", str(paths["hood_data"]),
                            "--manifest", str(dynamic_manifest),
                            "--checkpoint", str(paths["checkpoint"]),
                            "--smplx-model", str(paths["smplx_models"] / f"SMPLX_{gender}.pkl"),
                            "--motion", str(body_root / "motion.npz"),
                            "--rest-body-params", str(body_root / "registered_params.pkl"),
                            "--output-root", str(job_root / "outputs" / "dynamic"),
                            "--blender", str(paths["blender"]),
                            "--fps", "30",
                        ],
                        cwd=self.root,
                        frame_progress={
                            "frames_per_case": max(1, motion_frames),
                            "case_count": len(config["input_files"]),
                            "progress_start": 65,
                            "simulation_end": 80,
                            "progress_end": 92,
                        },
                    )

                self.set_stage(job_id, "collecting", 93, "正在整理产物并生成一键下载包")
                chatgarment_output = job_root / "outputs" / "chatgarment"
                shutil.copytree(vis_root, chatgarment_output)
            finally:
                if transient_inputs.is_dir():
                    shutil.rmtree(transient_inputs)
                if transient_run.is_dir():
                    shutil.rmtree(transient_run)

            self.store.update(job_id, message="正在从板片规格导出 1:1 DXF 样片")
            self.export_job_dxfs(job_root)
            self.write_result_manifest(job_id, config)
            self.make_bundle(job_id)
            size = tree_size(job_root)
            self.store.update(
                job_id,
                state="completed",
                step="completed",
                progress=100,
                message="All requested outputs are ready",
                error=None,
                size_bytes=size,
            )
        except InterruptedError:
            self.store.update(
                job_id,
                state="cancelled",
                step="cancelled",
                message="Cancelled at a safe step boundary",
                error=None,
                size_bytes=tree_size(job_root),
            )
            self.apply_deferred_delete(job_id)
        except Exception as exc:
            self.store.update(
                job_id,
                state="failed",
                step="failed",
                message="Pipeline failed; partial outputs and log were preserved",
                error=f"{type(exc).__name__}: {exc}",
                size_bytes=tree_size(job_root),
            )

    def artifact_records(self, job_id: str) -> list[dict[str, Any]]:
        root = self.store.job_root(job_id)
        records = []
        if not root.is_dir():
            return records
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() not in VISIBLE_ARTIFACT_SUFFIXES:
                continue
            if relative.startswith("work/") or path.name == "result_bundle.zip":
                continue
            category = relative.split("/", 1)[0]
            records.append(
                {
                    "name": path.name,
                    "path": relative,
                    "category": category,
                    "bytes": path.stat().st_size,
                    "url": f"/job-output/{job_id}/{relative}",
                }
            )
        return records

    def export_job_dxfs(self, job_root: Path) -> dict[str, Any]:
        """Export every final pattern specification under a job's outputs."""
        output_root = job_root / "outputs"
        specifications = sorted(output_root.rglob("*_specification.json"))
        if not specifications:
            raise FileNotFoundError("No GarmentCode specifications were produced for DXF export")
        exports: list[dict[str, Any]] = []
        for specification in specifications:
            pattern_name = specification.stem.removesuffix("_specification")
            dxf_path = specification.with_name(f"{pattern_name}_pattern.dxf")
            preview_path = specification.with_name(f"{pattern_name}_pattern_dxf_preview.svg")
            report = export_specification(specification, dxf_path, preview=preview_path)
            report["source_specification"] = specification.relative_to(job_root).as_posix()
            report["output_dxf"] = dxf_path.relative_to(job_root).as_posix()
            report["output_preview"] = preview_path.relative_to(job_root).as_posix()
            exports.append(report)
        manifest = {
            "format": "AutoCAD DXF 2000 (AC1015)",
            "units": "millimetres",
            "export_count": len(exports),
            "exports": exports,
        }
        (output_root / "dxf_export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def write_result_manifest(self, job_id: str, config: dict[str, Any]) -> None:
        root = self.store.job_root(job_id)
        is_suit = config.get("garment_mode") == "mens_suit"
        payload = {
            "job_id": job_id,
            "created_at": utc_now(),
            "config": config,
            "artifacts": self.artifact_records(job_id),
            "notes": [
                "GarmentCode 26-field YAML is an auditable coarse standards adapter output.",
                (
                    "The mens-suit branch transfers generated panel dimensions onto the K62 topology, stitch graph, and 3D placement, then closes K62 virtual-button edges according to the model-predicted button_count before Warp simulation."
                    if is_suit
                    else "Dynamic video uses the selected preset or custom SMPL-X body."
                ),
                (
                    "Mens-suit static outputs include separate upper/lower previews and a combined outfit preview; both static branches use the same mean_all collision body. Optional motion uses a standard male SMPL-X body and the selected action. User measurements drive drafting and panel adaptation, not these 3D bodies."
                    if is_suit
                    else "Static 3D uses the body mode selected for this task."
                ),
                (
                    "For mens-suit tasks, the base checkpoint recognizes the lower garment and produces lower-body drafting and static previews; detected lower garments are merged with the K62 upper for ContourCraft dynamic simulation."
                    if is_suit
                    else "General tasks retain the official upper/lower/whole-body output structure."
                ),
                "Runtime motion assets are intentionally excluded from result downloads.",
                "DXF files are exported at 1:1 metric scale from GarmentCode panel geometry; output units are millimetres.",
            ],
        }
        (root / "result_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def make_bundle(self, job_id: str) -> Path:
        root = self.store.job_root(job_id)
        return build_result_bundle(root)

    def apply_deferred_delete(self, job_id: str) -> None:
        row = self.store.row(job_id)
        if row and row["delete_after_cancel"]:
            delete_job(self.store, job_id, row["delete_after_cancel"])


def row_payload(store: JobStore, pipeline: Pipeline, row: sqlite3.Row) -> dict[str, Any]:
    job_id = row["id"]
    config = json.loads(row["config_json"])
    root = store.job_root(job_id)
    artifacts = pipeline.artifact_records(job_id) if row["state"] != "trashed" else []
    body_summary = None
    audit_path = root / "outputs" / "body_measurements" / "audit.json"
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            body_type = audit.get("body_type", {})
            target = audit.get("target", {})
            body_summary = {
                "inferred_type": body_type.get("inferred"),
                "mapped_type": body_type.get("mapped"),
                "base_source_id": audit.get("base_source_id"),
                "hips_cm": target.get("hips_cm"),
                "hips_source": target.get("hips_source"),
                "body_field_count": len(audit.get("body", {})),
                "warnings": audit.get("warnings", []),
            }
        except (OSError, ValueError, TypeError):
            body_summary = None
    simulation_quality = None
    suit_closure_summary = None
    official_lower_summary = None
    if config.get("garment_mode") == "mens_suit":
        try:
            closure_audits = sorted(
                (root / "outputs" / "mens_suit").rglob("*_k62_3d_adapter_audit.json")
            )
            counts: dict[str, int] = {}
            stitch_count = 0
            closure_case_count = 0
            for path in closure_audits:
                audit = json.loads(path.read_text(encoding="utf-8"))
                closure = audit.get("front_closure")
                if not isinstance(closure, dict):
                    continue
                closure_case_count += 1
                count = int(closure.get("button_count", 0))
                counts[str(count)] = counts.get(str(count), 0) + 1
                stitch_count += len(closure.get("stitches_added") or [])
            if closure_case_count:
                suit_closure_summary = {
                    "mode": "model_button_count",
                    "case_count": closure_case_count,
                    "counts": counts,
                    "closure_stitch_count": stitch_count,
                }
        except (OSError, ValueError, TypeError, AttributeError):
            suit_closure_summary = None
        lower_manifest = root / "outputs" / "official_lower" / "manifest.json"
        if lower_manifest.is_file():
            try:
                lower_payload = json.loads(lower_manifest.read_text(encoding="utf-8"))
                lower_quality = lower_payload.get("simulation_quality")
                lower_static_summary = (
                    root / "outputs" / "official_lower_static_summary.json"
                )
                if lower_quality is None and lower_static_summary.is_file():
                    lower_quality = summarize_simulation_entries(
                        json.loads(lower_static_summary.read_text(encoding="utf-8"))
                    )
                official_lower_summary = {
                    "expected_image_count": int(
                        lower_payload.get("expected_image_count", 0)
                    ),
                    "detected_lower_count": int(
                        lower_payload.get("detected_lower_count", 0)
                    ),
                    "static_completed_count": int(
                        lower_payload.get("static_completed_count", 0)
                    ),
                    "garment_counts": dict(lower_payload.get("garment_counts") or {}),
                    "dynamic_included": bool(lower_payload.get("dynamic_included")),
                    "dynamic_case_count": int(
                        lower_payload.get("dynamic_case_count", 0)
                    ),
                    "simulation_quality": lower_quality,
                }
            except (OSError, ValueError, TypeError, AttributeError):
                official_lower_summary = None
    summary_candidates = (
        root / "outputs" / "mens_suit" / "static_3d_simulation_summary.json",
        root / "outputs" / "static_sim_summary.json",
    )
    summary_path = next((path for path in summary_candidates if path.is_file()), None)
    if summary_path:
        try:
            entries = json.loads(summary_path.read_text(encoding="utf-8"))
            completed = [item for item in entries if item.get("status") == "completed"]
            body_collisions: list[int] = []
            self_collisions: list[int] = []
            frames: list[int] = []
            warnings: set[str] = set()
            for item in completed:
                stats = item.get("simulation_stats") or {}
                body_collisions.extend(
                    int(value) for value in (stats.get("body_collisions") or {}).values()
                )
                self_collisions.extend(
                    int(value) for value in (stats.get("self_collisions") or {}).values()
                )
                frames.extend(int(value) for value in (stats.get("fin_frame") or {}).values())
                for name, affected in (stats.get("fails") or {}).items():
                    if affected:
                        warnings.add(str(name))
            simulation_quality = {
                "case_count": len(entries),
                "completed_count": len(completed),
                "body_collisions": sum(body_collisions),
                "self_collisions": sum(self_collisions),
                "min_frames": min(frames) if frames else None,
                "max_frames": max(frames) if frames else None,
                "warnings": sorted(warnings),
            }
        except (OSError, ValueError, TypeError, AttributeError):
            simulation_quality = None
    return {
        "id": job_id,
        "name": row["name"],
        "state": row["state"],
        "step": row["step"],
        "progress": row["progress"],
        "message": row["message"],
        "runtime": pipeline.runtime_for_job(
            state=row["state"],
            step=row["step"],
            message=row["message"],
            config=config,
        ),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row["error"],
        "cancel_requested": bool(row["cancel_requested"]),
        "size_bytes": row["size_bytes"] or (tree_size(root) if root.exists() else 0),
        "config": config,
        "image_count": len(config.get("input_files", [])),
        "body_summary": body_summary,
        "suit_closure_summary": suit_closure_summary,
        "official_lower_summary": official_lower_summary,
        "simulation_quality": simulation_quality,
        "artifacts": artifacts,
        "bundle_url": (
            f"/job-output/{job_id}/result_bundle.zip"
            if (root / "result_bundle.zip").is_file()
            else None
        ),
    }


def delete_job(store: JobStore, job_id: str, mode: str) -> None:
    job_id = safe_job_id(job_id)
    row = store.row(job_id)
    if row is None:
        raise FileNotFoundError(job_id)
    if row["state"] == "running":
        store.update(
            job_id,
            cancel_requested=1,
            delete_after_cancel=mode,
            message="Cancellation requested; deletion will run at the next safe boundary",
        )
        return
    root = store.job_root(job_id).resolve()
    root.relative_to(store.runs_root.resolve())
    if mode == "cache":
        work = root / "work"
        if work.is_dir():
            shutil.rmtree(work)
        cache_names = {
            "contourcraft_sequence.npz",
            "body_sequence.pkl",
            "garment_template.pkl",
        }
        for path in root.rglob("*"):
            if path.is_file() and path.name in cache_names:
                path.unlink()
        if (root / "result_bundle.zip").is_file():
            build_result_bundle(root)
        store.update(job_id, size_bytes=tree_size(root), message="Intermediate cache cleared")
        return
    if mode == "trash":
        destination = store.trash_root / job_id
        if destination.exists():
            raise FileExistsError(destination)
        if root.exists():
            shutil.move(str(root), str(destination))
        store.update(
            job_id,
            state="trashed",
            step="trashed",
            message="Moved to recycle bin",
            trash_path=str(destination),
            size_bytes=tree_size(destination),
        )
        return
    if mode == "permanent":
        candidates = [root]
        if row["trash_path"]:
            trash_path = Path(row["trash_path"]).resolve()
            trash_path.relative_to(store.trash_root.resolve())
            candidates.append(trash_path)
        for target in candidates:
            if target.is_dir():
                shutil.rmtree(target)
        with store.connect() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return
    raise ValueError("delete mode must be trash, permanent, or cache")


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ChatGarmentJobs/0.1"

    @property
    def store(self) -> JobStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def pipeline(self) -> Pipeline:
        return self.server.pipeline  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.log_date_time_string()} {fmt % args}\n")

    def send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self, maximum: int = 1024 * 1024) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > maximum:
            raise ValueError("invalid JSON request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON request must be an object")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in {"/api/jobs/health", "/api/jobs/schema"}:
                free = shutil.disk_usage(self.store.data_root).free
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "max_images": MAX_IMAGES,
                        "max_file_bytes": MAX_FILE_BYTES,
                        "minimum_free_bytes": MIN_FREE_BYTES,
                        "body_modes": ["preset", "custom"],
                        "genders": ["female", "male", "neutral"],
                        "runtime": self.pipeline.runtime_profile,
                    },
                )
                return
            if path == "/api/jobs/storage":
                usage = shutil.disk_usage(self.store.data_root)
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "runs": tree_size(self.store.runs_root),
                        "trash": tree_size(self.store.trash_root),
                        "accepting_jobs": usage.free >= MIN_FREE_BYTES,
                    },
                )
                return
            if path == "/api/jobs":
                self.send_json(
                    HTTPStatus.OK,
                    {"jobs": [row_payload(self.store, self.pipeline, row) for row in self.store.rows()]},
                )
                return
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
            if match:
                row = self.store.row(match.group(1))
                if row is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
                else:
                    self.send_json(HTTPStatus.OK, row_payload(self.store, self.pipeline, row))
                return
            if path.startswith("/job-output/"):
                self.serve_artifact(path)
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__, "message": str(exc)})

    def serve_artifact(self, path: str, *, head_only: bool = False) -> None:
        remainder = path[len("/job-output/"):]
        job_id, separator, relative = remainder.partition("/")
        if not separator:
            raise ValueError("artifact path is incomplete")
        root = self.store.job_root(safe_job_id(job_id))
        target = safe_child(root, relative)
        if not target.is_file():
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "file_not_found"})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        size = target.stat().st_size
        try:
            byte_range = parse_byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range is not None else (0, max(0, size - 1))
        content_length = end - start + 1 if size else 0
        self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if byte_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        disposition = "inline" if content_type.startswith(("image/", "video/", "text/")) else "attachment"
        self.send_header("Content-Disposition", f'{disposition}; filename="{target.name}"')
        self.end_headers()
        if head_only or not content_length:
            return
        with target.open("rb") as stream:
            stream.seek(start)
            remaining = content_length
            while remaining and (chunk := stream.read(min(1024 * 1024, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        try:
            if path.startswith("/job-output/"):
                self.serve_artifact(path, head_only=True)
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except Exception:
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/jobs":
                self.create_job()
                return
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/(delete|cancel|restore)", path)
            if match:
                self.mutate_job(match.group(1), match.group(2))
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
        except FileNotFoundError:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "message": str(exc)})

    def create_job(self) -> None:
        usage = shutil.disk_usage(self.store.data_root)
        if usage.free < MIN_FREE_BYTES:
            self.send_json(
                HTTPStatus.INSUFFICIENT_STORAGE,
                {"error": "low_disk_space", "free": usage.free, "minimum": MIN_FREE_BYTES},
            )
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Content-Type must be multipart/form-data")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )
        config_raw = form.getfirst("config")
        if not config_raw:
            raise ValueError("config field is required")
        config = json.loads(config_raw)
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        self.validate_config(config)
        image_fields = form["images"] if "images" in form else []
        if not isinstance(image_fields, list):
            image_fields = [image_fields]
        if not 1 <= len(image_fields) <= MAX_IMAGES:
            raise ValueError(f"upload between 1 and {MAX_IMAGES} images")

        clean_files = []
        payloads = []
        for index, field in enumerate(image_fields, start=1):
            original = Path(field.filename or f"image_{index}.png").name
            suffix = Path(original).suffix.lower()
            if suffix not in ALLOWED_IMAGE_SUFFIXES:
                raise ValueError(f"unsupported image type: {original}")
            data = field.file.read(MAX_FILE_BYTES + 1)
            if not data or len(data) > MAX_FILE_BYTES:
                raise ValueError(f"invalid image size: {original}")
            filename = f"{index:03d}_{re.sub(r'[^A-Za-z0-9._-]', '_', original)}"
            clean_files.append(filename)
            payloads.append((filename, data))

        config["input_files"] = clean_files
        job_id = self.store.create(str(config.get("name") or "服装制版任务"), config)
        job_root = self.store.job_root(job_id)
        inputs = job_root / "inputs"
        inputs.mkdir(parents=True, exist_ok=False)
        try:
            for filename, data in payloads:
                (inputs / filename).write_bytes(data)
            (job_root / "job_config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            if job_root.is_dir():
                shutil.rmtree(job_root)
            with self.store.connect() as db:
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            raise
        self.send_json(HTTPStatus.CREATED, {"job": row_payload(self.store, self.pipeline, self.store.row(job_id))})

    def validate_config(self, config: dict[str, Any]) -> None:
        config.setdefault("garment_mode", "general")
        if config["garment_mode"] not in {"general", "mens_suit"}:
            raise ValueError("garment_mode must be general or mens_suit")
        if config.get("body_mode") not in {"preset", "custom"}:
            raise ValueError("body_mode must be preset or custom")
        if config.get("gender") not in {"female", "male", "neutral"}:
            raise ValueError("gender must be female, male, or neutral")
        if config.get("action_id") not in {
            "none", "official_showcase", "standing_turn", "wave", "walk_in_place"
        }:
            raise ValueError("invalid action_id")
        if config["garment_mode"] == "mens_suit":
            config["gender"] = "male"
            config["body_mode"] = "preset"
            if config["action_id"] not in {"none", "official_showcase"}:
                raise ValueError("mens_suit supports none or official_showcase action")
        ranges = {
            "height_cm": (130, 220),
            "chest_cm": (50, 180),
            "waist_cm": (45, 180),
        }
        for key, limits in ranges.items():
            value = float(config[key])
            if not limits[0] <= value <= limits[1]:
                raise ValueError(f"{key} out of range")
            config[key] = value
        hips = config.get("hips_cm")
        config["hips_cm"] = None if hips in {None, ""} else float(hips)
        if config["hips_cm"] is not None and not 50 <= config["hips_cm"] <= 190:
            raise ValueError("hips_cm out of range")
        weight = config.get("weight_kg")
        config["weight_kg"] = None if weight in {None, ""} else float(weight)
        if config["weight_kg"] is not None and not 30 <= config["weight_kg"] <= 220:
            raise ValueError("weight_kg out of range")
        config.setdefault("semantic_profile", "female" if config["gender"] == "neutral" else config["gender"])
        config.setdefault("attributes", {})
        config.setdefault("boundary_policy", "extrapolate")

    def mutate_job(self, job_id: str, action: str) -> None:
        row = self.store.row(job_id)
        if row is None:
            raise FileNotFoundError(job_id)
        if action == "cancel":
            if row["state"] not in {"queued", "running"}:
                raise ValueError("only queued or running jobs can be cancelled")
            if row["state"] == "queued":
                self.store.update(job_id, state="cancelled", step="cancelled", message="Cancelled before execution")
            else:
                self.store.update(job_id, cancel_requested=1, message="Cancellation requested")
        elif action == "delete":
            payload = self.read_json()
            delete_job(self.store, job_id, str(payload.get("mode", "trash")))
        else:
            if row["state"] != "trashed" or not row["trash_path"]:
                raise ValueError("job is not in recycle bin")
            source = Path(row["trash_path"]).resolve()
            source.relative_to(self.store.trash_root.resolve())
            destination = self.store.job_root(job_id)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(source), str(destination))
            self.store.update(
                job_id,
                state="failed" if row["error"] else "completed",
                step="restored",
                message="Restored from recycle bin",
                trash_path=None,
            )
        updated = self.store.row(job_id)
        self.send_json(HTTPStatus.OK, {"job": row_payload(self.store, self.pipeline, updated)} if updated else {"deleted": True})


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = (args.data_root or project_root / "app_data").resolve()
    store = JobStore(data_root)
    pipeline = Pipeline(project_root, store)
    pipeline.start()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    server.store = store  # type: ignore[attr-defined]
    server.pipeline = pipeline  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "event": "listening",
                "host": args.host,
                "port": args.port,
                "project_root": str(project_root),
                "data_root": str(data_root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
