#!/usr/bin/env python3
"""Small HTTP service for gender-aware SHAPY to SMPL-X generation."""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from gbt1335_adapter import (
    GBT1335Error,
    schema_payload as gbt1335_schema_payload,
    write_garmentcode_artifacts,
)
from motion_catalog import DEFAULT_ACTION_ID, action_by_id, public_actions
from shapy_adapter import (
    ATTRIBUTE_KEYS,
    ShapyCheckpointMissing,
    checkpoint_path,
    route_status,
)


ATTRIBUTE_LABELS_ZH = {
    "average": "体型平均",
    "big": "体型较大",
    "broad_shoulders": "宽肩",
    "delicate_build": "纤细骨架",
    "feminine": "女性化",
    "large_breasts": "胸部丰满",
    "long_legs": "腿长",
    "long_neck": "颈长",
    "long_torso": "躯干长",
    "masculine": "男性化",
    "muscular": "肌肉感",
    "pear_shaped": "梨形",
    "petite": "娇小",
    "rectangular": "直筒形",
    "short": "整体偏矮",
    "short_arms": "手臂偏短",
    "skinny_arms": "手臂纤细",
    "skinny_legs": "腿部纤细",
    "slim_waist": "腰部纤细",
    "soft_body": "体型柔和",
    "tall": "整体偏高",
}

GENERATION_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--shapy-data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def validate_request(payload: Any, project_root: Path | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    result = dict(payload)
    gender = str(result.get("gender", "")).lower()
    if gender not in {"female", "male", "neutral"}:
        raise ValueError("gender must be female, male, or neutral")
    result["gender"] = gender

    semantic_profile = str(
        result.get(
            "semantic_profile",
            gender if gender != "neutral" else "female",
        )
    ).lower()
    if semantic_profile not in {"female", "male"}:
        raise ValueError("semantic_profile must be female or male")
    if gender != "neutral" and semantic_profile != gender:
        raise ValueError("gender-specific models require a matching semantic profile")
    result["semantic_profile"] = semantic_profile

    ranges = {
        "height_cm": (130.0, 220.0),
        "weight_kg": (30.0, 220.0),
        "chest_cm": (50.0, 180.0),
        "waist_cm": (45.0, 180.0),
        "hips_cm": (50.0, 190.0),
    }
    for key, (minimum, maximum) in ranges.items():
        if key == "weight_kg" and result.get(key) in (None, ""):
            result[key] = None
            continue
        value = float(result[key])
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        result[key] = value

    submitted = result.get("attributes", {})
    if not isinstance(submitted, dict):
        raise ValueError("attributes must be an object")
    allowed = ATTRIBUTE_KEYS[semantic_profile]
    result["attributes"] = {}
    for key in allowed:
        value = float(submitted.get(key, 3.0))
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"attribute {key!r} must be between 1 and 5")
        result["attributes"][key] = value

    name = str(result.get("name", "")).strip()
    result["name"] = name[:80]
    action_id = str(result.get("action_id", DEFAULT_ACTION_ID))
    if project_root is not None:
        action_by_id(project_root, action_id)
    result["action_id"] = action_id
    return result


class BodyHandler(BaseHTTPRequestHandler):
    server_version = "ChatGarmentBody/0.1"

    @property
    def config(self) -> dict[str, Path]:
        return self.server.config  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            f"{self.log_date_time_string()} {self.client_address[0]} "
            f"{format % args}\n"
        )

    def send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/body/gbt1335/schema":
            self.send_json(HTTPStatus.OK, gbt1335_schema_payload())
            return
        if path in {"/api/body/health", "/health"}:
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "chatgarment-body",
                    "generator": "shapy-to-smplx",
                },
            )
            return
        if path in {"/api/body/schema", "/schema"}:
            routes = route_status(self.config["shapy_data_root"])
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ready" if any(r["available"] for r in routes) else "waiting_for_weights",
                    "research_only": True,
                    "model_genders": ["female", "male", "neutral"],
                    "actions": public_actions(self.config["project_root"]),
                    "semantic_profiles": {
                        profile: [
                            {
                                "key": key,
                                "label": ATTRIBUTE_LABELS_ZH[key],
                                "minimum": 1,
                                "maximum": 5,
                                "default": 3,
                            }
                            for key in keys
                        ]
                        for profile, keys in ATTRIBUTE_KEYS.items()
                    },
                    "measurements": {
                        "required": ["height_cm", "chest_cm", "waist_cm", "hips_cm"],
                        "optional": ["weight_kg"],
                        "units": {"height_cm": "cm", "weight_kg": "kg", "chest_cm": "cm", "waist_cm": "cm", "hips_cm": "cm"},
                    },
                    "checkpoint_routes": routes,
                },
            )
            return
        if path.startswith("/body-output/"):
            self.serve_output(path)
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def serve_output(self, request_path: str) -> None:
        relative = unquote(request_path[len("/body-output/") :])
        output_root = self.config["output_root"].resolve()
        target = (output_root / relative).resolve()
        try:
            target.relative_to(output_root)
        except ValueError:
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "invalid_path"})
            return
        if not target.is_file() or target.name.startswith("."):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "file_not_found"})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        size = target.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        disposition = "inline" if content_type.startswith("image/") else "attachment"
        self.send_header(
            "Content-Disposition",
            f'{disposition}; filename="{target.name}"',
        )
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with target.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/body/gbt1335/generate":
            self.generate_gbt1335()
            return
        if path not in {"/api/body/generate", "/generate"}:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("JSON request must be between 1 byte and 1 MiB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request = validate_request(payload, self.config["project_root"])

            variant = (
                "05b_ahwcwh2s"
                if request["weight_kg"] is not None
                else "04b_ahcwh2s"
            )
            checkpoint = checkpoint_path(
                self.config["shapy_data_root"],
                model_gender=request["gender"],
                semantic_profile=request["semantic_profile"],
                variant=variant,
            )
            if not checkpoint.is_file():
                raise ShapyCheckpointMissing(checkpoint)

            request_id = uuid.uuid4().hex
            output_root = self.config["output_root"]
            request_root = output_root / "_requests"
            request_root.mkdir(parents=True, exist_ok=True)
            request_path = request_root / f"{request_id}.json"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_dir = output_root / request_id
            generator = Path(__file__).with_name("body_generator.py")
            with GENERATION_LOCK:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(generator),
                        "--request",
                        str(request_path),
                        "--output-dir",
                        str(output_dir),
                        "--project-root",
                        str(self.config["project_root"]),
                        "--shapy-data-root",
                        str(self.config["shapy_data_root"]),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
            if process.returncode:
                raise RuntimeError(
                    process.stderr.strip() or process.stdout.strip() or "generator failed"
                )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["request_id"] = request_id
            manifest["downloads"] = {
                key: f"/body-output/{request_id}/{value}"
                for key, value in manifest["files"].items()
            }
            self.send_json(HTTPStatus.CREATED, manifest)
        except ShapyCheckpointMissing as exc:
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": "shapy_checkpoint_missing",
                    "message": "人体模型与接口已部署，但 SHAPY 官方权重尚未安装。",
                    "checkpoint": str(exc.checkpoint),
                    "next_step": "从 SHAPY 官方站点下载并上传 shapy_data.zip；不使用第三方权重。",
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(exc)},
            )
        except subprocess.TimeoutExpired:
            self.send_json(
                HTTPStatus.GATEWAY_TIMEOUT,
                {"error": "generation_timeout"},
            )
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "generation_failed", "message": str(exc)},
            )

    def generate_gbt1335(self) -> None:
        """Generate a 26-field GarmentCode body without invoking SHAPY/GPU."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise GBT1335Error(
                    "JSON request must be between 1 byte and 1 MiB"
                )
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request_id = f"gbt1335_{uuid.uuid4().hex}"
            output_dir = self.config["output_root"] / request_id
            body, audit, files = write_garmentcode_artifacts(
                payload, output_dir
            )
            downloads = {
                key: f"/body-output/{request_id}/{filename}"
                for key, filename in files.items()
            }
            audit.update(
                {
                    "request_id": request_id,
                    "files": files,
                    "downloads": downloads,
                }
            )
            (output_dir / files["audit_json"]).write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            response = {
                "status": "complete",
                **audit,
                "body": body,
            }
            self.send_json(HTTPStatus.CREATED, response)
        except (
            GBT1335Error,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(exc)},
            )
        except FileExistsError:
            self.send_json(
                HTTPStatus.CONFLICT,
                {"error": "request_id_collision"},
            )
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "generation_failed", "message": str(exc)},
            )


def main() -> None:
    args = parse_args()
    inferred_root = Path(__file__).resolve().parents[2]
    project_root = (args.project_root or inferred_root).resolve()
    config = {
        "project_root": project_root,
        "shapy_data_root": (
            args.shapy_data_root
            or project_root / "third_party" / "shapy" / "data"
        ).resolve(),
        "output_root": (
            args.output_root
            or project_root / "dynamic3d" / "outputs_custom_bodies"
        ).resolve(),
    }
    config["output_root"].mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), BodyHandler)
    server.config = config  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "event": "listening",
                "host": args.host,
                "port": args.port,
                **{key: str(value) for key, value in config.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
