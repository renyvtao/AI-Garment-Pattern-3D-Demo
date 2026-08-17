"""Validated access to the shared dynamic-action catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ACTION_ID = "official_showcase"


def catalog_path(project_root: Path) -> Path:
    return project_root / "dynamic3d" / "motions" / "catalog.json"


def load_catalog(project_root: Path) -> dict[str, Any]:
    path = catalog_path(project_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"motion catalog has no actions: {path}")
    ids = [str(action.get("id", "")) for action in actions]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError(f"motion catalog contains invalid action ids: {path}")
    return payload


def action_by_id(project_root: Path, action_id: str) -> dict[str, Any]:
    actions = load_catalog(project_root)["actions"]
    matches = [item for item in actions if item["id"] == action_id]
    if len(matches) != 1:
        allowed = ", ".join(item["id"] for item in actions)
        raise ValueError(f"unknown action_id {action_id!r}; allowed: {allowed}")
    return matches[0]


def resolve_action_asset(project_root: Path, action: dict[str, Any]) -> Path | None:
    value = action.get("asset")
    if value is None:
        return None
    root = project_root.resolve()
    target = (root / str(value)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("motion asset escapes project root") from exc
    return target


def public_actions(project_root: Path) -> list[dict[str, Any]]:
    result = []
    for action in load_catalog(project_root)["actions"]:
        asset = resolve_action_asset(project_root, action)
        result.append(
            {
                "id": action["id"],
                "label_zh": action["label_zh"],
                "kind": action["kind"],
                "available": asset is None or asset.is_file(),
            }
        )
    return result
