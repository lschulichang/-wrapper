"""Collision-safe experiment output allocation."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def artifact_family_exists(output: Path) -> bool:
    """Return true when writing an output prefix could overwrite any artifact."""

    output = output.resolve()
    if output.exists():
        return True
    if not output.parent.exists():
        return False
    prefix = output.stem + "."
    return any(item.is_file() and item.name.startswith(prefix) for item in output.parent.iterdir())


def resolve_experiment_output(
    requested_output: Path | None,
    project_root: Path,
    topic: str,
    argv: list[str] | None = None,
) -> tuple[Path, Path | None, str | None]:
    """Resolve an output path without ever overwriting an existing artifact family.

    Returns ``(output_json, run_dir, redirect_reason)``. A new standard run
    directory is allocated when no explicit output is supplied or when the
    requested prefix already exists.
    """

    reason = None
    if requested_output is not None:
        requested_output = requested_output.expanduser().resolve()
        if not artifact_family_exists(requested_output):
            requested_output.parent.mkdir(parents=True, exist_ok=True)
            return requested_output, None, None
        reason = f"requested output prefix already exists: {requested_output}"
    else:
        reason = "no --output was supplied"

    creator = project_root / "scripts" / "create_run_dir.sh"
    completed = subprocess.run(
        ["bash", str(creator), topic],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    run_dir = Path(completed.stdout.strip()).resolve()
    output = run_dir / "assets" / "ground_grid.json"
    command = [sys.executable, *sys.argv] if argv is None else argv
    (run_dir / "cmd.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return output, run_dir, reason
