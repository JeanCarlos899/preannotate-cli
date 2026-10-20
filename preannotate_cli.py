"""Public command-line interface for segmentation pre-annotation and remote GPU orchestration.

Configuration and SSH connection details are read from env/profile and never hardcoded.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import uuid
from urllib.parse import urlparse

from preannotate_core import (
    ensure_new_path,
    export_proposals,
    inspect_dataset,
    merge_datasets,
    render_video,
    run_smooth,
    run_thresholds,
)

VERSION = "0.2.0"
DEFAULT_ENV = Path.home() / ".config/preannotate/.env"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError(f"Restrict permissions first: chmod 600 {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError("Invalid env line (expected KEY=value)")
            values[key.strip()] = value.strip().strip("\"'")
    return values


def validate_safe_token(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned or not re.fullmatch(r"[A-Za-z0-9._/@:-]+", cleaned):
        raise ValueError(f"Invalid or unsafe value for {field_name}: {value!r}")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Path traversal not allowed in {field_name}: {value!r}")
    return cleaned


def parse_ssh_target(
    raw_target: str, port_override: str = "", identity_override: str = ""
) -> dict:
    target = raw_target.strip()
    if not target:
        raise ValueError(
            "Missing PREANNOTATE_SSH_TARGET. Configure ~/.config/preannotate/.env or pass --ssh."
        )
    port = port_override.strip()
    identity = identity_override.strip()

    if target.startswith("ssh://"):
        parsed = urlparse(target)
        if not parsed.hostname:
            raise ValueError(f"Invalid SSH URL: {target}")
        host = (
            f"{parsed.username}@{parsed.hostname}"
            if parsed.username
            else parsed.hostname
        )
        if parsed.port and not port:
            port = str(parsed.port)
    else:
        tokens = shlex.split(target)
        host_candidates = []
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == "-p" and idx + 1 < len(tokens):
                port = port or tokens[idx + 1]
                idx += 2
            elif tok == "-i" and idx + 1 < len(tokens):
                identity = identity or tokens[idx + 1]
                idx += 2
            elif tok.startswith("-"):
                raise ValueError(f"Unsupported SSH option in target: {tok}")
            else:
                host_candidates.append(tok)
                idx += 1
        if len(host_candidates) != 1:
            raise ValueError(
                f"Expected a single SSH host or alias, got: {raw_target!r}"
            )
        host = host_candidates[0]

    validate_safe_token(host, "SSH host")
    if host.startswith("-"):
        raise ValueError("SSH host cannot begin with an option")
    if port and (not port.isdigit() or not 1 <= int(port) <= 65535):
        raise ValueError(f"Invalid SSH port: {port!r}")
    if identity:
        identity = str(Path(identity).expanduser().resolve())
    return {"host": host, "port": port, "identity": identity}


def resolve_remote_config(args: argparse.Namespace) -> dict:
    env_file = Path(args.env).expanduser()
    file_vars = read_env(env_file)

    def get_cfg(key: str, cli_val: str | None = None, default: str = "") -> str:
        if cli_val is not None and str(cli_val).strip() != "":
            return str(cli_val).strip()
        if key in os.environ and os.environ[key].strip() != "":
            return os.environ[key].strip()
        return file_vars.get(key, default).strip()

    ssh_info = parse_ssh_target(
        get_cfg("PREANNOTATE_SSH_TARGET", getattr(args, "ssh", None)),
        port_override=get_cfg("PREANNOTATE_SSH_PORT"),
        identity_override=get_cfg("PREANNOTATE_SSH_IDENTITY"),
    )
    remote_root = validate_safe_token(
        get_cfg(
            "PREANNOTATE_REMOTE_ROOT",
            getattr(args, "remote_root", None),
            "/home/getter",
        ),
        "PREANNOTATE_REMOTE_ROOT",
    ).rstrip("/")
    workspace_user = validate_safe_token(
        get_cfg(
            "PREANNOTATE_WORKSPACE_USER", getattr(args, "user", None), getpass.getuser()
        ),
        "PREANNOTATE_WORKSPACE_USER",
    ).strip("/")

    if not remote_root.startswith("/") or remote_root == "/":
        raise ValueError("Remote root must be an absolute non-root directory")
    project_name = validate_safe_token(
        get_cfg(
            "PREANNOTATE_WORKSPACE_DIR", getattr(args, "project", None), "preannotate"
        ),
        "PREANNOTATE_WORKSPACE_DIR",
    ).strip("/")
    for name, value in (("user", workspace_user), ("project", project_name)):
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
            raise ValueError(f"{name} must be a single directory name")

    user_base = f"{remote_root}/{workspace_user}"
    workspace_dir = f"{user_base}/{project_name}"

    remote_venv = get_cfg("PREANNOTATE_REMOTE_VENV")
    if remote_venv:
        remote_venv = validate_safe_token(remote_venv, "PREANNOTATE_REMOTE_VENV")
    else:
        remote_venv = f"{workspace_dir}/.venv"

    sam2_checkpoint = get_cfg("PREANNOTATE_SAM2_CHECKPOINT")
    if sam2_checkpoint:
        sam2_checkpoint = validate_safe_token(
            sam2_checkpoint, "PREANNOTATE_SAM2_CHECKPOINT"
        )
    else:
        sam2_checkpoint = (
            f"{workspace_dir}/vendor/sam2/checkpoints/sam2.1_hiera_large.pt"
        )

    sam2_config = get_cfg(
        "PREANNOTATE_SAM2_CONFIG", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )

    return {
        "ssh": ssh_info,
        "remote_root": remote_root,
        "workspace_user": workspace_user,
        "project_name": project_name,
        "user_base": user_base,
        "workspace_dir": workspace_dir,
        "remote_venv": remote_venv,
        "sam2_checkpoint": sam2_checkpoint,
        "sam2_config": sam2_config,
    }


def ssh_base_cmd(ssh_info: dict) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if ssh_info["port"]:
        cmd.extend(["-p", ssh_info["port"]])
    if ssh_info["identity"]:
        cmd.extend(["-i", ssh_info["identity"]])
    cmd.append(ssh_info["host"])
    return cmd


def rsync_rsh_arg(ssh_info: dict) -> str:
    parts = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if ssh_info["port"]:
        parts.extend(["-p", ssh_info["port"]])
    if ssh_info["identity"]:
        parts.extend(["-i", shlex.quote(ssh_info["identity"])])
    return " ".join(parts)


def run_remote_shell(ssh_info: dict, script: str) -> str:
    cmd = ssh_base_cmd(ssh_info) + [script]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"Remote SSH command failed (exit {proc.returncode}): {err}")
    return proc.stdout


def worker_result(output: str) -> dict:
    """Parse a multiline JSON result following optional model progress logs."""
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, end = json.JSONDecoder().raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not output[index + end :].strip():
            return value
    raise ValueError(
        "Worker did not return a complete JSON object; inspect retained artifacts"
    )


def rsync_upload(ssh_info: dict, local_path: Path, remote_path: str) -> None:
    local_resolved = local_path.expanduser().resolve()
    src = f"{local_resolved}/" if local_resolved.is_dir() else str(local_resolved)
    dst = f"{ssh_info['host']}:{remote_path}" + ("/" if local_resolved.is_dir() else "")
    cmd = ["rsync", "-az", "-e", rsync_rsh_arg(ssh_info), src, dst]
    subprocess.run(cmd, check=True)


def rsync_download(ssh_info: dict, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    src = f"{ssh_info['host']}:{remote_dir.rstrip('/')}/"
    dst = f"{local_dir.resolve()}/"
    cmd = ["rsync", "-az", "-e", rsync_rsh_arg(ssh_info), src, dst]
    subprocess.run(cmd, check=True)


def cmd_remote_check(args: argparse.Namespace) -> dict:
    cfg = resolve_remote_config(args)
    local_tools = {
        name: bool(shutil.which(name)) for name in ("ssh", "rsync", "ffmpeg")
    }

    probe_script = f"""
python3 - <<'PY'
import json, os, shutil, subprocess
from pathlib import Path

work = Path({cfg["workspace_dir"]!r})
venv_py = Path({cfg["remote_venv"]!r}) / "bin" / "python"
ckpt = Path({cfg["sam2_checkpoint"]!r})

gpu_info = None
if shutil.which("nvidia-smi"):
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if p.returncode == 0:
        gpu_info = [line.strip() for line in p.stdout.splitlines() if line.strip()]

torch_info = None
sam2_installed = False
if venv_py.is_file():
    code = "import json, torch; import sam2; print(json.dumps({{'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'sam2': True}}))"
    p2 = subprocess.run([str(venv_py), "-c", code], capture_output=True, text=True)
    if p2.returncode == 0:
        data = json.loads(p2.stdout.strip())
        torch_info = data
        sam2_installed = bool(data.get("sam2"))

print(json.dumps({{
    "connected": True,
    "workspace_dir": str(work),
    "workspace_exists": work.is_dir(),
    "venv_python": str(venv_py),
    "venv_exists": venv_py.is_file(),
    "sam2_checkpoint": str(ckpt),
    "checkpoint_exists": ckpt.is_file(),
    "gpus": gpu_info,
    "runtime": torch_info,
    "sam2_ready": bool(venv_py.is_file() and ckpt.is_file() and sam2_installed),
}}))
PY
"""
    raw = run_remote_shell(cfg["ssh"], probe_script)
    remote_status = json.loads(raw.strip().splitlines()[-1])
    return {
        "local_tools": local_tools,
        "ssh_host": cfg["ssh"]["host"],
        "workspace_user": cfg["workspace_user"],
        "project_name": cfg["project_name"],
        **remote_status,
    }


def cmd_remote_setup(args: argparse.Namespace) -> dict:
    cfg = resolve_remote_config(args)
    work = cfg["workspace_dir"]
    venv = cfg["remote_venv"]
    ckpt = cfg["sam2_checkpoint"]

    if not args.apply:
        return {
            "dry_run": True,
            "ssh_host": cfg["ssh"]["host"],
            "workspace_dir": work,
            "remote_venv": venv,
            "sam2_checkpoint": ckpt,
            "install_sam2": not args.skip_sam2,
            "install_ultralytics": args.with_ultralytics,
            "message": "Pass --apply to create directories, virtualenv, and download checkpoints on the remote server.",
        }

    q_work = shlex.quote(work)
    q_venv = shlex.quote(venv)
    q_ckpt = shlex.quote(ckpt)
    q_ckpt_dir = shlex.quote(str(Path(ckpt).parent))

    steps = [
        "set -e",
        f"mkdir -p {q_work} {q_ckpt_dir}",
        f"if [ ! -x {q_venv}/bin/python ]; then python3 -m venv {q_venv}; fi",
        f"{q_venv}/bin/python -m pip install --upgrade pip",
        f"{q_venv}/bin/python -m pip install 'numpy>=1.24' 'opencv-python-headless>=4.8' 'Pillow>=10.0'",
    ]
    if not args.skip_sam2:
        steps.extend(
            [
                f"if ! {q_venv}/bin/python -c 'import torch' >/dev/null 2>&1; then {q_venv}/bin/python -m pip install torch torchvision; fi",
                f"if ! {q_venv}/bin/python -c 'import sam2' >/dev/null 2>&1; then SAM2_BUILD_CUDA=0 {q_venv}/bin/python -m pip install 'git+https://github.com/facebookresearch/sam2.git@{SAM2_COMMIT}'; fi",
                f"if [ ! -f {q_ckpt} ]; then curl -L --fail -o {q_ckpt}.partial {shlex.quote(SAM2_CHECKPOINT_URL)} && mv {q_ckpt}.partial {q_ckpt}; fi",
            ]
        )
    if args.with_ultralytics:
        steps.append(
            f"if ! {q_venv}/bin/python -c 'import ultralytics' >/dev/null 2>&1; then {q_venv}/bin/python -m pip install 'ultralytics>=8.3'; fi"
        )

    run_remote_shell(cfg["ssh"], "\n".join(steps))
    return cmd_remote_check(args)


def dispatch_worker_command(
    args: argparse.Namespace,
    action: str,
    worker_args: list[str],
    local_out: Path,
    upload_paths: dict[str, Path],
) -> dict:
    out_resolved = ensure_new_path(local_out)
    if getattr(args, "local", False):
        from preannotate_worker import main as worker_main
        from io import StringIO
        import contextlib

        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            worker_main([action] + worker_args)
        return worker_result(buf.getvalue())

    cfg = resolve_remote_config(args)
    ssh_info = cfg["ssh"]
    run_id = f"{action}_{uuid.uuid4().hex}"
    remote_run_dir = f"{cfg['workspace_dir']}/runs/{run_id}"
    remote_runtime_dir = f"{remote_run_dir}/runtime"
    remote_in_dir = f"{remote_run_dir}/inputs"
    remote_out_dir = f"{remote_run_dir}/output"

    run_remote_shell(
        ssh_info,
        f"mkdir -p {shlex.quote(remote_runtime_dir)} {shlex.quote(remote_in_dir)} {shlex.quote(remote_out_dir)}",
    )

    here = Path(__file__).resolve().parent
    for mod_name in ("preannotate_core.py", "preannotate_worker.py"):
        rsync_upload(ssh_info, here / mod_name, f"{remote_runtime_dir}/{mod_name}")

    remapped_map: dict[str, str] = {}
    for key, path_obj in upload_paths.items():
        resolved = path_obj.expanduser().resolve()
        remote_target = f"{remote_in_dir}/{key}_{resolved.name}"
        if resolved.is_dir():
            run_remote_shell(ssh_info, f"mkdir -p {shlex.quote(remote_target)}")
        rsync_upload(ssh_info, resolved, remote_target)
        remapped_map[key] = remote_target

    final_worker_args: list[str] = []
    i = 0
    while i < len(worker_args):
        tok = worker_args[i]
        if tok == "--out":
            final_worker_args.extend(["--out", remote_out_dir])
            i += 2
        elif tok == "--dataset" and "dataset" in remapped_map:
            final_worker_args.extend(["--dataset", remapped_map["dataset"]])
            i += 2
        elif tok == "--weights" and "weights" in remapped_map:
            final_worker_args.extend(["--weights", remapped_map["weights"]])
            i += 2
        elif tok == "--model" and "model" in remapped_map:
            final_worker_args.extend(["--model", remapped_map["model"]])
            i += 2
        elif tok == "--ref-dataset" and i + 1 < len(worker_args):
            ref_key = worker_args[i + 1]
            final_worker_args.extend(
                ["--ref-dataset", remapped_map.get(ref_key, ref_key)]
            )
            i += 2
        else:
            final_worker_args.append(tok)
            i += 1

    venv_py = f"{cfg['remote_venv']}/bin/python"
    worker_py = f"{remote_runtime_dir}/preannotate_worker.py"
    quoted_args = " ".join(shlex.quote(str(a)) for a in [action] + final_worker_args)
    remote_cmd = (
        f"PYTHONPATH={shlex.quote(remote_runtime_dir)} "
        f"{shlex.quote(venv_py)} {shlex.quote(worker_py)} {quoted_args}"
    )
    try:
        raw_out = run_remote_shell(ssh_info, remote_cmd)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\nRemote artifacts retained at {remote_run_dir}"
        ) from None
    stage = out_resolved.with_name(out_resolved.name + f".partial-{run_id}")
    rsync_download(ssh_info, remote_out_dir, stage)
    if out_resolved.exists():
        raise ValueError(
            f"Output appeared during execution; results retained at {stage}"
        )
    stage.rename(out_resolved)

    remote_result = worker_result(raw_out)

    def localize(value):
        if isinstance(value, str) and value.startswith(remote_out_dir):
            return str(out_resolved) + value[len(remote_out_dir) :]
        if isinstance(value, list):
            return [localize(item) for item in value]
        if isinstance(value, dict):
            return {key: localize(item) for key, item in value.items()}
        return value

    remote_result = localize(remote_result)
    remote_result["local_output_dir"] = str(out_resolved)
    remote_result["remote_workspace_dir"] = cfg["workspace_dir"]
    remote_result["remote_run_dir"] = remote_run_dir
    return remote_result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preannotate",
        description="CLI for segmentation pre-annotation, SAM 2.1 propagation, smoothing, and CVAT proposal export.",
    )
    p.add_argument(
        "--env",
        default=str(DEFAULT_ENV),
        help="Path to KEY=value profile (default: ~/.config/preannotate/.env)",
    )
    p.add_argument(
        "--ssh", help="Override PREANNOTATE_SSH_TARGET (host alias or user@host)"
    )
    p.add_argument("--remote-root", help="Override PREANNOTATE_REMOTE_ROOT")
    p.add_argument("--user", help="Override PREANNOTATE_WORKSPACE_USER")
    p.add_argument(
        "--project",
        help="Override PREANNOTATE_WORKSPACE_DIR (remote project subfolder)",
    )
    p.add_argument("--version", action="version", version=f"preannotate {VERSION}")

    sub = p.add_subparsers(dest="command", required=True)

    # 1. remote-check
    sub.add_parser(
        "remote-check",
        help="Check local tools and remote SSH GPU workspace, virtualenv, and SAM 2 checkpoint.",
    )

    # 2. remote-setup
    rs = sub.add_parser(
        "remote-setup",
        help="Create remote user workspace, install Python/CUDA dependencies, and download SAM 2.1 checkpoint.",
    )
    rs.add_argument(
        "--apply",
        action="store_true",
        help="Execute remote installation (default is dry-run preview)",
    )
    rs.add_argument(
        "--skip-sam2",
        action="store_true",
        help="Skip PyTorch/SAM 2 installation and checkpoint download",
    )
    rs.add_argument(
        "--with-ultralytics",
        action="store_true",
        help="Also install ultralytics for YOLO-seg training",
    )

    # 3. inspect-dataset
    ins = sub.add_parser(
        "inspect-dataset",
        help="Inspect a CVAT 1.1 dataset directory or .zip file, reporting frame order and manual annotations.",
    )
    ins.add_argument(
        "dataset", type=Path, help="Dataset directory or CVAT 1.1 .zip export"
    )
    ins.add_argument("--xml", type=Path, help="Optional explicit annotations.xml path")
    ins.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )

    # 4. merge-datasets
    mg = sub.add_parser(
        "merge-datasets",
        help="Merge multiple CVAT 1.1 datasets (directories or .zip exports) into a unified dataset.",
    )
    mg.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input dataset folders or .zip files",
    )
    mg.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory for merged dataset",
    )
    mg.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    mg.add_argument(
        "--yolo-seg",
        action="store_true",
        help="Also export YOLO segmentation train/val split",
    )
    mg.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for --yolo-seg (default: 0.2)",
    )
    mg.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )

    # 5. baselines
    bl = sub.add_parser(
        "baselines",
        help="Run classical propagation baselines (optical flow, GrabCut, Lab appearance).",
    )
    bl.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory or CVAT 1.1 .zip"
    )
    bl.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    bl.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    bl.add_argument(
        "--local", action="store_true", help="Run locally instead of over SSH"
    )

    # 6. propagate
    pr = sub.add_parser(
        "propagate",
        help="Propagate manual mask annotations across video frames using SAM 2.1 on GPU.",
    )
    pr.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Target dataset directory or CVAT 1.1 .zip",
    )
    pr.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    pr.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    pr.add_argument(
        "--mode",
        choices=["full", "crop"],
        default="full",
        help="Full frame or vertical ROI crop",
    )
    pr.add_argument(
        "--crop-y",
        help="Vertical crop band Y0:Y1 (e.g. 300:620); auto-estimated if omitted in crop mode",
    )
    pr.add_argument(
        "--fusion",
        choices=["none", "background", "background-zero", "temporal"],
        default="none",
        help="Optional camera-stabilized background or temporal evidence fusion",
    )
    pr.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="SAM 2 logit threshold (default: 0.0)",
    )
    pr.add_argument(
        "--prompts",
        help="Comma-separated sequence indices of manual frames to use (default: all annotated)",
    )
    pr.add_argument(
        "--ref-dataset",
        action="append",
        default=[],
        help="Experimental: prepend annotated past-task frames; not training or a continuous video",
    )
    pr.add_argument("--checkpoint", help="Override SAM 2 checkpoint path")
    pr.add_argument("--config", help="Override SAM 2 config name")
    pr.add_argument("--device", default="cuda", help="Execution device (default: cuda)")
    pr.add_argument(
        "--local",
        action="store_true",
        help="Run on local machine instead of remote SSH server",
    )

    # 7. evaluate
    ev = sub.add_parser(
        "evaluate",
        help="Run Leave-One-Annotation-Out cross-validation across manual reference frames.",
    )
    ev.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset with at least 2 manual reference frames",
    )
    ev.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    ev.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    ev.add_argument("--crop-y", help="Vertical crop band Y0:Y1 (e.g. 300:620)")
    ev.add_argument("--checkpoint", help="Override SAM 2 checkpoint path")
    ev.add_argument("--config", help="Override SAM 2 config name")
    ev.add_argument("--device", default="cuda", help="Execution device (default: cuda)")
    ev.add_argument(
        "--local", action="store_true", help="Run locally instead of remote SSH server"
    )

    # 8. train-seg
    tr = sub.add_parser(
        "train-seg",
        help="Train a YOLO segmentation model on a merged or single CVAT 1.1 dataset.",
    )
    tr.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset directory (from merge-datasets or CVAT 1.1)",
    )
    tr.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New local output directory for trained weights and logs",
    )
    tr.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    tr.add_argument(
        "--model", default="yolo11n-seg.pt", help="Base Ultralytics segmentation model"
    )
    tr.add_argument(
        "--epochs", type=int, default=25, help="Training epochs (default: 25)"
    )
    tr.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    tr.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    tr.add_argument("--device", default="0", help="Training device (default: 0)")
    tr.add_argument(
        "--local", action="store_true", help="Run locally instead of remote SSH server"
    )

    # 9. predict-seg
    ps = sub.add_parser(
        "predict-seg",
        help="Predict segmentation masks using trained weights, optionally refined with SAM 2.1.",
    )
    ps.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Target dataset directory or CVAT 1.1 .zip",
    )
    ps.add_argument(
        "--weights", type=Path, required=True, help="Trained segmentation weights (.pt)"
    )
    ps.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    ps.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    ps.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    ps.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)"
    )
    ps.add_argument(
        "--refine-sam2",
        action="store_true",
        help="Use predicted masks to prompt SAM 2.1 temporal propagation",
    )
    ps.add_argument(
        "--checkpoint", help="Override SAM 2 checkpoint path when using --refine-sam2"
    )
    ps.add_argument("--config", help="Override SAM 2 config name")
    ps.add_argument("--device", default="cuda", help="Execution device (default: cuda)")
    ps.add_argument(
        "--local", action="store_true", help="Run locally instead of remote SSH server"
    )

    # 10. thresholds
    th = sub.add_parser(
        "thresholds",
        help="Compare multiple logit thresholds on saved SAM 2 logits without re-running GPU inference.",
    )
    th.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory or CVAT 1.1 .zip"
    )
    th.add_argument(
        "--logits",
        type=Path,
        required=True,
        help="Path to logits.npz or directory of .npy files",
    )
    th.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    th.add_argument(
        "--levels",
        default="0,2,4,6",
        help="Comma-separated logit thresholds (default: 0,2,4,6)",
    )
    th.add_argument(
        "--frames", help="Comma-separated frame indices for visual comparison sheet"
    )

    # 11. smooth
    sm = sub.add_parser(
        "smooth",
        help="Apply subpixel circular Gaussian contour smoothing and generate visual comparison sheets.",
    )
    sm.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory or CVAT 1.1 .zip"
    )
    sm.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Input directory containing binary masks",
    )
    sm.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    sm.add_argument(
        "--sigma",
        type=float,
        default=0.8,
        help="Contour smoothing sigma in pixels (default: 0.8)",
    )
    sm.add_argument(
        "--compare-levels",
        help="Optional comma-separated sigma levels to generate comparison sheets (e.g. 0,0.5,0.8,1.2,1.6,2.0)",
    )
    sm.add_argument(
        "--frames",
        help="Optional comma-separated frame indices for comparison sheets (e.g. 3,20,30)",
    )
    sm.add_argument(
        "--step",
        type=float,
        default=0.4,
        help="Resampling step in pixels (default: 0.4)",
    )

    # 12. render-video
    rv = sub.add_parser(
        "render-video",
        help="Render H.264 MP4 (and optional GIF) comparison or demonstration video from masks.",
    )
    rv.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory or CVAT 1.1 .zip"
    )
    rv.add_argument(
        "--masks",
        nargs="+",
        type=Path,
        required=True,
        help="One or more mask directories to compare",
    )
    rv.add_argument(
        "--titles", nargs="+", help="Optional panel titles matching --masks"
    )
    rv.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    rv.add_argument(
        "--fps", type=int, default=4, help="Playback frame rate (default: 4)"
    )
    rv.add_argument("--crop-y", help="Optional vertical crop band Y0:Y1")
    rv.add_argument("--gif", action="store_true", help="Also generate animated GIF")

    # 13. export-proposals
    ep = sub.add_parser(
        "export-proposals",
        help="Export predicted masks (with optional contour smoothing) to proposals.json (for cvat append) and CVAT 1.1 XML/ZIP (for cvat import-missing).",
    )
    ep.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset directory or CVAT 1.1 .zip with current manual annotations",
    )
    ep.add_argument(
        "--masks", type=Path, required=True, help="Directory of predicted binary masks"
    )
    ep.add_argument(
        "--out", type=Path, required=True, help="New local output directory"
    )
    ep.add_argument(
        "--label", default="smoke", help="Target label name (default: smoke)"
    )
    ep.add_argument(
        "--sigma",
        type=float,
        default=0.8,
        help="Subpixel contour smoothing sigma in px (0 disables smoothing; default: 0.8)",
    )
    ep.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="Minimum contour area in px (default: 1.0)",
    )
    ep.add_argument(
        "--step", type=float, default=0.4, help="Resampling step in px (default: 0.4)"
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cmd = args.command

    if cmd == "remote-check":
        res = cmd_remote_check(args)
    elif cmd == "remote-setup":
        res = cmd_remote_setup(args)
    elif cmd == "inspect-dataset":
        res = inspect_dataset(args.dataset, xml_override=args.xml, label=args.label)
    elif cmd == "merge-datasets":
        res = merge_datasets(
            inputs=args.inputs,
            out_dir=args.out,
            label=args.label,
            export_yolo_seg=args.yolo_seg,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif cmd == "baselines":
        res = dispatch_worker_command(
            args,
            "baselines",
            [
                "--dataset",
                str(args.dataset),
                "--out",
                str(args.out),
                "--label",
                args.label,
            ],
            local_out=args.out,
            upload_paths={"dataset": args.dataset},
        )
    elif cmd == "propagate":
        cfg = resolve_remote_config(args) if not args.local else {}
        ckpt = args.checkpoint or cfg.get("sam2_checkpoint", "")
        config_name = args.config or cfg.get(
            "sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        if not ckpt:
            raise ValueError(
                "Missing SAM 2 checkpoint path (--checkpoint or PREANNOTATE_SAM2_CHECKPOINT)"
            )
        w_args = [
            "--dataset",
            str(args.dataset),
            "--out",
            str(args.out),
            "--checkpoint",
            str(ckpt),
            "--config",
            str(config_name),
            "--label",
            args.label,
            "--mode",
            args.mode,
            "--fusion",
            args.fusion,
            "--threshold",
            str(args.threshold),
            "--device",
            args.device,
        ]
        if args.crop_y:
            w_args.extend(["--crop-y", args.crop_y])
        if args.prompts:
            w_args.extend(["--prompts", args.prompts])
        uploads = {"dataset": args.dataset}
        for r_idx, r_path in enumerate(args.ref_dataset):
            key = f"ref_{r_idx:02d}"
            uploads[key] = Path(r_path)
            w_args.extend(
                [
                    "--ref-dataset",
                    key if not args.local else str(Path(r_path).resolve()),
                ]
            )
        res = dispatch_worker_command(
            args, "propagate", w_args, local_out=args.out, upload_paths=uploads
        )
    elif cmd == "evaluate":
        cfg = resolve_remote_config(args) if not args.local else {}
        ckpt = args.checkpoint or cfg.get("sam2_checkpoint", "")
        config_name = args.config or cfg.get(
            "sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        if not ckpt:
            raise ValueError(
                "Missing SAM 2 checkpoint path (--checkpoint or PREANNOTATE_SAM2_CHECKPOINT)"
            )
        w_args = [
            "--dataset",
            str(args.dataset),
            "--out",
            str(args.out),
            "--checkpoint",
            str(ckpt),
            "--config",
            str(config_name),
            "--label",
            args.label,
            "--device",
            args.device,
        ]
        if args.crop_y:
            w_args.extend(["--crop-y", args.crop_y])
        res = dispatch_worker_command(
            args,
            "evaluate",
            w_args,
            local_out=args.out,
            upload_paths={"dataset": args.dataset},
        )
    elif cmd == "train-seg":
        w_args = [
            "--dataset",
            str(args.dataset),
            "--out",
            str(args.out),
            "--label",
            args.label,
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
        ]
        uploads = {"dataset": args.dataset}
        if Path(args.model).is_file():
            uploads["model"] = Path(args.model)
        res = dispatch_worker_command(
            args,
            "train-seg",
            w_args,
            local_out=args.out,
            upload_paths=uploads,
        )
    elif cmd == "predict-seg":
        cfg = resolve_remote_config(args) if not args.local else {}
        w_args = [
            "--dataset",
            str(args.dataset),
            "--weights",
            str(args.weights),
            "--out",
            str(args.out),
            "--label",
            args.label,
            "--conf",
            str(args.conf),
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
        ]
        if args.refine_sam2:
            ckpt = args.checkpoint or cfg.get("sam2_checkpoint", "")
            config_name = args.config or cfg.get(
                "sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"
            )
            w_args.extend(
                [
                    "--refine-sam2",
                    "--checkpoint",
                    str(ckpt),
                    "--config",
                    str(config_name),
                ]
            )
        res = dispatch_worker_command(
            args,
            "predict-seg",
            w_args,
            local_out=args.out,
            upload_paths={"dataset": args.dataset, "weights": args.weights},
        )
    elif cmd == "thresholds":
        levels = [float(x) for x in args.levels.split(",") if x.strip()]
        frames_list = (
            [int(x) for x in args.frames.split(",") if x.strip()]
            if args.frames
            else None
        )
        res = run_thresholds(
            dataset_path=args.dataset,
            logits_path=args.logits,
            out_dir=args.out,
            levels=levels,
            sample_frames=frames_list,
        )
    elif cmd == "smooth":
        comp_levels = (
            [float(x) for x in args.compare_levels.split(",") if x.strip()]
            if args.compare_levels
            else None
        )
        comp_frames = (
            [int(x) for x in args.frames.split(",") if x.strip()]
            if args.frames
            else None
        )
        res = run_smooth(
            dataset_path=args.dataset,
            masks_dir=args.masks,
            out_dir=args.out,
            sigma=args.sigma,
            compare_levels=comp_levels,
            compare_frames=comp_frames,
            step=args.step,
        )
    elif cmd == "render-video":
        crop_band = None
        if args.crop_y:
            y0_s, y1_s = args.crop_y.split(":")
            crop_band = (int(y0_s), int(y1_s))
        res = render_video(
            dataset_path=args.dataset,
            masks_dirs=args.masks,
            out_dir=args.out,
            titles=args.titles,
            fps=args.fps,
            crop_y=crop_band,
            gif=args.gif,
        )
    elif cmd == "export-proposals":
        res = export_proposals(
            dataset_path=args.dataset,
            masks_dir=args.masks,
            out_dir=args.out,
            label=args.label,
            sigma=args.sigma,
            min_area=args.min_area,
            step=args.step,
        )
    else:
        raise ValueError(f"Unknown command: {cmd}")

    print(json.dumps(res, indent=2))
    return 0


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    entrypoint()
