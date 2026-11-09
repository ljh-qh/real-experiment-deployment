#!/usr/bin/env python3
"""Verify deployment isolation without contacting a GPU, robot, or network.

Run with deployment/envs/robot/bin/python -B deployment/scripts/verify_isolation.py.
The report separates historical/documentary paths from active runtime dependencies.
Default checks use only this deployment and system dependencies. Add
--compare-sources to explicitly recheck the original source directories.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tokenize

ROOT = Path(__file__).resolve().parents[1]
OLD_PROJECT = "/home/admin2026/guchenyang/Project/"
TEXT_SUFFIXES = {".py", ".sh", ".bash", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".pth", ".md", ".txt", ".xml", ".urdf", ".cmake"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
HISTORICAL_PARTS = {"fixtures", "fixture", "references", "reference", "tests", "test", "docs", "history", "archive", "archived", "historical", "migration", "offline_checks", "runs", "logs", "experiments"}
ACTIVE_PREFIXES = ("src/", "configs/", "scripts/", "vendor/bridge/", "vendor/universal_robot_infra/src/", "vendor/mujoco_marvin/mujoco_marvin/", "sdk/")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files(base: Path, prune: set[str] | None = None):
    for current, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [name for name in dirs if name not in (prune or set())]
        for name in files:
            yield Path(current) / name


def sha256(path: Path, *, allow_source: bool = False) -> str:
    if not allow_source and not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Refusing to read outside deployment: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def documentary_lines(path: Path, text: str) -> set[int]:
    lines = set()
    if path.suffix == ".py":
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    # A trailing comment does not exempt executable text before it.
                    if OLD_PROJECT in token.string and OLD_PROJECT not in text.splitlines()[token.start[0] - 1][:token.start[1]]:
                        lines.add(token.start[0])
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    statement = node.body[0]
                    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                        lines.update(range(statement.lineno, statement.end_lineno + 1))
        except (SyntaxError, tokenize.TokenError, IndentationError):
            pass
    else:
        lines.update(i for i, line in enumerate(text.splitlines(), 1) if line.lstrip().startswith(("#", "<!--")))
    return lines


def source_paths() -> dict:
    result = {"active_roots": list(ACTIVE_PREFIXES), "active_violations": [], "excluded_references": [], "scanned_text_files": 0}
    for path in iter_files(ROOT, SKIP_DIRS | {"envs", "deps", "weights", "checkpoints", "offline_checks"}):
        rel = relative(path)
        if path.is_symlink() or path.suffix not in TEXT_SUFFIXES or path.stat().st_size > 4_000_000:
            continue
        try:
            text = path.read_text()
        except (UnicodeError, OSError):
            continue
        result["scanned_text_files"] += 1
        if OLD_PROJECT not in text:
            continue
        comments = documentary_lines(path, text)
        parts = set(path.relative_to(ROOT).parts)
        historical = bool(parts & HISTORICAL_PARTS) or rel.startswith(("assets/checkpoints/", "assets/references/"))
        active = rel.startswith(ACTIVE_PREFIXES)
        for lineno, line in enumerate(text.splitlines(), 1):
            if OLD_PROJECT not in line:
                continue
            # The checker necessarily records the old prefix it is checking for.
            reason = "verification_source_constant" if path.resolve() == Path(__file__).resolve() else "historical_or_fixture" if historical else "comment_or_docstring" if lineno in comments else "non_runtime_source" if not active else None
            item = {"file": rel, "line": lineno, "text": line.strip()[:600]}
            if reason:
                item["reason"] = reason
                result["excluded_references"].append(item)
            else:
                result["active_violations"].append(item)
    result["ok"] = not result["active_violations"]
    return result


def yaml_paths() -> dict:
    import yaml

    output = ROOT / "runs" / "__isolation_check_not_started__"
    result = {"checked": [], "deferred_outputs": [], "errors": [], "yaml_files": [], "output_placeholder": str(output)}
    configs = ROOT / "configs"
    for path in sorted(list(configs.rglob("*.yaml")) + list(configs.rglob("*.yml"))):
        result["yaml_files"].append(relative(path))
        try:
            document = yaml.safe_load(path.read_text())
        except Exception as exc:
            result["errors"].append({"file": relative(path), "error": repr(exc)})
            continue

        def check_value(value, field=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    check_value(child, f"{field}.{key}" if field else str(key))
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    check_value(child, f"{field}[{i}]")
            elif isinstance(value, str):
                resolved = value.replace("$DEPLOY", str(ROOT)).replace("$OUTPUT", str(output))
                if re.search(r"\$\{?[A-Za-z_]", resolved):
                    result["errors"].append({"file": relative(path), "field": field, "value": value, "error": "unresolved placeholder"})
                # All $DEPLOY values are filesystem paths. List environment paths independently.
                is_env_paths = field.endswith(("PYTHONPATH", "LD_LIBRARY_PATH", ".PATH"))
                is_path_field = field in {"worker.python", "worker.script", "worker.cwd"} or field.startswith("worker.args.") and field.rsplit(".", 1)[-1] in {"checkpoint", "config", "codec", "warmup-fixtures", "checkpoint-root", "weights", "reference", "validation-file"}
                if "$DEPLOY" in value or "$OUTPUT" in value or is_path_field:
                    for candidate in resolved.split(os.pathsep) if is_env_paths else [resolved]:
                        item = {"file": relative(path), "field": field, "resolved": candidate}
                        if candidate == str(output) or candidate.startswith(str(output) + "/"):
                            item["reason"] = "created by a real run; not required during offline validation"
                            result["deferred_outputs"].append(item)
                        else:
                            target = Path(candidate)
                            item["exists"] = False if OLD_PROJECT in str(target) else target.exists()
                            result["checked"].append(item)
                            if not item["exists"]:
                                result["errors"].append({**item, "error": "referenced input does not exist"})

        check_value(document)
        if isinstance(document, dict) and isinstance(document.get("worker"), dict):
            args = document["worker"].get("args", {})
            if isinstance(args, dict) and args.get("checkpoint") and args.get("checkpoint-name"):
                target = Path(str(args["checkpoint"]).replace("$DEPLOY", str(ROOT))) / args["checkpoint-name"]
                item = {"file": relative(path), "field": "worker.args.checkpoint + checkpoint-name", "resolved": str(target), "exists": False if OLD_PROJECT in str(target) else target.exists()}
                result["checked"].append(item)
                if not item["exists"]:
                    result["errors"].append({**item, "error": "checkpoint file does not exist"})
            if document.get("client_entrypoint"):
                target = ROOT / "src" / document["client_entrypoint"]
                item = {"file": relative(path), "field": "client_entrypoint", "resolved": str(target), "exists": False if OLD_PROJECT in str(target) else target.exists()}
                result["checked"].append(item)
                if not item["exists"]:
                    result["errors"].append({**item, "error": "client entry point does not exist"})
    if not result["yaml_files"]:
        result["errors"].append({"error": "no deployment config YAML files found"})
    result["ok"] = not result["errors"]
    return result


def links_and_entries() -> dict:
    result = {"symlinks": [], "violations": [], "pth_files": [], "shebang_files_checked": 0}
    for current, dirs, files in os.walk(ROOT, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            if path.is_symlink():
                raw = os.readlink(path)
                resolved = str(path.resolve())
                item = {"file": relative(path), "target": raw, "resolved": resolved, "exists": None if OLD_PROJECT in resolved else path.exists()}
                result["symlinks"].append(item)
                if OLD_PROJECT in raw or OLD_PROJECT in resolved:
                    result["violations"].append({**item, "error": "symlink returns to another project"})
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and name != "offline_checks"]
        for name in files:
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            # Only read likely startup entries, never multi-GB model files.
            if path.suffix != ".pth" and path.parent.name != "bin" and path.suffix not in {".py", ".sh", ".bash"}:
                continue
            with path.open("rb") as stream:
                data = stream.read(8192)
            if path.suffix == ".pth":
                result["pth_files"].append(relative(path))
                if OLD_PROJECT.encode() in data:
                    result["violations"].append({"file": relative(path), "error": ".pth references another project"})
            if data.startswith(b"#!"):
                result["shebang_files_checked"] += 1
                if OLD_PROJECT.encode() in data.splitlines()[0]:
                    result["violations"].append({"file": relative(path), "error": "shebang references another project"})
    result["ok"] = not result["violations"]
    return result


def official_hashes(compare_sources: bool = False) -> dict:
    baseline_path = ROOT / "migration/source_hashes.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("algorithm") != "sha256" or baseline.get("schema_version") != 1:
        raise ValueError("Unsupported official source hash baseline")
    result = {"baseline": relative(baseline_path), "baseline_sha256": sha256(baseline_path), "source_comparison_requested": compare_sources, "trees": {}, "ok": True}
    for name, tree in baseline["trees"].items():
        target = ROOT / tree["target_relative"]
        expected = tree["files"]
        entry = {"target": str(target), "checked_files": 0, "missing": [], "different": [], "extra": [], "excluded": sorted(SKIP_DIRS)}
        if not target.resolve().is_relative_to(ROOT):
            entry.update(error="target tree escapes deployment", ok=False)
        elif not target.is_dir():
            entry.update(error="target tree missing", ok=False)
        else:
            target_files = {p.relative_to(target).as_posix(): p for p in iter_files(target, SKIP_DIRS) if p.suffix != ".pyc"}
            for rel, expected_hash in expected.items():
                copied = target_files.get(rel)
                if copied is None:
                    entry["missing"].append(rel)
                elif not copied.resolve().is_relative_to(ROOT):
                    entry["different"].append({"file": rel, "error": "refused to read external link target"})
                elif sha256(copied) != expected_hash:
                    entry["different"].append(rel)
                entry["checked_files"] += 1
            entry["extra"] = sorted(target_files.keys() - expected.keys())
            entry["ok"] = not (entry["missing"] or entry["different"] or entry["extra"])
        # Source provenance is data until the caller explicitly opts in.
        if compare_sources:
            source = Path(tree["source_provenance"])
            source_result = {"source": str(source), "missing": [], "different": [], "extra": []}
            if not source.is_dir():
                source_result.update(error="optional source tree is unavailable", ok=False)
            else:
                source_files = {p.relative_to(source).as_posix(): p for p in iter_files(source, SKIP_DIRS) if p.suffix != ".pyc"}
                for rel, expected_hash in expected.items():
                    original = source_files.get(rel)
                    if original is None:
                        source_result["missing"].append(rel)
                    elif sha256(original, allow_source=True) != expected_hash:
                        source_result["different"].append(rel)
                source_result["extra"] = sorted(source_files.keys() - expected.keys())
                source_result["ok"] = not (source_result["missing"] or source_result["different"] or source_result["extra"])
            entry["optional_source_comparison"] = source_result
            entry["ok"] = entry["ok"] and source_result["ok"]
        result["trees"][name] = entry
        result["ok"] = result["ok"] and entry["ok"]
    return result


def environment_imports(report_dir: Path) -> dict:
    import yaml

    # Use the launcher's actual environment; injecting extra paths here would
    # conceal missing runtime dependencies.
    overrides = {}
    for name, profile in [("openwam", "openwam_sync"), ("easywam", "easywam_v1")]:
        config = yaml.safe_load((ROOT / "configs/profiles" / (profile + ".yaml")).read_text())
        overrides[name] = {key: str(value).replace("$DEPLOY", str(ROOT)) for key, value in config["worker"]["env"].items()}
    spec = importlib.util.spec_from_file_location("_isolation_deployment_paths", ROOT / "src/deployment_paths.py")
    path_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(path_module)
    overrides["robot"] = path_module.robot_env()
    modules = {
        "openwam": ["numpy", "torch", "transformers", "diffusers", "openwam", "zmq", "zmq_chunk_worker"],
        "easywam": ["numpy", "torch", "transformers", "diffusers", "flash_attn", "zmq", "easywam_worker"],
        "robot": ["numpy", "scipy", "mujoco", "mujoco_marvin", "unirobot_infra", "zmq", "msgpack", "omegaconf", "return_full_origin", "mujoco_marvin.tianji_backend", "mujoco_marvin.sharpa_backend", "policy_hands", "infra_state_client", "rtc_state_client", "live_observation_source", "zmq_chunk_client", "pyrealsense2"],
    }
    code = """import importlib,json,sys,os
blocked=[]
def audit(event,args):
 if event in ('open','os.scandir','os.listdir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=os.fsdecode(args[0])
  if sys.argv[2] in path:
   blocked.append({'event':event,'path':path});raise PermissionError('Old project access is prohibited during offline import verification')
sys.addaudithook(audit)
r={'prefix':sys.prefix,'executable':sys.executable,'sys_path_before_imports':list(sys.path),'imports':{}}
for n in json.loads(sys.argv[1]):
 try:
  m=importlib.import_module(n);r['imports'][n]={'file':getattr(m,'__file__',None),'paths':list(getattr(m,'__path__',[])),'ok':True}
 except Exception as e:r['imports'][n]={'ok':False,'error':repr(e)}
if 'torch' in sys.modules:r['cuda_initialized']=sys.modules['torch'].cuda.is_initialized()
r['blocked_source_accesses']=blocked
print(json.dumps(r))
"""
    result = {"environments": {}, "ok": True, "gpu_and_hardware_started": False}
    with tempfile.TemporaryDirectory(prefix="isolation-import-", dir=report_dir) as tmp:
        for name in overrides:
            env = os.environ.copy()
            env.update(overrides[name])
            env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TMPDIR": tmp, "XDG_CACHE_HOME": tmp, "HF_HOME": str(Path(tmp) / "huggingface"), "TORCH_HOME": str(Path(tmp) / "torch"), "TRITON_CACHE_DIR": str(Path(tmp) / "triton")})
            python = ROOT / "envs" / name / "bin/python"
            try:
                run = subprocess.run([str(python), "-B", "-c", code, json.dumps(modules[name]), OLD_PROJECT], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
                item = json.loads(run.stdout.splitlines()[-1])
                item.update({"exit_code": run.returncode, "stderr": run.stderr, "violations": [], "runtime_pythonpath": env.get("PYTHONPATH", "")})
                if item.get("blocked_source_accesses"):
                    item["violations"].append("attempted to read another project")
                if item["prefix"] != str(ROOT / "envs" / name):
                    item["violations"].append("wrong virtual environment prefix")
                if item.get("cuda_initialized", False):
                    item["violations"].append("CUDA was unexpectedly initialized")
                for module, info in item["imports"].items():
                    origins = [info["file"]] if info.get("file") else info.get("paths", [])
                    if not info["ok"] or not origins or any(not str(origin).startswith(str(ROOT) + "/") for origin in origins):
                        item["violations"].append({"module": module, "detail": info})
                if any(OLD_PROJECT in x for x in item["sys_path_before_imports"]):
                    item["violations"].append("sys.path still references another project")
                item["ok"] = run.returncode == 0 and not item["violations"]
            except Exception as exc:
                item = {"ok": False, "error": repr(exc)}
            result["environments"][name] = item
            result["ok"] = result["ok"] and item["ok"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "offline_checks/migration_verify.json")
    parser.add_argument("--compare-sources", action="store_true", help="Also read the original source directories; default uses local hash baseline only")
    args = parser.parse_args()
    blocked_source_accesses = []
    if not args.compare_sources:
        def audit(event, arguments):
            if event in ("open", "os.scandir", "os.listdir") and arguments and isinstance(arguments[0], (str, bytes, os.PathLike)):
                path = os.fsdecode(arguments[0])
                if OLD_PROJECT in path:
                    blocked_source_accesses.append({"event": event, "path": path})
                    raise PermissionError("Default verification must not read the original projects; use --compare-sources to opt in")
        sys.addaudithook(audit)
    report_path = args.report.resolve()
    if not report_path.is_relative_to(ROOT):
        parser.error("report must stay inside this deployment")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"deployment_root": str(ROOT), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "scope": "offline filesystem and CPU import checks only; no model inference or hardware connection", "compare_sources_requested": args.compare_sources, "checks": {}}
    checks = [("source_paths", source_paths), ("yaml_paths", yaml_paths), ("links_and_entries", links_and_entries), ("official_hashes", lambda: official_hashes(args.compare_sources)), ("environment_imports", lambda: environment_imports(report_path.parent))]
    for name, check in checks:
        try:
            report["checks"][name] = check()
        except Exception as exc:
            report["checks"][name] = {"ok": False, "error": repr(exc)}
        print(f"{name}: {'PASS' if report['checks'][name]['ok'] else 'FAIL'}", flush=True)
    report["blocked_source_accesses"] = blocked_source_accesses
    report["ok"] = not blocked_source_accesses and all(item["ok"] for item in report["checks"].values())
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {report_path}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
