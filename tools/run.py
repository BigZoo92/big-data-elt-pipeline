"""
Cross-OS runner to bootstrap the full stack (docker compose + pipelines + serving).
Usage:
  python -m tools.run up     # start stack, run pipelines, publish, benchmarks
  python -m tools.run down   # stop services
  python -m tools.run reset  # stop + remove volumes
  python -m tools.run logs   # follow logs
  python -m tools.run status # docker compose ps
"""
from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

from urllib import request

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
COMPOSE_FILES = ["docker-compose.yml", "docker-compose.serving.yml", "docker-compose.dev.yml"]

HTTP_HEALTH = [
    ("minio", "http://localhost:9000/minio/health/live"),
    ("flask-api", "http://localhost:5000/health"),
    ("metabase", "http://localhost:3000/api/health"),
]


def _compose_base(args: Iterable[str]) -> List[str]:
    cmd: List[str] = ["docker", "compose"]
    for file in COMPOSE_FILES:
        path = ROOT / file
        if path.exists():
            cmd.extend(["-f", str(path)])
    cmd.extend(args)
    return cmd


def _run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def _ensure_env_file() -> None:
    if ENV_FILE.exists():
        return
    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        print(f"Copied {ENV_EXAMPLE.name} -> {ENV_FILE.name}")
    else:
        print("No .env or .env.example found; continuing without env file.")


def _check_docker() -> None:
    for cmd in (["docker", "--version"], ["docker", "compose", "version"]):
        _run(cmd, check=True)


def _wait_http(name: str, url: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    print(f"✓ {name} healthy ({url})")
                    return True
        except Exception:
            pass
        time.sleep(5)
    print(f"✗ Timeout waiting for {name} at {url}", file=sys.stderr)
    return False


def _wait_tcp(name: str, host: str, port: int, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"✓ {name} reachable on {host}:{port}")
                return True
        except OSError:
            time.sleep(5)
    print(f"✗ Timeout waiting for {name} on {host}:{port}", file=sys.stderr)
    return False


def wait_stack() -> bool:
    ok = True
    for name, url in HTTP_HEALTH:
        ok = _wait_http(name, url) and ok
    ok = _wait_tcp("mongodb", "localhost", 27017) and ok
    return ok


def compose_up() -> None:
    _check_docker()
    _ensure_env_file()
    print("Starting stack (docker compose up -d --build)...")
    _run(_compose_base(["up", "-d", "--build"]))
    if not wait_stack():
        raise RuntimeError("Stack health checks failed")


def compose_down(volumes: bool = False) -> None:
    args = ["down"]
    if volumes:
        args.append("-v")
    _run(_compose_base(args))


def compose_logs() -> None:
    _run(_compose_base(["logs", "-f"]), check=False)


def compose_status() -> None:
    _run(_compose_base(["ps"]))


def run_pipeline() -> None:
    steps = [
        ("generate data", ["python", "scripts/generate_data.py"]),
        ("pandas bronze", ["python", "-m", "flows.bronze_ingestion"]),
        ("pandas silver", ["python", "-m", "flows.silver_transformation"]),
        ("pandas gold", ["python", "-m", "flows.gold_transformation"]),
        ("spark bronze", ["python", "-m", "flows_spark.bronze_ingestion_spark"]),
        ("spark silver", ["python", "-m", "flows_spark.silver_transformation_spark"]),
        ("spark gold", ["python", "-m", "flows_spark.gold_transformation_spark"]),
        ("publish gold -> mongo", ["python", "-m", "serving_mongo.publish_gold_to_mongo"]),
        ("benchmark", ["python", "scripts/benchmark.py"]),
    ]
    for label, cmd in steps:
        print(f"-> {label}")
        _run(_compose_base(["run", "--rm", "runner", *cmd]))


def cmd_up() -> None:
    compose_up()
    run_pipeline()
    print("\nStack ready:")
    print("  ✓ Streamlit: http://localhost:8501")
    print("  ✓ Metabase:  http://localhost:3000")
    print("  ✓ MLflow:    http://localhost:5001")
    print("  ✓ API:       http://localhost:5000/health")


def cmd_down() -> None:
    compose_down(volumes=False)


def cmd_reset() -> None:
    compose_down(volumes=True)


def cmd_logs() -> None:
    compose_logs()


def cmd_status() -> None:
    compose_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack runner (docker compose + pipelines).")
    parser.add_argument(
        "command",
        choices=["up", "down", "reset", "logs", "status"],
        help="Action to run",
    )
    args = parser.parse_args()

    actions = {
        "up": cmd_up,
        "down": cmd_down,
        "reset": cmd_reset,
        "logs": cmd_logs,
        "status": cmd_status,
    }
    try:
        actions[args.command]()
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
