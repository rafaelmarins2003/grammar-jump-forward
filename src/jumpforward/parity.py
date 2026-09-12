"""Check offline token IDs against an owned, unmodified llama-server process."""

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from llguidance import LLTokenizer

from .corpus import SEED, digest, read_jsonl
from .phase0 import ROOT, provenance, write_json
from .spans import SERIALIZATIONS


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


@contextmanager
def owned_server(run: Path, gpu: bool = False):
    config = json.loads((ROOT / "config/models.json").read_text())["targets"]["qwen3-8b-q4_k_m"]
    model = Path(config["path"]).expanduser()
    actual_hash = file_hash(model)
    if actual_hash != config["sha256"]:
        raise ValueError("GGUF content does not match the pinned model")
    binary = Path.home() / "llama.cpp-jf/build/bin/llama-server"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        str(binary),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "99" if gpu else "0",
        "-c",
        "8192" if gpu else "512",
        "-np",
        "1",
        "--reasoning",
        "off",
        "--no-warmup",
        "-fa",
        "on",
    ]
    # Linux: stop the owned server even if Python is interrupted before finally runs.
    # setpriv is used only to set the parent-death signal, not to change privileges.
    launcher = shutil.which("setpriv") if sys.platform == "linux" else None
    launched_command = [launcher, "--pdeathsig", "TERM", "--", *command] if launcher else command
    write_json(
        run / "server-manifest.json",
        {
            **provenance(),
            "server_command": command,
            "launched_command": launched_command,
            "binary_sha256": file_hash(binary),
            "verified_model_sha256": actual_hash,
            "engine_head": subprocess.check_output(
                ["git", "-C", str(binary.parents[2]), "rev-parse", "HEAD"], text=True
            ).strip(),
            "engine_status": subprocess.check_output(
                ["git", "-C", str(binary.parents[2]), "status", "--short"], text=True
            ).strip(),
        },
    )
    with (run / "server.log").open("x") as log:
        process = subprocess.Popen(launched_command, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"owned server exited: {process.returncode}")
                    try:
                        response = client.get("/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.25)
                else:
                    raise TimeoutError("owned server did not become healthy")
                yield client
        finally:
            if process.poll() is None:
                process.terminate()  # Only the child created above, never an existing user server.
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    calls = [
        {"subset": c["subset"], "id": c["id"], "call_index": j, "call": call}
        for c in read_jsonl(args.cases)
        if c["status"] == "ok"
        for j, call in enumerate(c["calls"])
    ]
    calls.sort(
        key=lambda c: digest(f"{SEED}:parity:{c['subset']}:{c['id']}:{c['call_index']}".encode())
    )
    if len(calls) < args.count:
        raise ValueError("not enough distinct call records for requested parity sample")
    calls = calls[: args.count]
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(args.run / "selection.json", calls)
    tokenizer = LLTokenizer(
        str(ROOT / "data/raw/tokenizers/qwen3/tokenizer.json"), eos_token=151645
    )
    rows = []
    with owned_server(args.run) as client, (args.run / "parity.jsonl").open("x") as stream:
        for call in calls:
            for serial, separators in SERIALIZATIONS.items():
                text = json.dumps(call["call"], ensure_ascii=False, separators=separators)
                response = client.post(
                    "/tokenize",
                    json={
                        "content": text,
                        "add_special": False,
                        "parse_special": False,
                        "with_pieces": False,
                    },
                )
                response.raise_for_status()
                native = response.json()["tokens"]
                offline = tokenizer.tokenize_str(text)
                row = {
                    **call,
                    "serialization": serial,
                    "text": text,
                    "offline": offline,
                    "native": native,
                    "equal": offline == native,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    summary = {
        "n_calls": len(calls),
        "n_serializations": len(rows),
        "mismatches": sum(not r["equal"] for r in rows),
        "scope": "call-body IDs only; does not prove equality of whole matcher vocabularies",
        "cases_sha256": digest(args.cases.read_bytes()),
        "parity_sha256": digest((args.run / "parity.jsonl").read_bytes()),
    }
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
