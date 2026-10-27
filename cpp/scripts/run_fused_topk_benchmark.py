# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded public-API screening; invoke only inside the managed GPU queue.

Three fresh processes execute the seeded PK2 12-case or PK4 16-case matrix.
The optional near-tie cell is numerical evidence, not performance promotion.
This records opt-in policy timing, not a claim that the fused backend launched.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("pk2", "pk4"), default="pk2")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One Gaussian L2 case per process (PK2 D32; PK4 D128)",
    )
    parser.add_argument("--include-near-tie", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.include_near_tie:
        parser.error("--smoke and --include-near-tie are mutually exclusive")
    if args.family == "pk4" and args.include_near_tie:
        parser.error("PK4 near-tie semantics are covered by the unit fixtures")
    binary = args.binary.resolve(strict=True)
    root = Path(__file__).resolve().parents[2]

    def git(*command):
        return subprocess.check_output(["git", "-C", str(root), *command])

    def digest(path):
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        return checksum.hexdigest()

    linkage = subprocess.check_output(["ldd", str(binary)], text=True)
    libraries = []
    for line in linkage.splitlines():
        if "=>" not in line:
            continue
        name, location = line.split("=>", 1)
        location = location.strip().split(" (", 1)[0]
        if not location.startswith("/"):
            continue
        path = Path(location).resolve(strict=True)
        fingerprint = path.is_relative_to(root) or name.strip().startswith(
            "libcuvs.so"
        )
        libraries.append(
            {
                "name": name.strip(),
                "realpath": str(path),
                "sha256": digest(path) if fingerprint else None,
            }
        )
    if not any(row["name"].startswith("libcuvs.so") for row in libraries):
        raise RuntimeError(
            "Cannot identify loaded libcuvs.so; refusing incomplete evidence"
        )
    dirty = hashlib.sha256(git("diff", "HEAD", "--binary"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    for name in sorted(untracked.split(b"\0")):
        if name:
            dirty.update(name)
            dirty.update((root / name.decode()).read_bytes())
    metadata = {
        "source_revision": git("rev-parse", "HEAD").decode().strip(),
        "source_status": git("status", "--short").decode(),
        "dirty_digest_sha256": dirty.hexdigest(),
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "dynamic_libraries": libraries,
        "ldd_output": linkage,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "PATH",
            )
        },
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cmake_cache_sha256": digest(root / "cpp/build/CMakeCache.txt"),
        "cuda_compiler_metadata": {
            str(path.relative_to(root)): path.read_text()
            for path in (root / "cpp/build/CMakeFiles").glob(
                "*/CMakeCUDACompiler.cmake"
            )
        },
        "process_repetitions": 3,
        "family": args.family,
        "logical_ks": [2] if args.family == "pk2" else [3, 4],
        "physical_k": 2 if args.family == "pk2" else 4,
        "baseline_layout": "row_major (not best-of-row/column)",
        "scope": (
            "Opt-in policy screen, not pure-kernel or global certification"
        ),
        "oracle": (
            "Host sequential direct-FP32 metric with stable index tie break; "
            "same 16 sampled rows in each process"
        ),
        "nonself_recall_definition": (
            "Remove self from oracle top-k using each record's logical_k; "
            "do not refill. Processes reuse the same seeded quality rows."
        ),
        "recall_floor": None,
    }
    family_name = "Pk2" if args.family == "pk2" else "Pk4"
    prefix = (
        f"Representative/FusedTopk{family_name}Benchmark.DISABLED_PublicOptIn/"
    )
    if args.smoke:
        selection = prefix + (
            "Gaussian_D32_L2" if args.family == "pk2" else "Gaussian_D128_L2"
        )
        if args.family == "pk4":
            selection += "_K3"
        expected = 1
    else:
        selection = prefix + "*"
        expected = (
            16
            if args.family == "pk4"
            else (13 if args.include_near_tie else 12)
        )
        if not args.include_near_tie:
            selection += "-" + prefix + "NearTie*"
    command = [
        str(binary),
        "--gtest_also_run_disabled_tests",
        "--gtest_filter=" + selection,
    ]
    metadata["command"] = command
    # Never overwrite an earlier evidence packet.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    records = []
    for process in range(3):
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            partial = error.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            (args.output / f"process_{process}.log").write_text(partial)
            failure = {
                "process": process,
                "reason": "timeout",
                "timeout_seconds": 600,
                "completed_records": len(records),
                "command": command,
            }
            (args.output / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            raise RuntimeError(
                f"Process {process} timed out; partial log retained"
            ) from error
        (args.output / f"process_{process}.log").write_text(result.stdout)
        current = []
        for line in result.stdout.splitlines():
            if line.startswith("TOPK_BENCH_JSON "):
                row = json.loads(line.removeprefix("TOPK_BENCH_JSON "))
                row["process"] = process
                current.append(row)
        records.extend(current)
        (args.output / "samples.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        if result.returncode or len(current) != expected:
            failure = {
                "process": process,
                "reason": "test_failure_or_missing_records",
                "returncode": result.returncode,
                "records": len(current),
                "expected_records": expected,
                "command": command,
            }
            (args.output / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            raise RuntimeError(
                f"Process {process}: exit={result.returncode}, "
                f"records={len(current)}, expected={expected}; inspect its log"
            )
    summary = []
    for name in sorted({row["case"] for row in records}):
        cell = [row for row in records if row["case"] == name]
        base = [statistics.median(row["baseline_ms"]) for row in cell]
        candidate = [statistics.median(row["opt_in_ms"]) for row in cell]
        speedups = [a / b for a, b in zip(base, candidate)]
        summary.append(
            {
                "case": name,
                "baseline_process_medians_ms": base,
                "opt_in_process_medians_ms": candidate,
                "paired_process_speedups": speedups,
                "every_process_1_05x_gate": min(speedups) >= 1.05,
                "baseline_process_spread_fraction": (
                    (max(base) - min(base)) / statistics.median(base)
                ),
                "opt_in_process_spread_fraction": (
                    (max(candidate) - min(candidate))
                    / statistics.median(candidate)
                ),
                "min_sampled_direct_fp32_recall": min(
                    row["direct_fp32_recall"] for row in cell
                ),
                "min_sampled_nonself_recall": min(
                    row["self_excluded_direct_fp32_recall"] for row in cell
                ),
                "min_sampled_cuvs_overlap": min(
                    row["cuvs_membership_overlap"] for row in cell
                ),
                "promotion_decision": "not_automatic_requires_review",
            }
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "cells": len(summary),
                "processes": 3,
                "decision": "screen_only",
            }
        )
    )


if __name__ == "__main__":
    main()
