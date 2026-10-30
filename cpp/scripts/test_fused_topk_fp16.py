# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only PK2/PK4 typed export contract; no compiler or device execution."""

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    ScalarConstraint,
    ConstantConstraint,
)

CPP = Path(__file__).resolve().parents[1]
KERNEL = CPP / "src/neighbors/detail/cutile_topk"
sys.path.insert(0, str(KERNEL))
import export_fused_topk as exporter  # noqa: E402
import fused_topk_kernel as kernel  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "CPU contract test attempted compiler/export/device execution"
        )

    monkeypatch.setattr(exporter, "export_kernel", forbidden)
    monkeypatch.setattr(ct, "launch", forbidden)


def matrix_rows(matrix):
    spec = importlib.util.spec_from_file_location(
        "matrix_product", CPP / "cmake/modules/compute_matrix_product.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.iterate_matrix_product(matrix=matrix))


@pytest.mark.parametrize("pk", [2, 4])
def test_ten_fp32_architecture_leaves_and_two_sm120_half_leaves(pk):
    path = KERNEL / f"fused_topk_pk{pk}_cutile_matrix.json"
    current = matrix_rows(json.loads(path.read_text()))
    floats = [row for row in current if row["data_type"] == "float"]
    assert len(floats) == 10 and len(current) == 12
    # Portable contract: do not depend on unpublished research commit IDs.
    assert [
        tuple(
            row.get(key)
            for key in (
                "gpu_code",
                "matrix_layout",
                "tile_m",
                "tile_n",
                "tile_k",
                "occupancy",
            )
        )
        for row in floats
    ] == [
        ("sm_80", "relaxed", 64, 128, 32, None),
        ("sm_86", "relaxed", 64, 128, 32, None),
        ("sm_80", "strict", 64, 128, 32, 2),
        ("sm_86", "strict", 64, 128, 32, 2),
        ("sm_90", "relaxed", 128, 128, 64, None),
        ("sm_90", "strict", 64, 256, 32, None),
        ("sm_100", "relaxed", 128, 256, 16, None),
        ("sm_100", "strict", 128, 128, 32, None),
        ("sm_120", "relaxed", 64, 128, 32, 2),
        ("sm_120", "strict", 64, 128, 32, 2),
    ]
    assert all(
        row["physical_k"] == pk
        and row["index_type"] == "int32"
        and row["metric"] == "runtime"
        for row in current
    )
    half = [row for row in current if row["data_type"] == "half"]
    assert {r["matrix_layout"] for r in half} == {"strict", "relaxed"}
    for row in half:
        assert (
            row["gpu_code"],
            row["physical_k"],
            row["index_type"],
            row["tile_m"],
            row["tile_n"],
            row["tile_k"],
            row["occupancy"],
        ) == ("sm_120", pk, "int32", 64, 128, 32, 2)
    assert (
        len(
            {
                (r["data_type"], r["gpu_code"], r["matrix_layout"])
                for r in current
            }
        )
        == 12
    )


@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize("layout", ["strict", "relaxed"])
@pytest.mark.parametrize("arch", ["sm_80", "sm_120"])
@pytest.mark.parametrize("pk", [2, 4])
def test_actual_signature_full_constraints(dtype, layout, arch, pk):
    sig = exporter._kernel_signature(pk, 64, 128, 32, arch, layout, dtype)
    pitch = 8 if dtype == "half" else 4
    matrix = ArrayConstraint(
        ct.float16 if dtype == "half" else ct.float32,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=("read_only_inputs",),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(pitch, 1) if layout == "strict" else (1, 1),
        shape_divisible_by=(1, pitch)
        if layout == "strict" and arch == "sm_80"
        else (1, 1),
        base_addr_divisible_by=16,
    )

    def output(element):
        return ArrayConstraint(
            element,
            ndim=2,
            index_dtype=ct.int32,
            stride_lower_bound_incl=(0, None),
            alias_groups=(),
            may_alias_internally=False,
            stride_constant=(None, 1),
            stride_divisible_by=(1, 1),
            shape_divisible_by=(1, 1),
            base_addr_divisible_by=16,
        )

    assert sig.parameters == (
        matrix,
        matrix,
        output(ct.int32),
        output(ct.float32),
        *[ScalarConstraint(ct.int32)] * 5,
        *[ConstantConstraint(v) for v in (64, 128, 32, pk)],
    )
    assert (
        sig.calling_convention == exporter.CallingConvention.cutile_python_v1()
    )
    assert (
        sig.symbol
        == f"fused_topk_pk{pk}_{'h' if dtype == 'half' else 'f'}_i32"
        + ("_relaxed" if layout == "relaxed" else "")
    )
    # Four 2D int32 descriptors plus five runtime int32 scalars: actual ELF checked after build.
    assert (
        sum(
            5
            if isinstance(p, ArrayConstraint)
            else 1
            if isinstance(p, ScalarConstraint)
            else 0
            for p in sig.parameters
        )
        == 25
    )


@pytest.mark.parametrize("bad", ["double", "int8", "bfloat16"])
def test_reject_unsupported_input_type(bad):
    with pytest.raises(ValueError):
        exporter._kernel_signature(2, 64, 128, 32, "sm_120", "strict", bad)
    with pytest.raises(ValueError):
        kernel.make_kernel(bad, "runtime", 2, 64, 128, 32)
    with pytest.raises(ValueError):
        kernel.kernel_symbol(2, "strict", bad)


def test_producer_descriptor_width_is_not_public_index_width():
    with pytest.raises(ValueError):
        kernel.make_kernel(
            "half", "runtime", 2, 64, 128, 32, index_type="int64"
        )
    source = (KERNEL / "fused_topk_tile.cu").read_text()
    assert (
        "checked_tensor_bytes(query_rows, dimensions, sizeof(DataT))" in source
    )
    assert (
        "checked_tensor_bytes(corpus_rows, dimensions, sizeof(DataT))"
        in source
    )
    assert (
        "int shape_queries[2]" in source and "int stride_queries[2]" in source
    )


def assert_fp32_refinement(source):
    """Both retained-value operands must be widened before metric arithmetic."""
    loops = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.For):
            continue
        assigned = {
            target.id: statement.value
            for statement in node.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        if {"query_tile", "selected_values"} <= assigned.keys():
            loops.append(assigned)
    assert len(loops) == 1, "one retained-value refinement loop"
    for name in ("query_tile", "selected_values"):
        value = loops[0][name]
        assert isinstance(value, ast.Call), name
        assert (
            isinstance(value.func, ast.Attribute)
            and value.func.attr == "astype"
        ), name
        loaded = value.func.value
        assert isinstance(loaded, ast.Call) and isinstance(
            loaded.func, ast.Attribute
        ), name
        assert (
            isinstance(loaded.func.value, ast.Name)
            and loaded.func.value.id == "ct"
        ), name
        assert loaded.func.attr == (
            "load" if name == "query_tile" else "gather"
        ), name
        assert len(value.args) == 1 and ast.dump(value.args[0]) == ast.dump(
            ast.parse("ct.float32", mode="eval").body
        ), name


def test_fp32_widening_before_norm_and_refinement():
    source = (KERNEL / "fused_topk_kernel.py").read_text()
    assert (
        'mma_dtype = ct.float16 if data_type == "half" else ct.tfloat32'
        in source
    )
    assert "corpus_f32 = corpus_t.astype(ct.float32)" in source
    assert "corpus_f32 * corpus_f32" in source
    assert_fp32_refinement(source)
    assert "delta = query_tile[:, None, :] - selected_values" in source
    assert "accumulator = ct.zeros((tm, tn), ct.float32)" in source


@pytest.mark.parametrize("dtype", ["float", "half"])
def test_export_passes_actual_typed_signature_without_execution(
    monkeypatch, tmp_path, dtype
):
    captured = {}
    sentinel = object()
    monkeypatch.setattr(exporter, "make_kernel", lambda *a, **kw: sentinel)
    monkeypatch.setattr(
        exporter, "export_kernel", lambda **kw: captured.update(kw)
    )
    result = exporter.export_binary(
        tmp_path / "never-created.cubin",
        data_type=dtype,
        metric="runtime",
        index_type="int32",
        physical_k=2,
        tile_m=64,
        tile_n=128,
        tile_k=32,
        gpu_code="sm_120",
        matrix_layout="strict",
        occupancy=2,
    )
    assert captured["kernel"] is sentinel
    assert captured["signatures"] == [
        exporter._kernel_signature(2, 64, 128, 32, "sm_120", "strict", dtype)
    ]
    assert result == captured["signatures"][0].symbol
    assert not (tmp_path / "never-created.cubin").exists()


def test_family_predicates_and_half_cache_are_explicit():
    public = (CPP / "src/neighbors/detail/knn_brute_force.cuh").read_text()
    for pk in (2, 4):
        assert f"is_fused_topk_pk{pk}_candidate<value_t>" in public
    planner = (KERNEL / "fused_topk_pk4_planner.hpp").read_text()
    assert "template <typename DataT, typename AbiTag>" in planner
    assert "if constexpr (std::is_same_v<DataT, half>)" in planner
    assert "fused_topk_pk4_h_i32" in planner


def test_exact_stored_half_surplus_fixture():
    import numpy as np

    values = np.array([4, 3, 2, 1], dtype=np.float16) / np.float16(16)
    square = values.astype(np.float32) ** 2
    approximate = np.float32(1024**2) + square
    assert np.all(approximate == np.float32(1024**2))
    assert np.argsort(square).tolist() == [3, 2, 1, 0]
    assert np.array_equal(
        square[[3, 2, 1]], np.array([1, 4, 9], np.float32) / 256
    )
    y = np.float32(np.float16(3 / 8192))
    assert np.float32(1) + y * y > np.float32(1)
    assert np.sqrt(np.float32(1) + y * y, dtype=np.float32) == np.float32(1)
