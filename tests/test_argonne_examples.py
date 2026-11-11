# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for argonne examples."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples" / "argonne"


REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ex01_hello_fsspec": (),
    "ex02_open_streaming": (),
    "ex03_pandas_url": ("pandas",),
    "ex04_numpy_samples": ("numpy",),
    "ex05_torch_dataloader": ("numpy", "torch"),
    "ex06_torch_checkpoint": ("torch",),
    "ex07_safetensors": ("safetensors", "torch"),
    "ex08_hf_datasets": ("datasets",),
    "ex09_h5py": ("h5py", "numpy"),
    "ex10_pillow_images": ("PIL", "numpy"),
    "ex11_sklearn_joblib": ("joblib", "numpy", "sklearn"),
    "ex12_arrow_parquet": ("numpy", "pyarrow"),
    "ex13_chunked_arrays": ("numpy",),
    # mpi4py is imported only under an MPI launcher or a PBS allocation, so
    # this runs standalone as a single rank. See the example's docstring.
    "ex14_mpi_ranks": ("numpy",),
    "ex15_streaming_large": ("numpy",),
    "ex16_scipy_sparse": ("numpy", "scipy"),
    "ex17_matplotlib": ("PIL", "matplotlib", "numpy"),
    "ex18_config_formats": ("msgpack", "yaml"),
    "ex19_archives_pickle": ("cloudpickle", "numpy"),
    "ex20_tokenizers_transformers": ("tokenizers", "torch", "transformers"),
    "ex21_sqlite_stage_local": (),
}


RECORDED_OUTPUT: dict[str, tuple[str, ...]] = {
    "ex09_h5py": ("9559", "(100, 10)", "equal: True"),
    "ex18_config_formats": ("2103", "4127"),
}


KNOWN_BROKEN = {"ex20_tokenizers_transformers": "ex20 SIGSEGV, blocked on an ALCF backtrace"}

_EXAMPLE_TIMEOUT = 600.0


def _missing(packages: tuple[str, ...]) -> list[str]:
    return [p for p in packages if importlib.util.find_spec(p) is None]


def _squashed(text: str) -> str:
    """Drop all whitespace, so column alignment is not part of the assertion."""
    return "".join(text.split())


def test_every_example_file_is_accounted_for():
    """REQUIREMENTS must cover the examples on disk, exactly.

    Without this, adding an example silently leaves it unrun, and deleting one
    leaves a stale entry claiming coverage that does not exist; which is the
    failure mode this whole module is a response to.
    """
    on_disk = {p.stem for p in EXAMPLES.glob("ex*.py")}
    declared = set(REQUIREMENTS)

    assert on_disk == declared, (
        f"examples not declared here: {sorted(on_disk - declared)}; "
        f"declared but missing from disk: {sorted(declared - on_disk)}"
    )


@pytest.mark.live
@pytest.mark.parametrize("name", sorted(REQUIREMENTS))
def test_example_runs_against_real_daos(name, daos_pool, daos_container, dfs_fs):
    """Run one example as a user would, and require a clean exit.

    Also checks it removed its working directory. That assertion lives here,
    per example, rather than as one sweep at the end of the session: test order
    is randomised, so a session-level check could run before any example and
    pass while proving nothing.
    """
    if name in KNOWN_BROKEN:
        pytest.skip(KNOWN_BROKEN[name])

    absent = _missing(REQUIREMENTS[name])
    if absent:
        pytest.skip(f"{name} needs {', '.join(absent)}, not installed in this image")

    script = EXAMPLES / f"{name}.py"
    assert script.exists(), f"{script} is missing"

    env = dict(os.environ)
    env["DAOS_POOL"] = daos_pool
    env["DAOS_CONT"] = daos_container

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=_EXAMPLE_TIMEOUT,
        env=env,
        cwd=str(EXAMPLES.parent.parent),
    )

    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    squashed = _squashed(result.stdout)
    for expected in RECORDED_OUTPUT.get(name, ()):
        assert _squashed(expected) in squashed, (
            f"{name} no longer reproduces the figure {expected!r} recorded in "
            f"docs/validation.md; behaviour changed, not just "
            f"formatting\n--- stdout ---\n{result.stdout}"
        )

    # Every example works in /exNN_* and removes it again; ex14 shares one
    # collective directory, which rank 0 cleans up. A leftover means the
    # cleanup path is broken, which matters to users running these against a
    # real shared container and to this module re-running them.
    prefix = name[:4]
    leftovers = [
        entry
        for entry in dfs_fs.ls("/", detail=False)
        if pathlib.PurePosixPath(entry).name.startswith(prefix)
    ]
    assert not leftovers, f"{name} left {sorted(leftovers)} behind in the container"
