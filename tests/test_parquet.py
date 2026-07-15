# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Parquet round-trip tests for DFSFileSystem.

Requires: pip install pandas pyarrow
These tests are skipped if pandas/pyarrow are not installed.
"""

import pytest

pandas = pytest.importorskip("pandas")
pyarrow = pytest.importorskip("pyarrow")


@pytest.mark.live
class TestParquetRoundTrip:
    """Read/write Parquet files through DFS fsspec backend."""

    def test_basic_parquet(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/test.parquet"

        # Create a DataFrame
        df = pandas.DataFrame(
            {
                "id": range(100),
                "name": [f"item-{i}" for i in range(100)],
                "value": [i * 1.5 for i in range(100)],
            }
        )

        # Write through DFS
        with dfs_fs.open(path, "wb") as f:
            df.to_parquet(f)

        # Read back through DFS
        with dfs_fs.open(path, "rb") as f:
            df2 = pandas.read_parquet(f)

        pandas.testing.assert_frame_equal(df, df2)

    def test_large_parquet(self, dfs_fs):
        """10K rows with multiple columns."""
        d = dfs_fs._test_dir
        path = f"{d}/large.parquet"

        df = pandas.DataFrame(
            {
                "a": range(10000),
                "b": [f"string-{i}" for i in range(10000)],
                "c": [i * 0.1 for i in range(10000)],
                "d": [i % 2 == 0 for i in range(10000)],
            }
        )

        with dfs_fs.open(path, "wb") as f:
            df.to_parquet(f)

        with dfs_fs.open(path, "rb") as f:
            df2 = pandas.read_parquet(f)

        pandas.testing.assert_frame_equal(df, df2)

    def test_parquet_with_cat(self, dfs_fs):
        """Write parquet, read back with cat() and parse."""
        import io

        d = dfs_fs._test_dir
        path = f"{d}/cat-parquet.parquet"

        df = pandas.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

        with dfs_fs.open(path, "wb") as f:
            df.to_parquet(f)

        raw = dfs_fs.cat(path)
        df2 = pandas.read_parquet(io.BytesIO(raw))
        pandas.testing.assert_frame_equal(df, df2)
