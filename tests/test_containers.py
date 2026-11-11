# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for containers."""

import pytest


@pytest.mark.live
class TestContainerManagement:
    """Container create, list, destroy via bindings."""

    def test_list_containers(self, daos_pool):
        from daos_fsspec import _bindings as B

        B.daos_init()
        poh = B.pool_connect(daos_pool)
        try:
            containers = B.pool_list_cont(poh)
            assert isinstance(containers, list)
            # At least one container should exist
            assert len(containers) > 0
        finally:
            B.pool_disconnect(poh)

    def test_create_and_destroy(self, daos_pool):
        from daos_fsspec import _bindings as B

        B.daos_init()
        poh = B.pool_connect(daos_pool)
        test_label = "fsspec-ephemeral-test"

        try:
            # Create
            uuid_str = B.cont_create(poh, test_label)
            assert uuid_str  # Should be a UUID string
            assert len(uuid_str) == 36  # UUID format

            # Verify it appears in listing
            containers = B.pool_list_cont(poh)
            labels = [c.label for c in containers]
            assert test_label in labels

            # Destroy
            B.cont_destroy(poh, test_label, force=True)

            # Verify it's gone
            containers = B.pool_list_cont(poh)
            labels = [c.label for c in containers]
            assert test_label not in labels

        except Exception:
            # Clean up on failure
            try:
                B.cont_destroy(poh, test_label, force=True)
            except Exception:
                pass
            raise
        finally:
            B.pool_disconnect(poh)

    def test_create_duplicate_raises(self, daos_pool):
        from daos_fsspec import _bindings as B

        B.daos_init()
        poh = B.pool_connect(daos_pool)
        test_label = "fsspec-dup-test"

        try:
            B.cont_create(poh, test_label)
            with pytest.raises(FileExistsError):
                B.cont_create(poh, test_label)
        finally:
            try:
                B.cont_destroy(poh, test_label, force=True)
            except Exception:
                pass
            B.pool_disconnect(poh)

    def test_destroy_nonexistent_raises(self, daos_pool):
        from daos_fsspec import _bindings as B

        B.daos_init()
        poh = B.pool_connect(daos_pool)
        try:
            with pytest.raises(Exception):
                B.cont_destroy(poh, "fsspec-nonexistent-container-999", force=True)
        finally:
            B.pool_disconnect(poh)
