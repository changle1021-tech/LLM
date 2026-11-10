"""Unit tests for the congestion-aware (1-D) analytical backend wiring in
``serving/core/config_builder.py``.

The CA backend can only express a 1-D topology, so a leading size-1 dimension
(e.g. ``tp_size=1`` in a ``[1, dp_group_size]`` topology) must be collapsed, and
the matching leading ``involved_dim`` entries and per-dimension bandwidth/latency
list entries must be dropped with it. These tests pin that collapse behaviour and
guard the default (congestion-unaware) path against regression.
"""

import json

import pytest
import yaml

from serving.core.config_builder import (
    _collapse_leading_ones,
    _compute_network_dims,
    _create_network_config,
    _resolve_dp_groups,
    _sync_system_collective_dims,
)

_COLLECTIVE_KEYS = (
    "all-reduce-implementation",
    "all-gather-implementation",
    "reduce-scatter-implementation",
    "all-to-all-implementation",
)


def dp_instances(tp, ep, n, pp=1, is_moe=True):
    """Build ``n`` DP-group members with the given TP/EP/PP degrees."""
    return [
        {
            "tp_size": tp,
            "pp_size": pp,
            "ep_size": ep,
            "dp_group": "g0",
            "is_moe": is_moe,
            "num_npus": tp * pp,
            "pd_type": None,
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# _collapse_leading_ones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dims,expected",
    [
        ([1, 4], 1),
        ([1], 1),
        ([4], 0),
        ([1, 1, 4], 2),
        ([2, 4], 0),
        ([], 0),
    ],
)
def test_collapse_leading_ones(dims, expected):
    assert _collapse_leading_ones(dims) == expected


# ---------------------------------------------------------------------------
# _compute_network_dims
# ---------------------------------------------------------------------------
def test_compute_network_dims_unaware_keeps_leading_one():
    assert _compute_network_dims(dp_instances(1, 4, 4)) == [1, 4]


def test_compute_network_dims_ca_collapses_leading_one():
    assert _compute_network_dims(dp_instances(1, 4, 4), collapse_1d=True) == [4]


def test_compute_network_dims_ca_no_leading_one_is_noop():
    # tp=2 means the leading dim is 2, so collapse must not change anything.
    assert _compute_network_dims(dp_instances(2, 4, 4)) == [2, 4]
    assert _compute_network_dims(dp_instances(2, 4, 4), collapse_1d=True) == [2, 4]


def test_compute_network_dims_pp_dim_is_not_collapsed():
    # [tp=2, pp=2, dp=4] has no leading 1; collapse is a no-op.
    assert _compute_network_dims(dp_instances(2, 4, 4, pp=2)) == [2, 2, 4]
    assert _compute_network_dims(dp_instances(2, 4, 4, pp=2), collapse_1d=True) == [2, 2, 4]


# ---------------------------------------------------------------------------
# _resolve_dp_groups
# ---------------------------------------------------------------------------
def test_resolve_dp_groups_unaware_dims():
    insts = dp_instances(1, 4, 4)
    _resolve_dp_groups(insts)
    for m in insts:
        assert m["dp_group_size"] == 4
        assert m["ep_total"] == 4
        assert m["local_ep"] == 1
        assert m["tp_dim"] == [True, False]
        assert m["ep_dim"] == [False, True]


def test_resolve_dp_groups_ca_strips_leading_dim_entries():
    insts = dp_instances(1, 4, 4)
    _resolve_dp_groups(insts, collapse_1d=True)
    for m in insts:
        assert m["tp_dim"] == [False]
        assert m["ep_dim"] == [True]


def test_resolve_dp_groups_ca_no_leading_one_keeps_arity():
    insts = dp_instances(2, 4, 4)
    _resolve_dp_groups(insts, collapse_1d=True)
    for m in insts:
        # dims stay [2, 4]: nothing to strip, so the vectors keep 2 entries.
        assert m["tp_dim"] == [True, False]
        assert m["ep_dim"] == [True, True]


# ---------------------------------------------------------------------------
# _create_network_config
# ---------------------------------------------------------------------------
def _write_network(tmp_path, instances, **kwargs):
    path = str(tmp_path / "network.yml")
    _create_network_config(path, instances, kwargs.pop("link_bw", 24.0),
                           kwargs.pop("link_latency", 6600.0), **kwargs)
    with open(path) as f:
        return yaml.safe_load(f)


def test_create_network_config_unaware_is_two_dimensional(tmp_path):
    cfg = _write_network(tmp_path, dp_instances(1, 4, 4))
    assert cfg["topology"] == ["FullyConnected", "FullyConnected"]
    assert cfg["npus_count"] == [1, 4]


def test_create_network_config_ca_switch_collapses_to_one_dim(tmp_path):
    cfg = _write_network(tmp_path, dp_instances(1, 4, 4),
                         collapse_1d=True, ca_topology="Switch")
    assert cfg["topology"] == ["Switch"]
    assert cfg["npus_count"] == [4]
    assert cfg["bandwidth"] == [24.0]
    assert cfg["latency"] == [6600.0]


@pytest.mark.parametrize("topology", ["Switch", "Ring", "FullyConnected"])
def test_create_network_config_ca_topology_selection(tmp_path, topology):
    cfg = _write_network(tmp_path, dp_instances(1, 4, 4),
                         collapse_1d=True, ca_topology=topology)
    assert cfg["topology"] == [topology]


def test_create_network_config_ca_collapses_bandwidth_and_latency_arity(tmp_path):
    # A 2-entry list must drop its leading (now-meaningless) entry after the
    # leading size-1 dimension is collapsed.
    cfg = _write_network(tmp_path, dp_instances(1, 4, 4),
                         link_bw=[24.0, 48.0], link_latency=[6600.0, 3300.0],
                         collapse_1d=True, ca_topology="Switch")
    assert cfg["bandwidth"] == [48.0]
    assert cfg["latency"] == [3300.0]


def test_create_network_config_ca_rejects_multidimensional_topology(tmp_path):
    with pytest.raises(ValueError, match="1-D topology only"):
        _write_network(tmp_path, dp_instances(2, 4, 4),
                       collapse_1d=True, ca_topology="Switch")


# ---------------------------------------------------------------------------
# _sync_system_collective_dims
# ---------------------------------------------------------------------------
def _sync(tmp_path, instances, collapse_1d=False):
    path = str(tmp_path / "system.json")
    with open(path, "w") as f:
        json.dump({}, f)
    _sync_system_collective_dims(path, instances, collapse_1d=collapse_1d)
    with open(path) as f:
        return json.load(f)


def test_sync_system_collective_dims_unaware_arity(tmp_path):
    cfg = _sync(tmp_path, dp_instances(1, 4, 4))
    for key in _COLLECTIVE_KEYS:
        assert cfg[key] == ["ring", "ring"]


def test_sync_system_collective_dims_ca_arity(tmp_path):
    cfg = _sync(tmp_path, dp_instances(1, 4, 4), collapse_1d=True)
    for key in _COLLECTIVE_KEYS:
        assert cfg[key] == ["ring"]


# ---------------------------------------------------------------------------
# End-to-end arity consistency: trace vectors, network.yml, and system.json
# must all agree after collapse.
# ---------------------------------------------------------------------------
def test_collapsed_arity_is_consistent():
    insts = dp_instances(1, 4, 4)
    _resolve_dp_groups(insts, collapse_1d=True)

    # The trace side: every non-None involved_dim vector has exactly one entry.
    for m in insts:
        for key in ("tp_dim", "ep_dim"):
            if m[key] is not None:
                assert len(m[key]) == 1

    # The network side and the system side agree with that single dimension.
    assert _compute_network_dims(insts, collapse_1d=True) == [4]
