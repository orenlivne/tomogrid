import numpy as np
import pytest

from tomogrid.axis import Axis


def test_nodes_and_endpoints():
    a = Axis(origin=-1.0, h=0.5, n=5)
    assert np.allclose(a.nodes, [-1.0, -0.5, 0.0, 0.5, 1.0])
    assert a.lo == -1.0 and a.hi == 1.0


def test_from_endpoints_hits_both_ends():
    a = Axis.from_endpoints(-1.0, 2.0, 7)
    assert np.isclose(a.nodes[0], -1.0)
    assert np.isclose(a.nodes[-1], 2.0)
    assert len(a.nodes) == 7


def test_cell_centers_partition():
    a = Axis.cell_centers(0.0, 1.0, 4)
    assert np.allclose(a.nodes, [0.125, 0.375, 0.625, 0.875])


@pytest.mark.parametrize("kwargs", [dict(origin=0.0, h=1.0, n=0), dict(origin=0.0, h=0.0, n=3)])
def test_rejects_degenerate(kwargs):
    with pytest.raises(ValueError):
        Axis(**kwargs)


def test_from_endpoints_needs_two_nodes():
    with pytest.raises(ValueError):
        Axis.from_endpoints(0.0, 1.0, 1)
