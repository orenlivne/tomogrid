"""Direction grids on the sphere: nesting, frames, and refinement."""

import numpy as np
import pytest

from tomogrid.directions import N_FACES, DirectionGrid, grid_sequence


@pytest.mark.parametrize("m", [2, 3, 5, 9])
def test_vectors_are_unit(m):
    d = DirectionGrid(m).vectors()
    assert d.shape == (N_FACES * m * m, 3)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)


@pytest.mark.parametrize("m", [2, 3, 5, 9])
def test_frames_are_right_handed_orthonormal(m):
    e1, e2, d = DirectionGrid(m).frames()
    for a in (e1, e2, d):
        assert np.allclose(np.linalg.norm(a, axis=-1), 1.0)
    assert np.allclose(np.sum(e1 * e2, axis=-1), 0.0, atol=1e-12)
    assert np.allclose(np.sum(e1 * d, axis=-1), 0.0, atol=1e-12)
    assert np.allclose(np.sum(e2 * d, axis=-1), 0.0, atol=1e-12)
    assert np.allclose(np.cross(e1, e2), d, atol=1e-12)


@pytest.mark.parametrize("m", [2, 3, 5])
def test_refinement_nests_exactly(m):
    coarse = DirectionGrid(m)
    fine = coarse.refine()
    assert fine.m == 2 * m - 1
    keep = fine.nested_parent_slice()
    assert keep.sum() == coarse.n_dir
    assert np.allclose(fine.vectors()[keep], coarse.vectors())


def test_frames_are_consistent_under_refinement():
    """The frame must be a function of the direction alone, or the copied
    tables of the nested directions would be indexed differently."""
    coarse = DirectionGrid(3)
    fine = coarse.refine()
    keep = fine.nested_parent_slice()
    for a, b in zip(coarse.frames(), (x[keep] for x in fine.frames())):
        assert np.allclose(a, b)


def test_directions_cover_the_sphere():
    """Every axis direction is hit, and no two grid points coincide except the
    shared face edges."""
    d = DirectionGrid(3).vectors()
    for axis in np.eye(3):
        for sign in (1, -1):
            assert np.isclose(np.abs(d @ (sign * axis)).max(), 1.0)
    # A random direction is never far from some grid direction.
    rng = np.random.default_rng(0)
    probe = rng.normal(size=(200, 3))
    probe /= np.linalg.norm(probe, axis=-1, keepdims=True)
    worst = np.arccos(np.clip((probe @ d.T).max(axis=1), -1, 1)).max()
    assert worst < DirectionGrid(3).max_angular_step()


def test_angular_step_halves_under_refinement():
    """The gnomonic map is not uniform, so the ratio reaches 2 only in the
    limit; what matters is that it approaches 2 from below and never exceeds it."""
    ratios = []
    for m in (3, 5, 9, 17):
        coarse = DirectionGrid(m)
        ratios.append(coarse.max_angular_step() / coarse.refine().max_angular_step())
    assert all(1.5 < r <= 2.0 for r in ratios), ratios
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] > 1.98, ratios


def test_grid_sequence_quadruples_the_direction_count():
    """(2m-1)^2 / m^2 -> 4, again from below."""
    counts = [g.n_dir for g in grid_sequence(3, 4)]
    assert counts == [54, 150, 486, 1734, 6534]
    ratios = [b / a for a, b in zip(counts, counts[1:])]
    assert all(2.7 < r < 4.0 for r in ratios), ratios
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] > 3.75, ratios


def test_nested_parent_slice_rejects_unrefined_grid():
    with pytest.raises(ValueError):
        DirectionGrid(4).nested_parent_slice()
