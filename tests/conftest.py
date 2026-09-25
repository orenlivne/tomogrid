import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow accuracy sweeps"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running accuracy sweep")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="need --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def rel_err(got, ref):
    """Max absolute difference normalised by the peak of the reference."""
    ref = np.asarray(ref)
    scale = np.abs(ref).max()
    if scale == 0:
        return float(np.abs(np.asarray(got)).max())
    return float(np.abs(np.asarray(got) - ref).max() / scale)


def order_of(errors, refinements=2.0):
    """Observed convergence order between successive (halved-h) errors."""
    errors = np.asarray(errors, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(refinements)
