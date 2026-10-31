import pytest

from jersey_detect.synthetic import make_dataset


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("synthetic"))
