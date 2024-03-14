# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--randomize_julia_scripts",
        action="store_true",
        default=False,
        help="randomize Julia script tests to make them faster",
    )


@pytest.fixture
def randomize_julia_scripts(request):
    return request.config.getoption("--randomize_julia_scripts")
