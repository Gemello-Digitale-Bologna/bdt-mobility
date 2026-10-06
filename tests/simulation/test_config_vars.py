import pytest

from bdt_mobility.model.config_vars import config_params

pytestmark = pytest.mark.unit


def test_ethics_disabled_by_default():
    assert config_params["enable_ethics"] is False
