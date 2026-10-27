from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _tuning_off():
    """Turns tuning off, so no test pays for a tuning pass or depends on stored results."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AV_DENOISE_TUNE", "off")
        yield
