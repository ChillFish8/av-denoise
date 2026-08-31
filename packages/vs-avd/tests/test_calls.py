import pytest
from vsavd import Nlm, NlmHQ, Nl4d


class FakePlugin:
    """Records the call the wrapper makes, standing in for the real filter."""

    def __init__(self):
        self.calls = []

    def __call__(self, clip, **kwargs):
        self.calls.append((clip, kwargs))
        return clip


@pytest.fixture
def fake():
    return FakePlugin()


def test_nlm_selects_the_fast_variant(fake, monkeypatch):
    monkeypatch.setattr("vsavd._nlmeans_filter", lambda: fake)
    Nlm("clip")
    _, kwargs = fake.calls[0]
    assert kwargs["variant"] == "fast"


def test_nlmhq_selects_the_hq_variant(fake, monkeypatch):
    monkeypatch.setattr("vsavd._nlmeans_filter", lambda: fake)
    NlmHQ("clip")
    _, kwargs = fake.calls[0]
    assert kwargs["variant"] == "hq"


def test_no_defaults_are_injected(fake, monkeypatch):
    """Unset parameters must not reach the plugin at all.

    Core's preset tables are the single source of truth for defaults.
    A default declared here would be a second place for them to live.
    """
    monkeypatch.setattr("vsavd._nlmeans_filter", lambda: fake)
    Nlm("clip")
    _, kwargs = fake.calls[0]
    assert set(kwargs) == {"variant"}, f"unexpected arguments forwarded: {kwargs}"


def test_nl4d_forwards_no_arguments_when_none_are_set(fake, monkeypatch):
    monkeypatch.setattr("vsavd._nl4d_filter", lambda: fake)
    Nl4d("clip")
    _, kwargs = fake.calls[0]
    assert kwargs == {}, f"unexpected arguments forwarded: {kwargs}"


def test_set_parameters_are_forwarded(fake, monkeypatch):
    monkeypatch.setattr("vsavd._nl4d_filter", lambda: fake)
    Nl4d("clip", preset="slow", lambda_ht_scale=1.1)
    _, kwargs = fake.calls[0]
    assert kwargs == {"preset": "slow", "lambda_ht_scale": 1.1}


def test_kwargs_pass_through_verbatim(fake, monkeypatch):
    monkeypatch.setattr("vsavd._nl4d_filter", lambda: fake)
    Nl4d("clip", sigma=0.5, temporal_radius=3)
    _, kwargs = fake.calls[0]
    assert kwargs == {"sigma": 0.5, "temporal_radius": 3}


def test_sigma_is_not_a_typed_parameter():
    """sigma is reachable through kwargs but must not be in any signature.

    The README lists it under what not to touch, because pinning the
    noise level disables the per-scene measurement.
    """
    import inspect

    for fn in (Nlm, NlmHQ, Nl4d):
        assert "sigma" not in inspect.signature(fn).parameters, fn.__name__
