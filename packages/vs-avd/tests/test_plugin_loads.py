def test_artefact_suffixes_cover_this_platforms_extension_suffix():
    """`plugin_path` must recognise whatever setuptools-rust names the artefact.

    setuptools-rust uses the platform's Python extension suffix, so the file is
    `.pyd` on Windows and `.abi3.so` elsewhere. Missing `.pyd` made every Windows
    wheel install cleanly and then fail to find its own plugin.
    """
    import pathlib
    from importlib.machinery import EXTENSION_SUFFIXES

    from vsavd import _plugin

    platform_suffixes = {pathlib.Path("artefact" + s).suffix for s in EXTENSION_SUFFIXES}
    assert platform_suffixes <= set(_plugin._ARTEFACT_SUFFIXES)


def test_plugin_path_points_at_a_real_file():
    from vsavd import _plugin

    path = _plugin.plugin_path()
    assert path.is_file(), f"bundled plugin missing at {path}"


def test_ensure_loaded_registers_the_avd_namespace():
    import vapoursynth as vs
    from vsavd import _plugin

    core = vs.core
    _plugin.ensure_loaded(core)
    assert "avd" in [p.namespace for p in core.plugins()]


def test_ensure_loaded_is_idempotent():
    import vapoursynth as vs
    from vsavd import _plugin

    core = vs.core
    _plugin.ensure_loaded(core)
    _plugin.ensure_loaded(core)  # must not raise


def test_the_bundled_plugin_passes_verification(tmp_path):
    from vsavd import _plugin

    bundled = tmp_path / "bundled.so"
    bundled.write_bytes(b"plugin v2")

    _plugin.verify_registered(bundled, bundled)


def test_a_byte_identical_copy_passes_verification(tmp_path):
    from vsavd import _plugin

    bundled = tmp_path / "bundled.so"
    bundled.write_bytes(b"plugin v2")
    copy = tmp_path / "plugins" / "copy.so"
    copy.parent.mkdir()
    copy.write_bytes(b"plugin v2")

    _plugin.verify_registered(copy, bundled)


def test_a_stale_plugin_fails_verification_naming_both_files(tmp_path):
    import pytest
    from vsavd import _plugin

    bundled = tmp_path / "bundled.so"
    bundled.write_bytes(b"plugin v2")
    stale = tmp_path / "plugins" / "stale.so"
    stale.parent.mkdir()
    stale.write_bytes(b"plugin v1")

    with pytest.raises(RuntimeError) as raised:
        _plugin.verify_registered(stale, bundled)

    message = str(raised.value)
    assert str(stale) in message
    assert str(bundled) in message


def test_a_stale_plugin_of_the_same_size_fails_verification(tmp_path):
    import pytest
    from vsavd import _plugin

    bundled = tmp_path / "bundled.so"
    bundled.write_bytes(b"plugin v2")
    stale = tmp_path / "stale.so"
    stale.write_bytes(b"plugin v1")

    with pytest.raises(RuntimeError):
        _plugin.verify_registered(stale, bundled)


def test_ensure_loaded_rejects_a_stale_autoloaded_plugin(tmp_path):
    import types

    import pytest
    from vsavd import _plugin

    stale = tmp_path / "av_denoise_vs.dll"
    stale.write_bytes(b"plugin v1")
    stale_plugin = types.SimpleNamespace(namespace="avd", plugin_path=str(stale))
    core = types.SimpleNamespace(plugins=lambda: iter([stale_plugin]))

    with pytest.raises(RuntimeError) as raised:
        _plugin.ensure_loaded(core)

    message = str(raised.value)
    assert str(stale) in message
    assert str(_plugin.plugin_path()) in message


def test_ensure_loaded_accepts_the_plugin_it_registered():
    import vapoursynth as vs
    from vsavd import _plugin

    core = vs.core
    _plugin.ensure_loaded(core)

    registered = [p.plugin_path for p in core.plugins() if p.namespace == "avd"]
    _plugin.verify_registered(registered[0], _plugin.plugin_path())
