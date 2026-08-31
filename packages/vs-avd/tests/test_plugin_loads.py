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
