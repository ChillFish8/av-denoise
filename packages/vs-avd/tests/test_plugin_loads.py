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
