from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # vapoursynth ships a vapoursynth.pyi stub but no py.typed marker, so mypy
    # refuses to read it. The `ignore_missing_imports` override in pyproject.toml
    # covers both that and the case where the package is absent entirely.
    import vapoursynth as vs

_NAMESPACE = "avd"
_ARTEFACT_GLOB = "_avdenoise_vs*"
# setuptools-rust names the artefact with the platform's Python extension suffix,
# which on Windows is `.pyd` rather than `.dll`. `.dll` and `.dylib` stay for a
# plugin built by hand and copied into a source checkout.
_ARTEFACT_SUFFIXES = (".so", ".dll", ".dylib", ".pyd")


def plugin_path() -> pathlib.Path:
    """
    The bundled plugin's absolute path.

    setuptools-rust names the compiled artefact after the ext-module target
    and picks its suffix based on whether the extension opted into the
    limited API, so this helper globs for it instead of matching one fixed name.
    """
    here = pathlib.Path(__file__).parent
    candidates = sorted(p for p in here.glob(_ARTEFACT_GLOB) if p.suffix in _ARTEFACT_SUFFIXES)
    if not candidates:
        raise RuntimeError(
            f"no bundled av-denoise plugin found in {here}. "
            "A source checkout needs the plugin built and copied in, which the wheel does at build time."
        )
    if len(candidates) > 1:
        joined = ", ".join(str(c) for c in candidates)
        raise RuntimeError(
            f"multiple candidate av-denoise plugin artefacts found in {here}: {joined}. "
            "Remove the stale build output and rebuild the wheel."
        )
    return candidates[0]


def ensure_loaded(core: "vs.Core") -> None:
    """Registers the plugin with `core` unless it is already there."""
    if any(p.namespace == _NAMESPACE for p in core.plugins()):
        return
    core.std.LoadPlugin(str(plugin_path()))
