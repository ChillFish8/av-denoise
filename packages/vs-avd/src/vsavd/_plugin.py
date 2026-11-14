from __future__ import annotations

import hashlib
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
    """
    Registers the plugin with `core` unless it is already there.

    A plugin VapourSynth autoloaded under the same namespace must match the
    bundled one, so a stale copy in a plugins folder cannot run in its place.
    """
    registered = [p for p in core.plugins() if p.namespace == _NAMESPACE]
    if not registered:
        core.std.LoadPlugin(str(plugin_path()))
        return

    registered_path = getattr(registered[0], "plugin_path", None)
    if registered_path is None:
        return

    verify_registered(registered_path, plugin_path())


def verify_registered(registered: str | pathlib.Path, bundled: pathlib.Path) -> None:
    """Raises if the plugin registered under the namespace differs from the bundled one."""
    registered = pathlib.Path(registered)
    if registered.resolve() == bundled.resolve():
        return

    same_size = registered.stat().st_size == bundled.stat().st_size
    if same_size and _file_digest(registered) == _file_digest(bundled):
        return

    raise RuntimeError(
        f"a different av-denoise plugin is already loaded from {registered}, "
        f"so the one bundled with vsavd at {bundled} cannot load. "
        "Delete the old plugin from your VapourSynth plugins folder."
    )


def _file_digest(path: pathlib.Path) -> str:
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256")
    return digest.hexdigest()
