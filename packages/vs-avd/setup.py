"""Builds the bundled av-denoise-vs cdylib and tags the wheel as pure-py3.

The cargo feature set that backs the compiled accelerator support differs
by platform, so it is chosen here rather than in pyproject.toml, which has
no way to express a platform-conditional list. `rocm` is never selected for
a wheel build, because it hard-links `libamdhip64.so.7` and related
libraries, so a wheel containing it fails to load on any machine without
ROCm installed.

The wheel produced by an unmodified `setuptools-rust` build is tagged per
CPython interpreter (`cp313-cp313-linux_x86_64`), even though the artefact
has no CPython ABI coupling. Combined with `py-limited-api = true` on the
extension below, overriding `bdist_wheel.get_tag()` here retags the wheel
as `py3-none-<platform>`, so one wheel serves every interpreter on a given
platform.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from setuptools import setup
from setuptools_rust import Binding, RustExtension

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from _version import package_version

if sys.platform == "darwin":
    _FEATURES = ["metal"]
else:
    _FEATURES = ["vulkan", "cuda"]


def _read_long_description() -> str:
    readme_path = (
        pathlib.Path(__file__).resolve().parents[2] / "av-denoise-vs" / "README.md"
    )
    if not readme_path.is_file():
        raise FileNotFoundError(
            f"expected the shared README at {readme_path}, but it does not exist"
        )
    return readme_path.read_text(encoding="utf-8")


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


setup(
    version=package_version(),
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    rust_extensions=[
        RustExtension(
            target="vsavd._avdenoise_vs",
            path="../../av-denoise-vs/Cargo.toml",
            binding=Binding.NoBinding,
            py_limited_api=True,
            features=_FEATURES,
            args=["--no-default-features"],
        ),
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
)
