from __future__ import annotations

import sys

from vsavd._cuda import apply_bundled_cuda_path, bundled_cuda_root


def _bundle(tmp_path):
    libdevice = tmp_path / "nvvm" / "libdevice" / "libdevice.10.bc"
    libdevice.parent.mkdir(parents=True)
    libdevice.write_bytes(b"bitcode")
    return tmp_path


def test_unset_cuda_path_points_at_the_bundle(tmp_path):
    environ = {}
    apply_bundled_cuda_path(environ, _bundle(tmp_path))
    assert environ["CUDA_PATH"] == str(tmp_path)


def test_existing_cuda_path_is_kept(tmp_path):
    environ = {"CUDA_PATH": "/opt/my-toolkit"}
    apply_bundled_cuda_path(environ, _bundle(tmp_path))
    assert environ["CUDA_PATH"] == "/opt/my-toolkit"


def test_missing_bundle_leaves_cuda_path_unset(tmp_path):
    environ = {}
    apply_bundled_cuda_path(environ, tmp_path)
    assert "CUDA_PATH" not in environ


def test_wheel_bundles_libdevice_except_on_macos():
    libdevice = bundled_cuda_root() / "nvvm" / "libdevice" / "libdevice.10.bc"
    assert libdevice.exists() == (sys.platform != "darwin")
