from __future__ import annotations

import pathlib
from collections.abc import MutableMapping


def bundled_cuda_root() -> pathlib.Path:
    """Returns the directory holding the libdevice file shipped in the wheel."""
    return pathlib.Path(__file__).resolve().parent / "cuda"


def apply_bundled_cuda_path(environ: MutableMapping[str, str], root: pathlib.Path) -> None:
    """Points `CUDA_PATH` at `root` when it is unset and `root` holds libdevice."""
    if "CUDA_PATH" in environ:
        return

    libdevice = root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if libdevice.is_file():
        environ["CUDA_PATH"] = str(root)
