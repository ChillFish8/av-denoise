from __future__ import annotations

from typing import Literal

Preset = Literal["veryfast", "fast", "base", "slow", "veryslow"]
ChannelMode = Literal["luma", "chroma", "lumachroma", "yuv"]
Accelerators = Literal["cuda", "vulkan", "rocm", "metal"]