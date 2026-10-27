from __future__ import annotations

from typing import Literal

Preset = Literal["veryfast", "fast", "base", "slow", "veryslow"]
ChannelMode = Literal["luma", "chroma", "lumachroma", "yuv"]
Tune = Literal["auto", "cached", "off"]
Accelerators = Literal["cuda", "vulkan", "rocm", "metal"]
