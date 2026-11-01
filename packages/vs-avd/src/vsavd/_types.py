from __future__ import annotations

from typing import Literal

Preset = Literal["veryfast", "fast", "base", "slow", "veryslow"]
Nl4dPreset = Literal["base", "slow"]
ChannelMode = Literal["luma", "chroma", "lumachroma"]
Tune = Literal["auto", "cached", "off"]
Accelerators = Literal["cuda", "vulkan", "rocm", "metal"]
