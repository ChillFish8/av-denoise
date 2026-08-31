from __future__ import annotations

import random
from typing import get_args

import pytest
import vapoursynth as vs

from vsavd import Nl4d, Nlm, NlmHQ
from vsavd._types import ChannelMode, Preset

core = vs.core


def synthetic_clip(width=64, height=48, length=8, fmt=vs.YUV420P8, seed=7):
    """A deterministic noisy clip, scaled to the format's bit depth.

    Each frame is a spatial ramp offset by the frame number plus noise
    drawn from a generator seeded on `(seed, n)`, so a frame's content
    depends only on its own index and never on request order.
    """
    base = core.std.BlankClip(width=width, height=height, length=length, format=fmt)

    def draw(n, f):
        out = f.copy()
        rng = random.Random(f"{seed}:{n}")
        bits = out.format.bits_per_sample
        max_val = (1 << bits) - 1
        scale = max_val / 255
        for plane in range(out.format.num_planes):
            p = out[plane]
            h, w = p.shape
            for y in range(h):
                for x in range(w):
                    ramp = ((y + x) + n * 3) % 200
                    noise = rng.randint(-12, 12)
                    p[y, x] = max(0, min(max_val, int(round((ramp + noise) * scale))))
        return out

    return core.std.ModifyFrame(base, base, draw)


def _assert_shape_preserved(out, src):
    assert out.format.id == src.format.id
    assert out.num_frames == src.num_frames
    assert out.fps == src.fps
    for n in (0, 3, 7):
        frame = out.get_frame(n)
        assert frame.width == src.width
        assert frame.height == src.height


@pytest.mark.gpu
def test_nlm_renders_and_preserves_clip_properties():
    src = synthetic_clip()
    out = Nlm(src)
    _assert_shape_preserved(out, src)


@pytest.mark.gpu
def test_nlmhq_renders_and_preserves_clip_properties():
    src = synthetic_clip()
    out = NlmHQ(src)
    _assert_shape_preserved(out, src)


@pytest.mark.gpu
def test_nl4d_renders_and_preserves_clip_properties():
    src = synthetic_clip()
    out = Nl4d(src)
    _assert_shape_preserved(out, src)


@pytest.mark.gpu
def test_every_preset_literal_is_accepted():
    """_types.Preset duplicates strings the plugin owns. Prove they agree."""
    src = synthetic_clip()
    for preset in get_args(Preset):
        Nl4d(src, preset=preset).get_frame(0)


@pytest.mark.gpu
def test_every_channel_mode_literal_is_accepted():
    """_types.ChannelMode duplicates strings the plugin owns. Prove they agree.

    channel_mode="yuv" needs a 4:4:4 source (no subsampled chroma), so
    it gets its own clip here rather than being skipped.
    """
    src = synthetic_clip()
    src_444 = synthetic_clip(fmt=vs.YUV444P8)
    for mode in get_args(ChannelMode):
        clip = src_444 if mode == "yuv" else src
        Nl4d(clip, channel_mode=mode).get_frame(0)


@pytest.mark.gpu
def test_an_unknown_preset_is_rejected():
    """Without this, test_every_preset_literal_is_accepted would pass

    even if Preset's strings were wrong, as long as the plugin accepted
    anything at all.
    """
    src = synthetic_clip()
    with pytest.raises(Exception):
        Nl4d(src, preset="ludicrous").get_frame(0)


@pytest.mark.gpu
def test_an_unknown_channel_mode_is_rejected():
    """Same guard as the preset case, for ChannelMode."""
    src = synthetic_clip()
    with pytest.raises(Exception):
        Nl4d(src, channel_mode="ludicrous").get_frame(0)
