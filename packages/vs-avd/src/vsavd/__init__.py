from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _plugin
from ._types import Accelerators, ChannelMode, Preset

if TYPE_CHECKING:
    # vapoursynth ships a vapoursynth.pyi stub but no py.typed marker, so mypy
    # refuses to read it. The `ignore_missing_imports` override in pyproject.toml
    # covers both that and the case where the package is absent entirely.
    import vapoursynth as vs

__all__ = ["Nlm", "NlmHQ", "Nl4d", "Preset", "ChannelMode"]


def _nlmeans_filter() -> "vs.Function":
    """Returns the bound `avd.NLMeans` plugin function, loading the plugin first."""
    import vapoursynth as vs

    core = vs.core
    _plugin.ensure_loaded(core)
    return core.avd.NLMeans


def _nl4d_filter() -> "vs.Function":
    """Returns the bound `avd.NL4D` plugin function, loading the plugin first."""
    import vapoursynth as vs

    core = vs.core
    _plugin.ensure_loaded(core)
    return core.avd.NL4D


def _forward(fn: "vs.Function", clip: "vs.VideoNode", /, **params: Any) -> "vs.VideoNode":
    """Calls `fn` with only the parameters the caller actually set."""
    return fn(clip, **{k: v for k, v in params.items() if v is not None})


def Nlm(
    clip: "vs.VideoNode",
    *,
    preset: Preset | None = None,
    channel_mode: ChannelMode | None = None,
    device: str | None = None,
    strength: float | None = None,
    luma_strength: float | None = None,
    chroma_strength: float | None = None,
    motion_compensation: bool | None = None,
    accelerators: list[Accelerators] | None = None,
    **kwargs: Any,
) -> "vs.VideoNode":
    """
    Runs the NLMeans denoiser.

    `Nlm` is the regular NLMeans algorithm as originally defined in the paper.
    See https://github.com/ChillFish8/av-denoise/docs/CHOOSING_AN_ALGORITHM.md for working
    out what algorithm is right for you.

    Args:
        clip: The input video node to denoise.
        preset: Preset name. Unset dials fall back to whatever the av-denoise binary defaults to.
        channel_mode: Which planes to filter, one of `"luma"`, `"chroma"`, `"lumachroma"` or `"yuv"`.
        device: Compute device to run on. `"cpu"` selects a software device where the platform offers one,
            useful for testing, not for real encodes. Example: `"discrete:0"` for discrete GPU 0.
        strength: Overall filter strength. Mirrors FFmpeg's scaling.
        luma_strength: Strength override for the luma plane. Mirrors FFmpeg's scaling.
        chroma_strength: Strength override for the chroma planes. Mirrors FFmpeg's scaling.
        motion_compensation: Whether to motion-compensate neighbour frames before matching.
        accelerators: The accelerators to try and use in the order to attempt.
        **kwargs: Further parameters reachable by name. Not recommended

    Returns:
        A new `vs.VideoNode` holding the denoised clip.

    Raises:
        vs.Error: If a parameter that does not apply to the fast variant is passed, or if the clip's format
            is not supported.
    """

    accelerators_string: str | None = None
    if accelerators is not None:
        accelerators_string = ",".join(accelerators)

    return _forward(
        _nlmeans_filter(),
        clip,
        variant="fast",
        preset=preset,
        channel_mode=channel_mode,
        device=device,
        strength=strength,
        luma_strength=luma_strength,
        chroma_strength=chroma_strength,
        motion_compensation=motion_compensation,
        accelerators=accelerators_string,
        **kwargs,
    )


def NlmHQ(
    clip: "vs.VideoNode",
    *,
    preset: Preset | None = None,
    channel_mode: ChannelMode | None = None,
    device: str | None = None,
    sigma_scale: float | None = None,
    strength: float | None = None,
    luma_strength: float | None = None,
    chroma_strength: float | None = None,
    motion_compensation: bool | None = None,
    accelerators: list[Accelerators] | None = None,
    **kwargs: Any,
) -> "vs.VideoNode":
    """
    Runs the NLMeans-HQ denoiser.

    `NlmHQ` measures the noise level per scene and uses that to drive filtering.
    See https://github.com/ChillFish8/av-denoise/docs/CHOOSING_AN_ALGORITHM.md for working
    out what algorithm is right for you.

    Args:
        clip: The input video node to denoise.
        preset: Preset name. Unset dials fall back to whatever the av-denoise binary defaults to.
        channel_mode: Which planes to filter, one of `"luma"`, `"chroma"`, `"lumachroma"` or `"yuv"`.
        device: Compute device to run on. `"cpu"` selects a software device where the platform offers one,
            useful for testing, not for real encodes. Example: `"discrete:0"` for discrete GPU 0.
        sigma_scale: Multiplier nudging the measured noise level up or down.
            This is the right dial to reach for when leftover grain survives filtering,
            rather than raising `strength`.
        strength: Overall filter strength. Raising this to fight leftover grain is the wrong move, grain
            that survives usually means the noise level read low, correcting `sigma_scale` is the right
            move instead.
        luma_strength: Strength override for the luma plane.
        chroma_strength: Strength override for the chroma planes.
        motion_compensation: Whether to motion-compensate neighbour frames before matching.
        accelerators: The accelerators to try and use in the order to attempt.
        **kwargs: Further parameters reachable by name, documented in the
            `av-denoise` CLI documentation under their flag names.

    Returns:
        A new `vs.VideoNode` holding the denoised clip.

    Raises:
        vs.Error: If a parameter that does not apply to the hq variant is
            passed, or if the clip's format is not supported.
    """

    accelerators_string: str | None = None
    if accelerators is not None:
        accelerators_string = ",".join(accelerators)

    return _forward(
        _nlmeans_filter(),
        clip,
        variant="hq",
        preset=preset,
        channel_mode=channel_mode,
        device=device,
        sigma_scale=sigma_scale,
        strength=strength,
        luma_strength=luma_strength,
        chroma_strength=chroma_strength,
        motion_compensation=motion_compensation,
        accelerators=accelerators_string,
        **kwargs,
    )


def Nl4d(
    clip: "vs.VideoNode",
    *,
    preset: Preset | None = None,
    channel_mode: ChannelMode | None = None,
    device: str | None = None,
    lambda_ht_scale: float | None = None,
    lambda_ht: float | None = None,
    sigma_scale: float | None = None,
    spatial_radius: int | None = None,
    refine: int | None = None,
    noise_map: bool | None = None,
    enable_psy: bool | None = None,
    psy_flat_boost: float | None = None,
    psy_chroma_flat_boost: float | None = None,
    psy_shadow_soften: float | None = None,
    psy_flat_texture_cut: float | None = None,
    psy_line_ring: int | None = None,
    pooled_threshold: bool | None = None,
    accelerators: list[Accelerators] | None = None,
    **kwargs: Any,
) -> "vs.VideoNode":
    """
    Runs the NL4D spatio-temporal denoiser.

    See https://github.com/ChillFish8/av-denoise/docs/CHOOSING_AN_ALGORITHM.md for working
    out what algorithm is right for you.

    Args:
        clip: The input video node to denoise.
        preset: Preset name. Unset dials fall back to whatever the av-denoise binary defaults to.
        channel_mode: Which planes to filter, one of `"luma"`, `"chroma"`, `"lumachroma"` or `"yuv"`.
        device: Compute device to run on. `"cpu"` selects a software device where the platform offers one,
            useful for testing, not for real encodes. Example: `"discrete:0"` for discrete GPU 0.
        lambda_ht_scale: Threshold multiplier a transform coefficient's estimated-noise standard deviations
            must clear to survive. This  is the main dial, raising it removes more noise and takes more
            fine detail with it. Try it in steps of about 0.05.
        lambda_ht: Pins luma and chroma's thresholds to the same absolute number instead of
            their separate defaults, losing that separation. Prefer `lambda_ht_scale` first.
        sigma_scale: Multiplier nudging the measured noise level up or down, keeping the per-scene
            measurement rather than pinning it.
        spatial_radius: The speed dial. `preset` already resolves it, so setting this explicitly overrides
            whatever the preset picked. The centre-frame search covers `(2 * radius + 1)^2` positions,
            so it dominates the work, lowering it is the fastest way to speed a run-up.
        refine: Half-width of the window searched around each neighbour frame's motion-predicted position.
            Raise it when motion tracking lands close but not exact.
        noise_map: Whether to scale the luma threshold by how noisy each brightness level is.
            On by default, `False` turns it off.
        enable_psy: Turns on the psy options, which filter each area by what it looks like. Flat,
            grainy areas are filtered harder, textured dark areas more gently, and the areas around
            strong lines skip shadow soften and the flat texture cut. Off by default. Needs
            `noise_map`. The `psy_*` settings tune it and need it set.
        psy_flat_boost: How much harder flat, grainy areas are filtered, as a multiplier on the luma
            threshold. Between 1.0 and 3.0, 1.75 by default. `1.0` turns it off. Needs `enable_psy`.
        psy_chroma_flat_boost: The same for the colour planes. Between 1.0 and 3.0, 1.5 by default.
            `1.0` turns it off. Needs `enable_psy`.
        psy_shadow_soften: How much more gently textured dark areas are filtered, as a multiplier on
            the luma threshold. Between 0.1 and 1.0, 0.65 by default. `1.0` turns it off. Needs
            `enable_psy`.
        psy_flat_texture_cut: How strongly an area's surroundings may line up, like faint lines or
            edges, before it stops counting as flat. Between 0.0 and 1.0, 0.21 by default. Lower
            keeps more faint texture. `1.0` turns it off. Luma only. Needs `enable_psy`.
        psy_line_ring: How many pixels around a strong line skip shadow soften and the flat texture
            cut, so the grain beside dark ink lines is filtered like the rest of the frame. Between 0
            and 64, 16 by default. `0` turns it off. Luma only. Needs `enable_psy`.
        pooled_threshold: Whether to judge each frequency together with its neighbours, which
            keeps faint texture. On by default, `False` turns it off.
        accelerators: The accelerators to try and use in the order to attempt.
        **kwargs: Further parameters reachable by name, documented in the
            `av-denoise` CLI documentation under their flag names.

    Returns:
        A new `vs.VideoNode` holding the denoised clip.

    Raises:
        vs.Error: If a parameter that does not apply to this filter is
            passed, or if the clip's format is not supported.
    """

    accelerators_string: str | None = None
    if accelerators is not None:
        accelerators_string = ",".join(accelerators)

    return _forward(
        _nl4d_filter(),
        clip,
        preset=preset,
        channel_mode=channel_mode,
        device=device,
        lambda_ht_scale=lambda_ht_scale,
        lambda_ht=lambda_ht,
        sigma_scale=sigma_scale,
        spatial_radius=spatial_radius,
        refine=refine,
        noise_map=noise_map,
        enable_psy=enable_psy,
        psy_flat_boost=psy_flat_boost,
        psy_chroma_flat_boost=psy_chroma_flat_boost,
        psy_shadow_soften=psy_shadow_soften,
        psy_flat_texture_cut=psy_flat_texture_cut,
        psy_line_ring=psy_line_ring,
        pooled_threshold=pooled_threshold,
        accelerators=accelerators_string,
        **kwargs,
    )
