use cubecl::prelude::*;

/// Copies `length` elements from `src[src_offset..]` into
/// `dst[dst_offset..]`, entirely on the GPU.
///
/// The loop is strided so the grid can stay under the 65,535 dispatch
/// limit.
///
/// The offsets are kernel arguments rather than byte offsets on the
/// bound handles, which lets a caller address any slot of a ring buffer
/// whatever its stride. The GPU only accepts a buffer bound at a
/// multiple of its `min_storage_buffer_offset_alignment`, and a
/// `width * height * stored_ch` frame stride rarely lands on one.
#[cube(launch_unchecked)]
pub fn gpu_copy(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_offset: u32,
    dst_offset: u32,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[(dst_offset + idx) as usize] = src[(src_offset + idx) as usize];
        idx += total_threads;
    }
}

/// Zeroes `accum`, `weight_sum`, and `max_weight` in one dispatch.
///
/// The main loop covers all three up to `weight_len`, then a tail loop
/// finishes the channel-padded remainder of `accum`, which is always at
/// least as long as the other two.
#[cube(launch_unchecked)]
pub fn gpu_zero_buffers(
    accum: &mut Array<f32>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    #[comptime] accum_len: u32,
    #[comptime] weight_len: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;

    while idx < weight_len {
        accum[idx as usize] = 0.0f32;
        weight_sum[idx as usize] = 0.0f32;
        max_weight[idx as usize] = 0.0f32;
        idx += total_threads;
    }

    while idx < accum_len {
        accum[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}

/// Quantizes a denoised frame into wire bytes, packed into `u32` words.
///
/// `src` holds `pixels * stored_ch` values. The lanes between `channels`
/// and `stored_ch` are padding and are skipped here, so the host reads
/// back only the samples it asked for.
///
/// `outer` and `split_planes` decide how an output sample index maps
/// back into `src`. Interleaved output passes `outer = channels`, so the
/// quotient is the pixel and the remainder is the channel. Split output,
/// which is what a chroma pair needs, passes `outer = pixels` and gets
/// the reverse, laying each channel down as one contiguous region.
///
/// `samples_per_word` is 4 for 8-bit wire and 2 for 10 and 12-bit, which
/// matches the host's `Narrow` and `Wide` codecs.
///
/// A NaN sample does not come out as zero the way the host converter's
/// clamp does. The GPU clamp lowers to a min and max pair whose NaN
/// result is unspecified, and the float to integer cast is undefined on
/// NaN, so such a sample lands on an arbitrary byte. A denoised frame
/// holds no NaN, so nothing guards against it here.
///
/// Every index is clamped rather than guarded by a branch. A
/// branch-derived index inside an unrolled loop makes cubecl's GVN pass
/// panic while compiling the shader, after which the launch silently
/// writes nothing.
///
/// The loop is strided so the grid can stay under the dispatch limit.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a comptime shape the kernel specialises on"
)]
pub fn gpu_pack_wire(
    src: &Array<f32>,
    dst: &mut Array<u32>,
    max: f32,
    #[comptime] pixels: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] outer: u32,
    #[comptime] split_planes: bool,
    #[comptime] samples_per_word: u32,
    #[comptime] words: u32,
    #[comptime] total_threads: u32,
) {
    let samples = comptime![pixels * channels];
    let bits = comptime![32u32 / samples_per_word];

    let mut word = ABSOLUTE_POS_X;

    while word < words {
        let base = word * samples_per_word;
        let mut acc = 0u32;

        #[unroll]
        for lane in 0..samples_per_word {
            let s = base + lane;
            // Clamped, never branched. A lane past the last sample still
            // reads a valid slot, and its value is dropped below.
            let safe = u32::min(s, samples - 1);

            let a = safe / outer;
            let b = safe % outer;
            let src_idx = select(split_planes, b * stored_ch + a, a * stored_ch + b);

            let v = f32::clamp(src[src_idx as usize], 0.0, 1.0);
            let q = u32::cast_from(v * max + 0.5);

            acc |= select(s < samples, q, 0u32) << (lane * bits);
        }

        dst[word as usize] = acc;
        word += total_threads;
    }
}
