use cubecl::cmma;
use cubecl::features::TypeUsage;
use cubecl::ir::StorageType;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use super::{
    MATRIX_TILE,
    TILE_VALUES,
    combine_split_dot,
    combine_split_dot_i32,
    fragment,
    int8_part,
    split_code,
    store_guarded,
};
use crate::tune::gpu::{bind, read_f32, read_i32};
use crate::tune::tests::runtime::{TestRuntime, test_client};
use crate::tune::{
    DeviceLimits,
    Int8Layout,
    Int8Matrix,
    IntegerCodes,
    MATRIX_PLANE_WIDTHS,
    compiles_matrix_products,
    int8_matrix_plane_width,
    lists_int8_matrix,
    matrix_plane_width,
    offers_f16_matrix,
    visible_architectures,
};

/// The logical planes each smoke cube runs.
const PLANES: u32 = 2;

/// The top codes row 0 of `a` and column 0 of `b` hold in the near-limit products.
const NEAR_LIMIT_VALUES: u32 = 243;

const EIGHT_BIT: IntegerCodes = IntegerCodes { depth: 8 };
const TEN_BIT: IntegerCodes = IntegerCodes { depth: 10 };

/// The index in `a` of value `index` of a logical plane's 16x16 slice at k-step `step`.
///
/// Each plane's `a` block is 16 rows of `16 * k_steps` values, row-major.
#[cube]
fn a_index(plane: u32, step: u32, index: u32, #[comptime] k_steps: u32) -> u32 {
    let depth = comptime!(k_steps * MATRIX_TILE);
    let block = plane * comptime!(k_steps * TILE_VALUES);
    let row = index / MATRIX_TILE;
    let column = index % MATRIX_TILE;

    block + row * depth + step * MATRIX_TILE + column
}

/// The index in `b` of value `index` of a logical plane's 16x16 slice at k-step `step`.
///
/// Each plane's `b` block is `16 * k_steps` rows of 16 values, row-major.
#[cube]
fn b_index(plane: u32, step: u32, index: u32, #[comptime] k_steps: u32) -> u32 {
    let block = plane * comptime!(k_steps * TILE_VALUES);

    block + step * TILE_VALUES + index
}

/// Loads the 16x16 row-major slice at `start` of `slices` as an operand fragment.
#[cube]
fn load_operand<E: Scalar>(
    slices: &SharedMemory<E>,
    start: u32,
    #[comptime] ident: cmma::MatrixIdent,
) -> cmma::Matrix<E> {
    let end = start + TILE_VALUES;
    let slice = slices.slice(start as usize, end as usize);
    let operand = fragment::<E>(ident, cmma::MatrixLayout::RowMajor);
    cmma::load(&operand, &slice, MATRIX_TILE);

    operand
}

/// Multiplies each logical plane's 16xK `a` block by its Kx16 `b` block into `out`, with K = 16 * k_steps.
#[cube(launch_unchecked)]
fn matrix_smoke(
    a: &Array<f16>,
    b: &Array<f16>,
    out: &mut Array<f32>,
    #[comptime] plane_width: u32,
    #[comptime] k_steps: u32,
) {
    let plane = UNIT_POS_X / plane_width;
    let lane = UNIT_POS_X % plane_width;
    let tile_start = plane * TILE_VALUES;
    let rounds = comptime!(TILE_VALUES / plane_width);
    let shared_values = comptime!((PLANES * TILE_VALUES) as usize);

    let mut a_slices = SharedMemory::<f16>::new(shared_values);
    let mut b_slices = SharedMemory::<f16>::new(shared_values);
    let mut products = SharedMemory::<f32>::new(shared_values);

    let accumulator = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    cmma::fill(&accumulator, 0.0);

    for step in 0..k_steps {
        for round in 0..rounds {
            let index = round * plane_width + lane;
            let a_at = a_index(plane, step, index, k_steps);
            let b_at = b_index(plane, step, index, k_steps);
            let slot = (tile_start + index) as usize;

            a_slices[slot] = a[a_at as usize];
            b_slices[slot] = b[b_at as usize];
        }

        sync_cube();

        let lhs = load_operand(&a_slices, tile_start, cmma::MatrixIdent::A);
        let rhs = load_operand(&b_slices, tile_start, cmma::MatrixIdent::B);
        cmma::execute::<f16, f16, f32, f32>(&lhs, &rhs, &accumulator, &accumulator);

        sync_cube();
    }

    store_guarded(&mut products, tile_start, &accumulator, plane_width);
    sync_cube();

    for round in 0..rounds {
        let slot = (tile_start + round * plane_width + lane) as usize;
        out[slot] = products[slot];
    }
}

/// Multiplies each logical plane's blocks of codes as split f16 parts, and writes the exact dot
/// [combine_split_dot](crate::matrix::combine_split_dot) rebuilds from the three products.
#[cube(launch_unchecked)]
fn matrix_split_smoke(
    a: &Array<u32>,
    b: &Array<u32>,
    out: &mut Array<i32>,
    #[comptime] plane_width: u32,
    #[comptime] k_steps: u32,
    #[comptime] shift: u32,
) {
    let plane = UNIT_POS_X / plane_width;
    let lane = UNIT_POS_X % plane_width;
    let tile_start = plane * TILE_VALUES;
    let rounds = comptime!(TILE_VALUES / plane_width);
    let shared_values = comptime!((PLANES * TILE_VALUES) as usize);

    let mut a_highs = SharedMemory::<f16>::new(shared_values);
    let mut a_lows = SharedMemory::<f16>::new(shared_values);
    let mut b_highs = SharedMemory::<f16>::new(shared_values);
    let mut b_lows = SharedMemory::<f16>::new(shared_values);
    let mut high_products = SharedMemory::<f32>::new(shared_values);
    let mut cross_products = SharedMemory::<f32>::new(shared_values);
    let mut low_products = SharedMemory::<f32>::new(shared_values);

    let high_high = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    let cross = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    let low_low = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    cmma::fill(&high_high, 0.0);
    cmma::fill(&cross, 0.0);
    cmma::fill(&low_low, 0.0);

    for step in 0..k_steps {
        for round in 0..rounds {
            let index = round * plane_width + lane;
            let a_at = a_index(plane, step, index, k_steps);
            let b_at = b_index(plane, step, index, k_steps);
            let slot = (tile_start + index) as usize;
            let a_code = a[a_at as usize];
            let b_code = b[b_at as usize];
            let (a_high, a_low) = split_code(a_code, shift);
            let (b_high, b_low) = split_code(b_code, shift);

            a_highs[slot] = a_high;
            a_lows[slot] = a_low;
            b_highs[slot] = b_high;
            b_lows[slot] = b_low;
        }

        sync_cube();

        let lhs_high = load_operand(&a_highs, tile_start, cmma::MatrixIdent::A);
        let lhs_low = load_operand(&a_lows, tile_start, cmma::MatrixIdent::A);
        let rhs_high = load_operand(&b_highs, tile_start, cmma::MatrixIdent::B);
        let rhs_low = load_operand(&b_lows, tile_start, cmma::MatrixIdent::B);
        cmma::execute::<f16, f16, f32, f32>(&lhs_high, &rhs_high, &high_high, &high_high);
        cmma::execute::<f16, f16, f32, f32>(&lhs_high, &rhs_low, &cross, &cross);
        cmma::execute::<f16, f16, f32, f32>(&lhs_low, &rhs_high, &cross, &cross);
        cmma::execute::<f16, f16, f32, f32>(&lhs_low, &rhs_low, &low_low, &low_low);

        sync_cube();
    }

    store_guarded(&mut high_products, tile_start, &high_high, plane_width);
    store_guarded(&mut cross_products, tile_start, &cross, plane_width);
    store_guarded(&mut low_products, tile_start, &low_low, plane_width);
    sync_cube();

    for round in 0..rounds {
        let slot = (tile_start + round * plane_width + lane) as usize;
        let high_sum = high_products[slot];
        let cross_sum = cross_products[slot];
        let low_sum = low_products[slot];

        out[slot] = combine_split_dot(high_sum, cross_sum, low_sum, shift);
    }
}

/// Multiplies each logical plane's blocks of codes as int8 parts under `layout`, accumulating in
/// i32, and writes the exact dot.
///
/// A whole-code layout runs one product. A split layout runs four into three accumulators, which
/// [combine_split_dot_i32](crate::matrix::combine_split_dot_i32) rebuilds the dot from.
#[cube(launch_unchecked)]
fn matrix_smoke_int8<E: Int>(
    a: &Array<u32>,
    b: &Array<u32>,
    out: &mut Array<i32>,
    #[comptime] layout: Int8Layout,
    #[comptime] plane_width: u32,
    #[comptime] k_steps: u32,
) {
    let plane = UNIT_POS_X / plane_width;
    let lane = UNIT_POS_X % plane_width;
    let tile_start = plane * TILE_VALUES;
    let rounds = comptime!(TILE_VALUES / plane_width);
    let shared_values = comptime!((PLANES * TILE_VALUES) as usize);
    let split = comptime!(layout.low_bits > 0);

    let mut a_highs = SharedMemory::<E>::new(shared_values);
    let mut a_lows = SharedMemory::<E>::new(shared_values);
    let mut b_highs = SharedMemory::<E>::new(shared_values);
    let mut b_lows = SharedMemory::<E>::new(shared_values);
    let mut high_products = SharedMemory::<i32>::new(shared_values);
    let mut cross_products = SharedMemory::<i32>::new(shared_values);
    let mut low_products = SharedMemory::<i32>::new(shared_values);

    let high_high = fragment::<i32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    let cross = fragment::<i32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    let low_low = fragment::<i32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    cmma::fill(&high_high, 0i32);
    cmma::fill(&cross, 0i32);
    cmma::fill(&low_low, 0i32);

    for step in 0..k_steps {
        for round in 0..rounds {
            let index = round * plane_width + lane;
            let a_at = a_index(plane, step, index, k_steps);
            let b_at = b_index(plane, step, index, k_steps);
            let slot = (tile_start + index) as usize;
            let a_code = a[a_at as usize];
            let b_code = b[b_at as usize];

            a_highs[slot] = int8_part::<E>(a_code, layout, true);
            b_highs[slot] = int8_part::<E>(b_code, layout, true);

            if comptime!(split) {
                a_lows[slot] = int8_part::<E>(a_code, layout, false);
                b_lows[slot] = int8_part::<E>(b_code, layout, false);
            }
        }

        sync_cube();

        let lhs_high = load_operand(&a_highs, tile_start, cmma::MatrixIdent::A);
        let rhs_high = load_operand(&b_highs, tile_start, cmma::MatrixIdent::B);
        cmma::execute::<E, E, i32, i32>(&lhs_high, &rhs_high, &high_high, &high_high);

        if comptime!(split) {
            let lhs_low = load_operand(&a_lows, tile_start, cmma::MatrixIdent::A);
            let rhs_low = load_operand(&b_lows, tile_start, cmma::MatrixIdent::B);
            cmma::execute::<E, E, i32, i32>(&lhs_high, &rhs_low, &cross, &cross);
            cmma::execute::<E, E, i32, i32>(&lhs_low, &rhs_high, &cross, &cross);
            cmma::execute::<E, E, i32, i32>(&lhs_low, &rhs_low, &low_low, &low_low);
        }

        sync_cube();
    }

    store_guarded(&mut high_products, tile_start, &high_high, plane_width);

    if comptime!(split) {
        store_guarded(&mut cross_products, tile_start, &cross, plane_width);
        store_guarded(&mut low_products, tile_start, &low_low, plane_width);
    }

    sync_cube();

    for round in 0..rounds {
        let slot = (tile_start + round * plane_width + lane) as usize;
        let high_sum = high_products[slot];

        if comptime!(split) {
            let cross_sum = cross_products[slot];
            let low_sum = low_products[slot];
            out[slot] = combine_split_dot_i32(high_sum, cross_sum, low_sum, layout.low_bits);
        } else {
            out[slot] = high_sum;
        }
    }
}

fn hashed_code(index: u32, seed: u32, max_code: u32) -> u32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    hash % (max_code + 1)
}

/// The `a` and `b` code blocks of every logical plane.
///
/// Each plane's `a` block is 16 rows of `16 * k_steps` codes and its `b` block is `16 * k_steps`
/// rows of 16 codes, both row-major.
#[derive(Debug)]
struct Operands {
    a: Vec<u32>,
    b: Vec<u32>,
    k_steps: u32,
}

impl Operands {
    /// Hashed codes up to `max_code`, ending in a 0 and a `max_code` in both blocks.
    fn hashed(k_steps: u32, max_code: u32, seed: u32) -> Self {
        let len = PLANES * k_steps * TILE_VALUES;
        let mut a: Vec<u32> = (0..len).map(|index| hashed_code(index, seed, max_code)).collect();
        let mut b: Vec<u32> = (0..len)
            .map(|index| hashed_code(index, seed + 1, max_code))
            .collect();

        let last = len as usize - 1;
        for codes in [&mut a, &mut b] {
            codes[last - 1] = 0;
            codes[last] = max_code;
        }

        Self { a, b, k_steps }
    }

    /// Every code of both blocks set to `code`.
    fn filled(k_steps: u32, code: u32) -> Self {
        let len = (PLANES * k_steps * TILE_VALUES) as usize;
        let a = vec![code; len];
        let b = vec![code; len];

        Self { a, b, k_steps }
    }

    fn depth(&self) -> u32 {
        self.k_steps * MATRIX_TILE
    }

    /// Sets the first `count` codes of row 0 of `a` and of column 0 of `b` to `code`, and the rest
    /// of that row and column to 0, in every plane.
    fn with_corner(mut self, count: u32, code: u32) -> Self {
        let depth = self.depth();

        for plane in 0..PLANES {
            let block = plane * depth * MATRIX_TILE;

            for k in 0..depth {
                let value = if k < count { code } else { 0 };
                let b_at = block + k * MATRIX_TILE;

                self.a[(block + k) as usize] = value;
                self.b[b_at as usize] = value;
            }
        }

        self
    }

    /// The host's i64 dot of every output element, plane by plane and row-major.
    fn expected(&self) -> Vec<i64> {
        let depth = self.depth();
        let mut dots = Vec::with_capacity((PLANES * TILE_VALUES) as usize);

        for plane in 0..PLANES {
            let block = plane * depth * MATRIX_TILE;

            for row in 0..MATRIX_TILE {
                for column in 0..MATRIX_TILE {
                    let dot: i64 = (0..depth)
                        .map(|k| {
                            let a_value = self.a[(block + row * depth + k) as usize];
                            let b_value = self.b[(block + k * MATRIX_TILE + column) as usize];
                            i64::from(a_value) * i64::from(b_value)
                        })
                        .sum();
                    dots.push(dot);
                }
            }
        }

        dots
    }
}

/// Checks every output equals the host's dot, listing every element that does not.
fn assert_exact<T: Copy + PartialEq + std::fmt::Debug>(
    out: &[T],
    expected: &[i64],
    exact: impl Fn(i64) -> T,
    label: &str,
) {
    assert_eq!(out.len(), expected.len(), "{label}: output length");

    let mismatches: Vec<String> = out
        .iter()
        .zip(expected)
        .enumerate()
        .filter_map(|(index, (&value, &dot))| {
            let wanted = exact(dot);
            let plane = index as u32 / TILE_VALUES;
            let row = (index as u32 % TILE_VALUES) / MATRIX_TILE;
            let column = index as u32 % MATRIX_TILE;
            let line = format!("plane {plane} ({row}, {column}): got {value:?}, host {dot}");

            (value != wanted).then_some(line)
        })
        .collect();

    assert!(
        mismatches.is_empty(),
        "{label}: {} of {} elements differ:\n{}",
        mismatches.len(),
        out.len(),
        mismatches.join("\n"),
    );
}

/// Checks element (0, 0) of every logical plane is `corner`.
fn assert_corners<T: Copy + PartialEq + std::fmt::Debug>(out: &[T], corner: T, label: &str) {
    for plane in 0..PLANES {
        let value = out[(plane * TILE_VALUES) as usize];
        assert_eq!(value, corner, "{label}: plane {plane} (0, 0)");
    }
}

/// The bits of the f32 an exact f16 product holds for `dot`, which is below 2^24.
fn f32_dot(dot: i64) -> u32 {
    let value = dot as f32;
    value.to_bits()
}

fn i32_dot(dot: i64) -> i32 {
    i32::try_from(dot).expect("every smoke dot fits i32")
}

fn upload_halves(client: &ComputeClient<TestRuntime>, codes: &[u32]) -> Handle {
    let values: Vec<f16> = codes.iter().map(|&code| f16::from_f32(code as f32)).collect();
    let bytes = f16::as_bytes(&values);

    client.create_from_slice(bytes)
}

fn upload_codes(client: &ComputeClient<TestRuntime>, codes: &[u32]) -> Handle {
    let bytes = u32::as_bytes(codes);
    client.create_from_slice(bytes)
}

fn smoke_launch_shape(plane_width: u32) -> (CubeCount, CubeDim) {
    let cube_count = CubeCount::new_1d(1);
    let cube_dim = CubeDim::new_1d(PLANES * plane_width);

    (cube_count, cube_dim)
}

const OUT_LEN: usize = (PLANES * TILE_VALUES) as usize;

/// The bytes of every smoke output, whose f32 and i32 values both take 4 bytes.
const OUT_BYTES: usize = OUT_LEN * size_of::<f32>();

/// Runs [matrix_smoke] on whole codes as f16 and reads the f32 products back as bits.
fn run_f16_product(operands: &Operands, plane_width: u32) -> Vec<u32> {
    let client = test_client();
    let a = upload_halves(&client, &operands.a);
    let b = upload_halves(&client, &operands.b);
    let out = client.empty(OUT_BYTES);
    let a_arg = bind::<TestRuntime>(&a, operands.a.len());
    let b_arg = bind::<TestRuntime>(&b, operands.b.len());
    let out_arg = bind::<TestRuntime>(&out, OUT_LEN);
    let (cube_count, cube_dim) = smoke_launch_shape(plane_width);

    // SAFETY: each block holds `PLANES * 256 * k_steps` values and `out` holds `PLANES * 256`,
    // which is every index the kernel reads or writes.
    unsafe {
        matrix_smoke::launch_unchecked::<TestRuntime>(
            &client,
            cube_count,
            cube_dim,
            a_arg,
            b_arg,
            out_arg,
            plane_width,
            operands.k_steps,
        );
    }

    let products = read_f32(&client, &out, OUT_LEN).expect("matrix readback failed");
    products.iter().map(|value| value.to_bits()).collect()
}

/// Runs [matrix_split_smoke] with codes split at `shift` and reads the dots back.
fn run_split_product(operands: &Operands, plane_width: u32, shift: u32) -> Vec<i32> {
    let client = test_client();
    let a = upload_codes(&client, &operands.a);
    let b = upload_codes(&client, &operands.b);
    let out = client.empty(OUT_BYTES);
    let a_arg = bind::<TestRuntime>(&a, operands.a.len());
    let b_arg = bind::<TestRuntime>(&b, operands.b.len());
    let out_arg = bind::<TestRuntime>(&out, OUT_LEN);
    let (cube_count, cube_dim) = smoke_launch_shape(plane_width);

    // SAFETY: as in `run_f16_product`.
    unsafe {
        matrix_split_smoke::launch_unchecked::<TestRuntime>(
            &client,
            cube_count,
            cube_dim,
            a_arg,
            b_arg,
            out_arg,
            plane_width,
            operands.k_steps,
            shift,
        );
    }

    read_i32(&client, &out, OUT_LEN).expect("matrix readback failed")
}

/// Runs [matrix_smoke_int8] under `layout` and reads the dots back.
fn run_int8_product(operands: &Operands, plane_width: u32, layout: Int8Layout) -> Vec<i32> {
    let client = test_client();
    let a = upload_codes(&client, &operands.a);
    let b = upload_codes(&client, &operands.b);
    let out = client.empty(OUT_BYTES);
    let a_arg = bind::<TestRuntime>(&a, operands.a.len());
    let b_arg = bind::<TestRuntime>(&b, operands.b.len());
    let out_arg = bind::<TestRuntime>(&out, OUT_LEN);
    let (cube_count, cube_dim) = smoke_launch_shape(plane_width);
    let k_steps = operands.k_steps;

    // SAFETY: as in `run_f16_product`.
    unsafe {
        match layout.element {
            Int8Matrix::Unsigned => matrix_smoke_int8::launch_unchecked::<u8, TestRuntime>(
                &client,
                cube_count,
                cube_dim,
                a_arg,
                b_arg,
                out_arg,
                layout,
                plane_width,
                k_steps,
            ),
            Int8Matrix::Signed => matrix_smoke_int8::launch_unchecked::<i8, TestRuntime>(
                &client,
                cube_count,
                cube_dim,
                a_arg,
                b_arg,
                out_arg,
                layout,
                plane_width,
                k_steps,
            ),
        }
    }

    read_i32(&client, &out, OUT_LEN).expect("matrix readback failed")
}

/// Prints the f16 skip line, after checking a skip of a listed config comes from the matrix rule.
///
/// A device that lists the f16 config at a matrix plane width skips only when CubeCL cannot
/// compile matrix products for it, so any other skip fails here.
fn skip_f16(client: &ComputeClient<TestRuntime>, limits: &DeviceLimits) {
    let runtime = TestRuntime::name(client);
    let configs = &client.properties().features.matmul.cmma;
    let listed = offers_f16_matrix(configs);
    let plane_supported = MATRIX_PLANE_WIDTHS.contains(&limits.plane_size_max);

    if listed && plane_supported {
        let architectures = visible_architectures(runtime);
        let compiles = compiles_matrix_products(runtime, &architectures);

        assert!(
            !compiles,
            "{runtime} on {architectures:?} lists the f16 matrix config, and it was not offered"
        );
    }

    eprintln!("no 16x16x16 f16 matrix config on this backend, skipping");
}

/// The f16 matrix plane of the test device, or `None` after a skip line.
fn offered_f16_plane() -> Option<u32> {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let width = matrix_plane_width(&limits);

    if width.is_none() {
        skip_f16(&client, &limits);
    }

    width
}

/// Whether the test device lists `element`'s exact config and runs `element` arithmetic.
fn lists_int8_product(client: &ComputeClient<TestRuntime>, element: Int8Matrix) -> bool {
    let properties = client.properties();
    let element_type = element.element();
    let storage_type = StorageType::Scalar(element_type);
    let usage = properties.type_usage(storage_type);
    let configs = &properties.features.matmul.cmma;
    let listed = lists_int8_matrix(configs, element);

    listed && usage.contains(TypeUsage::Arithmetic)
}

/// Every int8 product the test device lists, with the int8 plane, or `None` after a skip line.
///
/// The offered product comes first. The other is listed too when the device also runs it.
fn offered_int8() -> Option<(Vec<Int8Matrix>, u32)> {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let width = int8_matrix_plane_width(&limits);
    let offered = limits.int8_matrix.zip(width);

    let Some((preferred, plane_width)) = offered else {
        eprintln!("no 16x16x16 int8 matrix config on this backend, skipping");
        return None;
    };

    let mut elements = vec![preferred];
    for element in [Int8Matrix::Unsigned, Int8Matrix::Signed] {
        if element != preferred && lists_int8_product(&client, element) {
            elements.push(element);
        }
    }

    Some((elements, plane_width))
}

#[test]
fn the_device_reports_its_matrix_limits() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let backend = TestRuntime::name(&client);
    let matrix_plane = matrix_plane_width(&limits);
    let int8_plane = int8_matrix_plane_width(&limits);
    let configs = &client.properties().features.matmul.cmma;
    let architectures = visible_architectures(backend);
    let compiles = compiles_matrix_products(backend, &architectures);

    eprintln!("backend: {backend}");
    eprintln!("hip architectures: {architectures:?}");
    eprintln!("compiles matrix products: {compiles}");
    eprintln!("f16_matrix: {}", limits.f16_matrix);
    eprintln!("int8_matrix: {:?}", limits.int8_matrix);
    eprintln!("u16_buffer: {}", limits.u16_buffer);
    eprintln!("plane_size_min: {}", limits.plane_size_min);
    eprintln!("plane_size_max: {}", limits.plane_size_max);
    eprintln!("matrix_plane_width: {matrix_plane:?}");
    eprintln!("int8_matrix_plane_width: {int8_plane:?}");
    eprintln!("cmma configs as CubeCL reports them: {}", configs.len());

    for config in configs {
        eprintln!("  {config:?}");
    }
}

#[test]
fn an_f16_matrix_product_of_codes_is_exact_at_the_offered_plane_width() {
    let Some(plane_width) = offered_f16_plane() else {
        return;
    };
    let operands = Operands::hashed(4, 255, 11).with_corner(64, 255);
    let expected = operands.expected();
    let corner = 64.0f32 * 65025.0;
    let corner_bits = corner.to_bits();

    let products = run_f16_product(&operands, plane_width);

    assert_exact(&products, &expected, f32_dot, "f16 at K 64");
    assert_corners(&products, corner_bits, "f16 at K 64");
}

#[test]
fn a_split_ten_bit_product_combines_to_the_exact_dot() {
    let Some(plane_width) = offered_f16_plane() else {
        return;
    };
    let max_code = TEN_BIT.max_code();
    let shift = TEN_BIT.split_shift();
    let operands = Operands::hashed(4, max_code, 23).with_corner(64, max_code);
    let expected = operands.expected();

    let dots = run_split_product(&operands, plane_width, shift);

    assert_exact(&dots, &expected, i32_dot, "split 10-bit at K 64");
    assert_corners(&dots, 64 * 1023 * 1023, "split 10-bit at K 64");
}

#[test]
fn a_product_near_the_exact_limit_stays_exact() {
    let Some(plane_width) = offered_f16_plane() else {
        return;
    };
    let operands = Operands::hashed(16, 255, 37).with_corner(NEAR_LIMIT_VALUES, 255);
    let expected = operands.expected();
    let corner = 243.0f32 * 65025.0;
    let corner_bits = corner.to_bits();

    let products = run_f16_product(&operands, plane_width);

    assert_eq!(corner, 15_801_075.0);
    assert_exact(&products, &expected, f32_dot, "f16 at K 256");
    assert_corners(&products, corner_bits, "f16 at K 256");
}

#[test]
fn a_split_ten_bit_product_near_the_exact_limit_combines_exactly() {
    let Some(plane_width) = offered_f16_plane() else {
        return;
    };
    let max_code = TEN_BIT.max_code();
    let shift = TEN_BIT.split_shift();
    let operands = Operands::hashed(16, max_code, 41).with_corner(NEAR_LIMIT_VALUES, max_code);
    let expected = operands.expected();

    let dots = run_split_product(&operands, plane_width, shift);

    assert_exact(&dots, &expected, i32_dot, "split 10-bit at K 256");
    assert_corners(&dots, 254_306_547, "split 10-bit at K 256");
}

#[test]
fn every_int8_layout_the_device_offers_is_exact() {
    let Some((elements, plane_width)) = offered_int8() else {
        return;
    };

    eprintln!("int8 products checked: {elements:?}");

    let layouts = elements.iter().flat_map(|&element| {
        [EIGHT_BIT, TEN_BIT]
            .into_iter()
            .map(move |codes| (codes, Int8Layout::for_codes(element, codes)))
    });
    for (codes, layout) in layouts {
        let top = codes.max_code();
        let squared = (top * top) as i32;
        let hashed_short = Operands::hashed(4, top, 53).with_corner(64, top);
        let hashed_near = Operands::hashed(16, top, 59).with_corner(NEAR_LIMIT_VALUES, top);
        let all_top = Operands::filled(16, top);
        let cases = [
            (hashed_short, 64 * squared, "hashed at K 64"),
            (hashed_near, 243 * squared, "hashed near the limit at K 256"),
            (all_top, 256 * squared, "every code at the top at K 256"),
        ];

        for (operands, corner, case) in cases {
            let label = format!("{layout:?} at {} bits, {case}", codes.depth);
            let expected = operands.expected();

            let dots = run_int8_product(&operands, plane_width, layout);

            assert_exact(&dots, &expected, i32_dot, &label);
            assert_corners(&dots, corner, &label);
        }
    }
}

#[test]
fn the_offered_plane_width_is_the_widest_plane() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let width = matrix_plane_width(&limits);

    match width {
        Some(width) => assert_eq!(width, limits.plane_size_max),
        None => skip_f16(&client, &limits),
    }
}
