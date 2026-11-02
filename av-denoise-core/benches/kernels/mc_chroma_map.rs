use av_denoise_core::bench_api::kernels::motion::mc_map_luma_fields;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, shapes_with_channels};

const STEP: u32 = 8;
const BLKSIZE: u32 = 16;
const FIELDS: u32 = 8;
const SHIFT: u32 = 1;

/// The luma to chroma motion mapping at 1080p 4:2:0, over a full window of eight fields.
///
/// Each chroma block takes the median of a 3x3 luma footprint, so the cost is nine vector reads
/// and a rank count per block.
pub struct ChromaMapBench {
    pub client: Client,
}

#[derive(Clone)]
pub struct ChromaMapInput {
    pub luma_field: Handle,
    pub chroma_field: Handle,
}

struct Grid {
    luma_blocks_x: u32,
    luma_blocks_y: u32,
    chroma_blocks_x: u32,
    chroma_blocks_y: u32,
}

impl Grid {
    fn new() -> Self {
        Self {
            luma_blocks_x: WIDTH.div_ceil(STEP),
            luma_blocks_y: HEIGHT.div_ceil(STEP),
            chroma_blocks_x: (WIDTH / 2).div_ceil(STEP),
            chroma_blocks_y: (HEIGHT / 2).div_ceil(STEP),
        }
    }

    fn luma_stride(&self) -> u32 {
        self.luma_blocks_x * self.luma_blocks_y * 2
    }

    fn chroma_stride(&self) -> u32 {
        self.chroma_blocks_x * self.chroma_blocks_y * 2
    }
}

impl Benchmark for ChromaMapBench {
    type Input = ChromaMapInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let grid = Grid::new();
        let luma_len = (FIELDS * grid.luma_stride()) as usize;
        let vectors: Vec<i32> = (0..luma_len)
            .map(|index| (index as i32 * 7919) % 33 - 16)
            .collect();
        let vector_bytes = i32::as_bytes(&vectors);
        let luma_field = self.client.create_from_slice(vector_bytes);

        let chroma_len = (FIELDS * grid.chroma_stride()) as usize;
        let chroma_field = self.client.empty(chroma_len * size_of::<i32>());

        ChromaMapInput {
            luma_field,
            chroma_field,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let grid = Grid::new();
        let luma_len = (FIELDS * grid.luma_stride()) as usize;
        let chroma_len = (FIELDS * grid.chroma_stride()) as usize;
        let covers = ((BLKSIZE << SHIFT) - BLKSIZE) / STEP + 1;

        let cubes_x = grid.chroma_blocks_x.div_ceil(8);
        let cubes_y = grid.chroma_blocks_y.div_ceil(8);
        let cubes = CubeCount::new_3d(cubes_x, cubes_y, FIELDS);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            mc_map_luma_fields::launch_unchecked(
                &self.client,
                cubes,
                dim,
                BufferArg::from_raw_parts(args.luma_field.clone(), luma_len),
                BufferArg::from_raw_parts(args.chroma_field.clone(), chroma_len),
                grid.luma_blocks_x,
                grid.luma_blocks_y,
                STEP,
                grid.luma_stride(),
                grid.chroma_blocks_x,
                grid.chroma_blocks_y,
                STEP,
                grid.chroma_stride(),
                SHIFT,
                SHIFT,
                covers,
                covers,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "mc_chroma_map_1080p_420_8_fields".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
