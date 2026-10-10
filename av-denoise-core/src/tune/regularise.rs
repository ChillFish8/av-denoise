use cubecl::AutotuneKey;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use serde::{Deserialize, Serialize};

use super::{TuneId, alternatives, fits, zeroed};
use crate::nl4d::kernels::{nl4d_mv_regularise, nl4d_mv_regularise_coop};

#[derive(Clone, Copy, Debug)]
pub enum RegulariseKernel {
    Serial,
    Cooperative,
}

#[derive(Clone, Copy, Debug)]
pub struct RegulariseCandidate {
    pub kernel: RegulariseKernel,
    pub dim_x: u32,
    pub dim_y: u32,
}

/// Every regularise launch the tuner tries. The first is the default launch.
pub const REGULARISE_CANDIDATES: [RegulariseCandidate; 5] = [
    RegulariseCandidate {
        kernel: RegulariseKernel::Serial,
        dim_x: 8,
        dim_y: 8,
    },
    RegulariseCandidate {
        kernel: RegulariseKernel::Serial,
        dim_x: 8,
        dim_y: 4,
    },
    RegulariseCandidate {
        kernel: RegulariseKernel::Serial,
        dim_x: 4,
        dim_y: 4,
    },
    RegulariseCandidate {
        kernel: RegulariseKernel::Cooperative,
        dim_x: 8,
        dim_y: 4,
    },
    RegulariseCandidate {
        kernel: RegulariseKernel::Cooperative,
        dim_x: 8,
        dim_y: 8,
    },
];

/// Whether every cooperative candidate has a power of two thread count of at least 8.
const fn cooperative_threads_are_valid() -> bool {
    let mut index = 0;
    while index < REGULARISE_CANDIDATES.len() {
        let candidate = REGULARISE_CANDIDATES[index];
        let threads = candidate.dim_x * candidate.dim_y;
        let cooperative = matches!(candidate.kernel, RegulariseKernel::Cooperative);
        if cooperative && (!threads.is_power_of_two() || threads < 8) {
            return false;
        }

        index += 1;
    }

    true
}

const _: () = assert!(cooperative_threads_are_valid());

/// Bytes of shared memory the cooperative kernel allocates for a cube of `threads`.
fn cooperative_shared_bytes(blksize: u32, threads: u32) -> usize {
    let centre_tile = blksize * blksize;
    let partials = 7 * threads;
    let scratch = 14 + 2 + 7 + 7;

    size_of::<f32>() * (centre_tile + partials + scratch) as usize
}

pub fn candidate_label(index: usize) -> String {
    let candidate = REGULARISE_CANDIDATES[index];
    let kernel = match candidate.kernel {
        RegulariseKernel::Serial => "serial",
        RegulariseKernel::Cooperative => "coop",
    };

    format!("c{index}_{kernel}_{}x{}", candidate.dim_x, candidate.dim_y)
}

#[derive(AutotuneKey, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct RegulariseKey {
    blksize: u32,
    step: u32,
    #[autotune(anchor)]
    width: usize,
    #[autotune(anchor)]
    height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct RegulariseShape {
    pub level_width: u32,
    pub level_height: u32,
    pub blksize: u32,
    pub step: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

/// One regularise launch over one neighbour, with everything a candidate needs to run it.
#[derive(Clone)]
pub struct RegulariseLaunch<R: Runtime> {
    pub client: ComputeClient<R>,
    pub centre: Handle,
    pub neighbour: Handle,
    pub level_len: usize,
    pub mv_in: Handle,
    pub mv_out: Handle,
    pub conf_out: Handle,
    pub lambda_pixel: f32,
    pub sad_noise_floor: f32,
    pub thsad: f32,
    pub shape: RegulariseShape,
}

impl<R: Runtime> RegulariseLaunch<R> {
    fn blocks(&self) -> usize {
        (self.shape.blocks_x * self.shape.blocks_y) as usize
    }

    pub(crate) fn key(&self) -> RegulariseKey {
        let width = self.shape.level_width as usize;
        let height = self.shape.level_height as usize;
        RegulariseKey::new(self.shape.blksize, self.shape.step, width, height)
    }

    /// A copy whose output buffers are fresh zeroed scratch.
    pub(crate) fn with_scratch(&self) -> Self {
        let blocks = self.blocks();
        let mv_bytes = 2 * blocks * size_of::<i32>();
        let conf_bytes = blocks * size_of::<f32>();

        Self {
            mv_out: zeroed(&self.client, mv_bytes),
            conf_out: zeroed(&self.client, conf_bytes),
            ..self.clone()
        }
    }

    pub fn launch_candidate(&self, index: usize) -> Result<(), String> {
        let candidate = REGULARISE_CANDIDATES[index];
        let threads = candidate.dim_x * candidate.dim_y;
        let hardware = &self.client.properties().hardware;
        let shared_bytes = match candidate.kernel {
            RegulariseKernel::Serial => 0,
            RegulariseKernel::Cooperative => cooperative_shared_bytes(self.shape.blksize, threads),
        };
        if index != 0 && !fits(hardware, threads, shared_bytes) {
            let label = candidate_label(index);
            return Err(format!("{label} does not fit this device"));
        }

        let blocks = self.blocks();
        let shape = self.shape;
        let grid = CubeCount::new_2d(shape.blocks_x, shape.blocks_y);
        let dim = CubeDim::new_2d(candidate.dim_x, candidate.dim_y);

        match candidate.kernel {
            RegulariseKernel::Serial => unsafe {
                nl4d_mv_regularise::launch_unchecked::<R>(
                    &self.client,
                    grid,
                    dim,
                    ArrayArg::from_raw_parts(self.centre.clone(), self.level_len),
                    ArrayArg::from_raw_parts(self.neighbour.clone(), self.level_len),
                    ArrayArg::from_raw_parts(self.mv_in.clone(), 2 * blocks),
                    ArrayArg::from_raw_parts(self.mv_out.clone(), 2 * blocks),
                    ArrayArg::from_raw_parts(self.conf_out.clone(), blocks),
                    self.lambda_pixel,
                    self.sad_noise_floor,
                    self.thsad,
                    shape.level_width,
                    shape.level_height,
                    shape.blksize,
                    shape.step,
                    shape.blocks_x,
                    shape.blocks_y,
                );
            },
            RegulariseKernel::Cooperative => unsafe {
                nl4d_mv_regularise_coop::launch_unchecked::<R>(
                    &self.client,
                    grid,
                    dim,
                    ArrayArg::from_raw_parts(self.centre.clone(), self.level_len),
                    ArrayArg::from_raw_parts(self.neighbour.clone(), self.level_len),
                    ArrayArg::from_raw_parts(self.mv_in.clone(), 2 * blocks),
                    ArrayArg::from_raw_parts(self.mv_out.clone(), 2 * blocks),
                    ArrayArg::from_raw_parts(self.conf_out.clone(), blocks),
                    self.lambda_pixel,
                    self.sad_noise_floor,
                    self.thsad,
                    shape.level_width,
                    shape.level_height,
                    shape.blksize,
                    shape.step,
                    shape.blocks_x,
                    shape.blocks_y,
                    threads,
                );
            },
        }

        Ok(())
    }
}

/// Runs the fastest regularise candidate for this device, tuning on the first call for a key.
pub(crate) fn launch<R: Runtime>(regularise: RegulariseLaunch<R>) {
    static TUNER: LocalTuner<RegulariseKey, TuneId> = local_tuner!("regularise");

    let tunables = TUNER.init(|| {
        let group = alternatives();
        let mut set = TunableSet::new(
            |regularise: &RegulariseLaunch<R>| regularise.key(),
            |_key: &RegulariseKey, regularise: &RegulariseLaunch<R>| regularise.with_scratch(),
        );

        for index in 0..REGULARISE_CANDIDATES.len() {
            let label = candidate_label(index);
            let tunable = Tunable::new(&label, move |regularise: RegulariseLaunch<R>| {
                regularise.launch_candidate(index)
            });

            set = match index {
                0 => set.with(tunable),
                _ => set.with(tunable.group(&group, |_key| 0)),
            };
        }

        set
    });

    let client = regularise.client.clone();
    let tune_id = TuneId::new(&client);
    TUNER.execute(&tune_id, &client, tunables, regularise);
}
