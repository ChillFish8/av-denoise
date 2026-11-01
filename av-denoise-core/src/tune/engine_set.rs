use std::sync::Arc;
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::config::TuningConfig;
use super::device::{DeviceLimits, DeviceProfile};
use super::frame::TuningFrame;
use super::gpu::sync;
use super::identity::describe;
use super::ledger::TuneLedger;
use super::plan::PipelineProfile;
use super::profile_key::profile_key;
use super::setup::{KernelSetup, engine_kernels};
use crate::engine::{DevicePlane, PassEngine, PassSpec, WindowSpan};
use crate::error::Error;

/// The engines of one group, driven together on one device.
pub(crate) struct EngineSet {
    client: Client,
    device: cubecl::Device,
    profile: DeviceProfile,
    specs: Vec<PassSpec>,
    engines: Vec<Box<dyn PassEngine>>,
    scratch: Vec<Vec<Handle>>,
    ledger: Option<Arc<TuneLedger>>,
}

impl EngineSet {
    /// Builds an engine per spec from the store's picks, wired to `setup`.
    ///
    /// On a Vulkan device with a ledger, the ledger also compiles candidates on helper clients.
    /// Empty `specs` are an [Error::InvalidOptions].
    pub(crate) fn build(
        device: &cubecl::Device,
        client: &Client,
        specs: &[PassSpec],
        config: &TuningConfig,
        setup: &KernelSetup,
    ) -> Result<Self, Error> {
        if specs.is_empty() {
            let message = "an engine set needs at least one engine".to_string();
            return Err(Error::InvalidOptions(message));
        }

        let identity = describe(device, client);
        let limits = DeviceLimits::from_client(client);
        let profile = DeviceProfile { identity, limits };

        #[cfg(feature = "vulkan")]
        if let (Some(ledger), cubecl::Device::Wgpu(wgpu_device)) = (&setup.ledger, device)
            && wgpu_device.backend == cubecl::device::WgpuBackend::Vulkan
        {
            super::compile_helpers::install_compile_helpers(ledger, wgpu_device);
        }

        let mut engines = Vec::with_capacity(specs.len());
        let mut scratch = Vec::with_capacity(specs.len());

        for spec in specs {
            let key = profile_key(spec);
            let kernels = engine_kernels(setup, config, &key, &profile);
            let engine = spec.build(client, kernels)?;
            engines.push(engine);

            let handles = scratch_planes(client, spec);
            scratch.push(handles);
        }

        Ok(Self {
            client: client.clone(),
            device: device.clone(),
            profile,
            specs: specs.to_vec(),
            engines,
            scratch,
            ledger: setup.ledger.clone(),
        })
    }

    /// Pushes `frame` into every engine, emits each ready frame and waits for the device.
    ///
    /// Returns the time that took, or `None` when no engine had a frame ready. The set needs a
    /// [Self::reset] after an error.
    pub(crate) fn timed_frame(&mut self, frame: &TuningFrame) -> Result<Option<Duration>, Error> {
        sync(&self.client)?;

        let started = Instant::now();
        let mut ready_counts = Vec::with_capacity(self.engines.len());

        for (index, engine) in self.engines.iter_mut().enumerate() {
            let geometry = self.specs[index].geometry();
            let planes = frame.planes_for(index, geometry);
            let ready = engine.push(&planes)?;
            ready_counts.push(ready);
        }

        let ready = ready_counts.first().copied().unwrap_or(0);
        if ready_counts.iter().any(|count| *count != ready) {
            let message = format!("engines are out of step, their ready frame counts are {ready_counts:?}");
            let error = Error::InvalidOptions(message);
            return Err(error);
        }

        if ready == 0 {
            return Ok(None);
        }

        for (index, engine) in self.engines.iter_mut().enumerate() {
            let geometry = self.specs[index].geometry();
            let planes = scratch_device_planes(&self.scratch[index], geometry.width, geometry.height);

            for _ in 0..ready {
                engine.emit_into(&planes)?;
            }
        }

        sync(&self.client)?;

        Ok(Some(started.elapsed()))
    }

    /// The widest window any engine of the set reads, with the edges of the engine that reads it.
    pub(crate) fn window_span(&self) -> WindowSpan {
        let spans = self.engines.iter().map(|engine| engine.window_span());
        let widest = spans.max_by_key(WindowSpan::frame_count);

        widest.expect("an engine set holds at least one engine")
    }

    /// Abandons every engine's stream.
    pub(crate) fn reset(&mut self) {
        for engine in &mut self.engines {
            engine.reset();
        }
    }

    /// Drops the grain every engine measured so far.
    pub(crate) fn discard_grain(&mut self) {
        for engine in &mut self.engines {
            if let Err(error) = engine.drain_grain_chunks() {
                tracing::warn!(%error, "cannot drop the grain measured while tuning");
            }
        }
    }

    pub(crate) fn specs(&self) -> &[PassSpec] {
        &self.specs
    }

    #[cfg(test)]
    pub(crate) fn engines_mut(&mut self) -> &mut [Box<dyn PassEngine>] {
        &mut self.engines
    }

    pub(crate) fn profile_keys(&self) -> Vec<String> {
        self.specs.iter().map(profile_key).collect()
    }

    pub(crate) fn device(&self) -> &DeviceProfile {
        &self.profile
    }

    pub(crate) fn ledger(&self) -> Option<&Arc<TuneLedger>> {
        self.ledger.as_ref()
    }

    /// The same engines rebuilt on the same ledger, all running `profile`.
    pub(crate) fn alternative(
        &self,
        config: &TuningConfig,
        profile: PipelineProfile,
    ) -> Result<EngineSet, Error> {
        let setup = KernelSetup {
            ledger: self.ledger.clone(),
            profile_override: Some(profile),
        };

        Self::build(&self.device, &self.client, &self.specs, config, &setup)
    }
}

fn scratch_planes(client: &Client, spec: &PassSpec) -> Vec<Handle> {
    let geometry = spec.geometry();
    let bytes = geometry.output.plane_bytes(geometry.pixels()) as usize;
    let count = geometry.channels.count() as usize;

    (0..count).map(|_| client.empty(bytes)).collect()
}

fn scratch_device_planes(handles: &[Handle], width: u32, height: u32) -> Vec<DevicePlane<'_>> {
    handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect()
}
