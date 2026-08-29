//! VapourSynth plugin exposing av-denoise as `avd` filters.

mod filter;

use anyhow::Error;
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FilterArgument, Metadata};
use vapoursynth::prelude::{API, Node};
use vapoursynth::{export_vapoursynth_plugin, make_filter_function};

use crate::filter::Passthrough;

make_filter_function! {
    PassthroughFunction, "Passthrough"

    fn create_passthrough<'core>(
        _api: API,
        _core: CoreRef<'core>,
        clip: Node<'core>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        filter::raise_stack_limit();
        Ok(Some(Box::new(Passthrough::new(clip))))
    }
}

export_vapoursynth_plugin! {
    Metadata {
        identifier: "com.chillfish8.avdenoise",
        namespace: "avd",
        name: "av-denoise",
        read_only: true,
    },
    [PassthroughFunction::new()]
}
