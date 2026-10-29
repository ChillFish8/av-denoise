mod census;
mod legacy_keys;
mod mode;
mod profile_order;
mod report;
mod setup;
mod warm_up;

#[cfg(feature = "vulkan")]
mod driver;
#[cfg(feature = "vulkan")]
mod placement;
