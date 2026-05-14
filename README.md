# av-denoise

Fast and efficient NLMEANS video denoising using CubeCL.

As mentioned this project uses [CubeCL](https://github.com/tracel-ai/cubecl) which is a JIT compiling compute
language framework. In essense, our CubeCL kernels can be JIT compiled on demand for the host CPU or GPU applying
host-specific optimisations rather than relying on specialised kernels for each arcitecture.

## Supported accelerators

- CPU (via `cpu`)
- AMD GPUs (via `rocm` or `vulkan`)
- Nvidia GPUs (via `cuda` or `vulkan`)
- Apple Silicon (via `metal`)