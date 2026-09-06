# Helper scripts

Does what it says on the tin.

## `src/benchmark_e2e.py`

End-to-end throughput benchmark. It reads a TOML config, runs every
variant in it, and reports wall-clock timings and amortized fps per
group.

```
just benchmark-e2e
just benchmark-e2e --group 1080p --only nl4d_base --repeats 3 --warmup
just benchmark-e2e --dry-run
```

A variant is a shell command. `{{input}}` is replaced with the input its
group names and `{{output}}` with a per-variant file under `output_dir`,
which is a temporary directory when the config leaves it out. Groups
exist so one set of commands can be measured against several inputs, such
as a 1080p clip and a 4K one.

Frame counts come from the `frame=` progress a run prints, and fall back
to a container probe of the input, which is what the checked-in config
relies on because its variants print no progress. fps divides that count
by the wall clock, so startup, shader compilation and scene detection all
count against it.

A variant may also carry an `env` table, whose entries are added to the
environment its command runs in. The checked-in config pins no GPU. Each
stack reads its card from a variable, and all three count in PCI order, so
one index selects one card across the lot:

```
AVD_DEVICE=discrete:1 BENCH_OCL_DEVICE=1 BENCH_HIP_DEVICE=1 just benchmark-e2e
```

`AVD_DEVICE` goes to av-denoise through Vulkan, `BENCH_OCL_DEVICE` is the
device half of ffmpeg's `ocl:0.N`, and `BENCH_HIP_DEVICE` is V-BM3D's HIP
index. A single variant can override any of them through `env`.

`AVD_DEVICE` reaches the CLI arms as an environment variable, which is
what the binary reads. The plugin does not read it, so the plugin arms
pick it up in the shell and pass it on as `--arg device=`.

The VapourSynth variants run under vspipe on its own defaults, no
`--requests` flag, so they measure what a VapourSynth user gets rather
than an idealised sequential render. They need the `vs` dependency group,
which `uv run --group vs` installs on first use, so nothing has to be set
up by hand.

- `vs/avd_denoise.vpy` runs av-denoise through the `vsavd` plugin, with
  `--arg algo=` picking `nl4d`, `nlmhq` or `nlm`. The `vs` group takes
  `vsavd` from `packages/vs-avd` rather than PyPI, so the plugin arms and
  the CLI arms measure the same working tree. `just benchmark-e2e`
  reinstalls it, so a Rust change is never measured against a stale build.
- `vs/vbm3d_denoise.vpy` runs the V-BM3D reference, with `--arg profile=`
  picking one of its profiles. It picks its own backend by rendering a
  probe frame through each GPU BM3D plugin in turn, so it runs on HIP on an
  AMD box and CUDA on an NVIDIA one. `--arg backend=` forces one.

See `configs/benchmark_e2e.toml` for the config the recipe runs by
default. It measures ten variants against a 1080p clip and nine against a
4K one.
