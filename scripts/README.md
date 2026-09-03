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
to a container probe of the input. fps divides that count by the wall
clock, so startup, shader compilation, scene detection and the sink all
count against it.

See `benchmark_e2e.toml` for the config the recipe runs by default.
