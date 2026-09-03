hello:

# Prefer `rustfmt +nightly <file>` for targeted edits; this formats the whole workspace.
format:
    cargo +nightly fmt --all

# Lints all three crates, matching the feature sets `test-rust` uses. Pass `-- -D warnings` to fail on any lint.
clippy *ARGS:
    cargo clippy -p av-denoise-core --features vulkan --all-targets {{ARGS}}
    cargo clippy -p av-denoise --features vulkan,binary --all-targets {{ARGS}}
    cargo clippy -p av-denoise-vs --features vulkan --all-targets {{ARGS}}

# Formats the shipped Python, matching the `python` job in the PR checks workflow. scripts/ is excluded.
format-py *ARGS:
    uvx ruff format . {{ARGS}}

# Lints the shipped Python, matching the `python` job in the PR checks workflow. scripts/ is excluded.
lint-py *ARGS:
    uvx ruff check . {{ARGS}}

# Type-checks the VapourSynth wrapper. `--no-project` skips building the setuptools-rust extension, which its types do not need.
typecheck-py:
    uv run --with mypy --with vapoursynth --no-project \
        mypy --config-file packages/vs-avd/pyproject.toml packages/vs-avd/src/vsavd

build *ARGS:
    cargo build -p av-denoise {{ARGS}}

run *ARGS:
    cargo run -p av-denoise {{ARGS}}

bench *ARGS:
    cargo bench -p av-denoise-core {{ARGS}}

build-vs *ARGS:
    cargo build -p av-denoise-vs --release {{ARGS}}

test-vs: build-vs
    uv run av-denoise-vs/tests/vs_harness.py

build-wheel *ARGS:
    uv build --wheel packages/vs-avd {{ARGS}}

# Runs every Rust and Python test. Needs an accelerator, both sides render.
test: test-rust test-py

# Every Rust test across the three crates.
test-rust:
    cargo nextest run -p av-denoise-core --features vulkan
    cargo nextest run -p av-denoise --features vulkan,binary
    cargo nextest run -p av-denoise-vs --features vulkan
    cargo test --doc -p av-denoise-core --features vulkan
    cargo check --workspace

# Every Python test, including the ones that render on the GPU.
test-py: _rebuild-vs-plugin
    uv run --directory packages/vs-avd --group test pytest tests

# The Python tests that run without an accelerator.
test-py-fast: _rebuild-vs-plugin
    uv run --directory packages/vs-avd --group test pytest tests -m "not gpu"

# `uv run` does not rebuild the cdylib after a Rust change, and the plugin is an
# editable install, so the tests would otherwise pass against a stale build. Cargo is
# incremental, so this costs about a second when nothing has moved.
_rebuild-vs-plugin:
    uv sync --directory packages/vs-avd --group test --reinstall-package vsavd

# End-to-end throughput benchmark. Runs every variant in scripts/configs/benchmark_e2e.toml
# and reports wall-clock timings and amortized fps per group. The variants run
# `target/release/av-denoise` directly, so compiling is never timed.
benchmark-e2e *ARGS: _build-benchmark-bin
    uv run --directory scripts src/benchmark_e2e.py {{ARGS}}

_build-benchmark-bin:
    cargo build --release --bin av-denoise --features vulkan,binary
