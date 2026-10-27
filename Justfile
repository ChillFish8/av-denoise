set dotenv-load

# Feature set for the end-to-end benchmark binary. Override to build against another
# backend, e.g. `AVD_BENCH_FEATURES="vulkan,cuda" just benchmark-e2e`. The `binary`
# feature is always added, without it Cargo skips the CLI target.
bench_features := env("AVD_BENCH_FEATURES", "vulkan") + ",binary"

# The vcpkg triplet `static-deps` installs for. Override on Intel macOS.
vcpkg_triplet := env("AVD_VCPKG_TRIPLET", if os() == "macos" { "arm64-osx-release" } else { "x64-linux-release" })
static_features := if os() == "macos" { "metal" } else { "vulkan" }

hello:

# Prefer `rustfmt +nightly <file>` for targeted edits; this formats the whole workspace.
format:
    cargo +nightly fmt --all

# The backend comes first and defaults to vulkan, so `just clippy -- -D warnings` lints Vulkan and
# `just clippy rocm -- -D warnings` lints ROCm.
#
# Lints all three crates on one backend, matching the feature sets `test-rust` uses.
clippy backend="vulkan" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    backend="{{backend}}"
    args=({{ARGS}})
    case "$backend" in
        -*) args=("$backend" "${args[@]}"); backend=vulkan ;;
    esac
    features=(--no-default-features --features "$backend")
    host_features=(--no-default-features --features "$backend,binary")
    cargo clippy -p av-denoise-core "${features[@]}" --all-targets "${args[@]}"
    cargo clippy -p av-denoise "${host_features[@]}" --all-targets "${args[@]}"
    cargo clippy -p av-denoise-vs "${features[@]}" --all-targets "${args[@]}"

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

# Builds the static ffms2 and FFmpeg tree from vcpkg.json into vcpkg_installed/.
static-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d .vcpkg ]; then
        git clone https://github.com/microsoft/vcpkg.git .vcpkg
    fi
    git -C .vcpkg fetch --quiet origin
    baseline="$(jq -r '."builtin-baseline"' vcpkg.json)"
    git -C .vcpkg checkout --quiet "$baseline"
    if [ ! -x .vcpkg/vcpkg ]; then
        .vcpkg/bootstrap-vcpkg.sh -disableMetrics
    fi
    .vcpkg/vcpkg install --triplet {{vcpkg_triplet}} --host-triplet {{vcpkg_triplet}}

# Builds a release CLI with ffms2 and FFmpeg linked statically. Run `static-deps` first.
build-static features=static_features:
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="$PWD/vcpkg_installed/{{vcpkg_triplet}}"
    export FFMS_INCLUDE_DIR="$prefix/include"
    export FFMS_LIB_DIR="$prefix/lib"
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
    cargo build --release -p av-denoise --features {{features}},static-ffms2

# The backend comes first and defaults to vulkan, so `just bench -- --device discrete:1` and
# `just bench rocm --bench bench_kernels` both work.
#
# Runs the core benches on one backend.
bench backend="vulkan" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    backend="{{backend}}"
    args=({{ARGS}})
    case "$backend" in
        -*) args=("$backend" "${args[@]}"); backend=vulkan ;;
    esac
    export AVD_TEST_ACCELERATOR="$backend"
    cargo bench -p av-denoise-core --no-default-features --features "$backend" "${args[@]}"

# Benchmarks the host layer, end to end through `HostDenoiser` and `PlanarDenoiser`.
bench-host *ARGS:
    cargo bench -p av-denoise --features vulkan {{ARGS}}

build-vs *ARGS:
    cargo build -p av-denoise-vs --release {{ARGS}}

test-vs: build-vs
    uv run av-denoise-vs/tests/vs_harness.py

build-wheel *ARGS:
    uv build --wheel packages/vs-avd {{ARGS}}

# Runs every Rust and Python test. Needs an accelerator, both sides render.
test: test-rust test-py

# Tests run on `AVD_TEST_DEVICE`, in the `--device` syntax, or on the backend's default device.
#
# Every Rust test across the three crates. Another backend, for example `just test-rust rocm`, runs
# the core crate's backend-generic tests on it instead. ROCm runs at most 8 test processes, because
# each HIP process takes copy queues from a small fixed pool.
test-rust backend="vulkan":
    #!/usr/bin/env bash
    set -euo pipefail
    export AVD_TEST_ACCELERATOR={{backend}}
    if [ "{{backend}}" != vulkan ]; then
        threads=()
        if [ "{{backend}}" = rocm ]; then threads=(--test-threads 8); fi
        cargo nextest run --release -p av-denoise-core --no-default-features --features {{backend}} "${threads[@]}"
        exit 0
    fi
    cargo nextest run --release -p av-denoise-core --features vulkan
    cargo nextest run --release -p av-denoise --features vulkan,binary
    cargo nextest run --release -p av-denoise --features vulkan,binary --profile tuning
    cargo nextest run --release -p av-denoise-vs --features vulkan
    cargo test --doc -p av-denoise-core --features vulkan
    cargo test --doc -p av-denoise --features vulkan
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
benchmark-e2e *ARGS: _build-benchmark-bin _rebuild-benchmark-vs-plugin
    uv run --directory scripts src/benchmark_e2e.py {{ARGS}}

_build-benchmark-bin:
    cargo build --release --bin av-denoise --features {{bench_features}}

# The benchmark's plugin arms take vsavd from `packages/vs-avd` as an editable
# install, and `uv run` does not rebuild the cdylib after a Rust change, so they
# would otherwise measure a stale build against a freshly built CLI. Cargo is
# incremental, so this costs about a second when nothing has moved.
#
# setuptools-rust builds editable installs in debug, so the profile is forced to
# release here, matching the CLI arms.
_rebuild-benchmark-vs-plugin:
    SETUPTOOLS_RUST_CARGO_PROFILE=release uv sync --directory scripts --group vs --reinstall-package vsavd
