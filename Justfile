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

compare-perf *ARGS:
    uv run scripts/bench_runs.py {{ARGS}}

compare-runs *ARGS:
    uv run scripts/compare_runs.py {{ARGS}}

quality-runs *ARGS:
    uv run scripts/quality_runs.py {{ARGS}}

quality-runs-light *ARGS:
    uv run scripts/quality_runs.py --config scripts/quality_runs_light.toml {{ARGS}}

# Builds data/clean-1080p.mkv, the reference clip for the light-noise harness.
make-clean-source:
    ffmpeg -hide_banner -loglevel error -y -ss 36 -i data/asterisk-war.mkv -frames:v 288 -c:v ffv1 -pix_fmt yuv420p -an -sn data/clean-1080p.mkv

# Builds data/bench-sample-10bit.mkv, the 10-bit input for the bit-depth bench row.
make-10bit-sample:
    ffmpeg -hide_banner -loglevel error -y -i data/bench-sample.mkv -pix_fmt yuv420p10le -c:v ffv1 data/bench-sample-10bit.mkv

[arg("input", long="input", short="i")]
[arg("output", long="output", short="o")]
[arg("workers", long="workers", short="w")]
denoise-file input output workers="2" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo run --release --bin av-denoise --features binary -- nlmeans {{ARGS}} --workers {{workers}} --input "{{input}}" \
        | ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info \
            -y -f yuv4mpegpipe -i - -c:v ffv1 "{{output}}"


[arg("input", long="input", short="i")]
[arg("output", long="output", short="o")]
[arg("patch", long="patch")]
[arg("search", long="search")]
[arg("strength", long="strength")]
denoise-file-ffmpeg input output search="5" patch="9" strength="1.2":
    #!/usr/bin/env bash
    set -euo pipefail
    ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info \
        -init_hw_device opencl=ocl:0.0 -filter_hw_device ocl \
        -y -i "{{input}}" -vf "hwupload,nlmeans_opencl=s={{strength}}:p={{patch}}:pc={{patch}}:r={{search}}:rc={{search}},hwdownload,format=yuv420p" -c:v ffv1 "{{output}}"

# Denoises with ffmpeg's CPU BM3D filter, a slow external quality reference.
# The clip is split across processes because bm3d's own slice threading
# stops scaling around eight threads. `jobs` of 0 picks a count from the
# core count. Sigma is on ffmpeg's own scale rather than the true noise
# sigma, so it needs sweeping per clip. Runs with collaborative filtering
# on (group=16/32, the reference implementation's per-stage sizes), which
# is much slower than ffmpeg's group=1 default.
[arg("input", long="input", short="i")]
[arg("output", long="output", short="o")]
[arg("sigma", long="sigma")]
[arg("jobs", long="jobs", short="j")]
denoise-file-bm3d input output sigma="15" jobs="0":
    @echo "[bm3d] collaborative filtering is on (group=16/32), roughly 30x slower than ffmpeg's group=1 default. This will take a while." >&2
    uv run scripts/bm3d_parallel.py --input "{{input}}" --output "{{output}}" --sigma {{sigma}} --jobs {{jobs}}

[arg("input", long="input", short="i")]
[arg("output", long="output", short="o")]
[arg("image", long="image")]
docker-test-run image="localhost/av-denoise:latest" input="data/test.mkv" output="data/test.denoised.mkv" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec 3>&2
    podman build -t "{{image}}" -f docker/vulkan.Dockerfile . 2> >(cat >&3)
    input_abs="$(realpath "{{input}}")"
    output_abs="$(realpath -m "{{output}}")"
    input_dir="$(dirname "${input_abs}")"
    output_dir="$(dirname "${output_abs}")"
    input_name="$(basename "${input_abs}")"
    mkdir -p "${output_dir}"
    podman run --rm --name av-denoise \
        --device /dev/kfd --device /dev/dri \
        --group-add video --group-add render \
        --security-opt seccomp=unconfined \
        --memory=48g \
        --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host \
        -v "${input_dir}:/in:ro" \
        "{{image}}" \
        --accelerators vulkan \
        nlmeans {{ARGS}} --input "/in/${input_name}" \
        | ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info \
            -y -f yuv4mpegpipe -i - -c:v ffv1 "${output_abs}"
