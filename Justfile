hello:

format:
    cargo +nightly fmt --all

build *ARGS:
    cargo build {{ARGS}}

run *ARGS:
    cargo run {{ARGS}}

bench *ARGS:
    cargo bench {{ARGS}}

test:
    cargo nextest run --features cpu,binary
    cargo test --doc --features cpu,binary
    cargo check

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

[arg("input", long="input", short="i")]
[arg("output", long="output", short="o")]
[arg("workers", long="workers", short="w")]
denoise-file input output workers="2" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo run --release --bin av-denoise --features binary -- {{ARGS}} file --workers {{workers}} --input "{{input}}" \
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
    podman build -t "{{image}}" . 2> >(cat >&3)
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
        --accelerators vulkan,cpu {{ARGS}} \
        file --input "/in/${input_name}" \
        | ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info \
            -y -f yuv4mpegpipe -i - -c:v ffv1 "${output_abs}"
