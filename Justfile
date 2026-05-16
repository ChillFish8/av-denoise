hello:

format:
    cargo +nightly fmt --all

build *ARGS:
    cargo build {{ARGS}}

run *ARGS:
    cargo run {{ARGS}}

bench *ARGS:
    cargo bench {{ARGS}}

[arg("input", long="input", short="i")]
[arg("output", long="output", short="o")]
denoise-file input output *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo run --release --bin av-denoise --features binary-full -- {{ARGS}} file --input "{{input}}" \
        | ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info \
            -y -f yuv4mpegpipe -i - -c:v ffv1 "{{output}}"


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
