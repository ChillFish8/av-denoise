hello:

format:
    cargo +nightly fmt --all

build *ARGS:
    cargo build {{ARGS}}

run *ARGS:
    cargo run {{ARGS}}

denoise-file input width height hdr="false" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec 3>&2
    if [[ "{{hdr}}" == "true" ]]; then
        pix_fmt="rgb48le"
        hdr_flag="--hdr"
    else
        pix_fmt="rgb24"
        hdr_flag=""
    fi
    ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info -i "{{input}}" -f rawvideo -pix_fmt "${pix_fmt}" - 2> >(cat >&3) | cargo run --release --bin main -- {{ARGS}} --width "{{width}}" --height "{{height}}" ${hdr_flag} 2> >(cat >&3) > /dev/null

denoise-video input output width height fps hdr="false" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec 3>&2
    if [[ "{{hdr}}" == "true" ]]; then
        pix_fmt="rgb48le"
        hdr_flag="--hdr"
        codec_opts=( -c:v ffv1 )
    else
        pix_fmt="rgb24"
        hdr_flag=""
        codec_opts=( -c:v libx264 -crf 18 -preset slow )
    fi
    ffmpeg -hide_banner -stats -stats_period 0.5 -loglevel info -i "{{input}}" -f rawvideo -pix_fmt "${pix_fmt}" - 2> >(cat >&3) | cargo run --release --bin main -- {{ARGS}} --width "{{width}}" --height "{{height}}" ${hdr_flag}

bench *ARGS:
    cargo bench {{ARGS}}

docker-test-run image="localhost/av-denoise:local" input="data/test.mkv" width="1920" height="1080" duration="1" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec 3>&2
    podman build -t "{{image}}" . 2> >(cat >&3)
    ffmpeg -hide_banner \
        -stats \
        -stats_period 0.5 \
        -loglevel info \
        -t "{{duration}}" \
        -i "{{input}}" \
        -f rawvideo \
        -pix_fmt rgb24 \
        - 2> >(cat >&3) | podman run --rm --device /dev/kfd --device /dev/dri \
            --group-add video --group-add render \
            --security-opt seccomp=unconfined \
            --memory=48g \
            --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host \
            -i "{{image}}" \
            --accelerators vulkan,cpu {{ARGS}} \
            --width "{{width}}" \
            --height "{{height}}"
