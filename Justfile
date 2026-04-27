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
    if [[ "{{hdr}}" == "true" ]]; then
        pix_fmt="yuv420p10le"
        hdr_flag="--hdr"
    else
        pix_fmt="yuv420p"
        hdr_flag=""
    fi
    ffmpeg -hide_banner -loglevel error -i "{{input}}" -f rawvideo -pix_fmt "${pix_fmt}" - | cargo run --release --bin main -- {{ARGS}} --width "{{width}}" --height "{{height}}" ${hdr_flag}

bench *ARGS:
    cargo bench {{ARGS}}
