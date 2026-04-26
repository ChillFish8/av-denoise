hello:

format:
    cargo +nightly fmt --all

build *ARGS:
    cargo build {{ARGS}}

run *ARGS:
    cargo run {{ARGS}}

bench *ARGS:
    cargo bench {{ARGS}}
