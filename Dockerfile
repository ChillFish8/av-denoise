FROM archlinux:latest AS builder

RUN pacman -Syu --noconfirm --needed \
    base-devel \
    git \
    rust

WORKDIR /build

COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY models ./models

RUN cargo build --release --bin main --no-default-features --features cpu,vulkan


FROM archlinux:latest AS runtime

RUN pacman -Syu --noconfirm --needed \
    ffmpeg \
    gcc-libs \
    vulkan-icd-loader \
    vulkan-radeon

WORKDIR /app

# CPU works everywhere. Vulkan requires host-provided devices and driver/ICD files.
COPY --from=builder /build/target/release/main /app/main
COPY models/rt_ldr.bpk models/rt_ldr_small.bpk /app/models/

ENTRYPOINT ["/app/main"]
