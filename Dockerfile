FROM archlinux:latest AS builder

RUN pacman -Syu --noconfirm --needed \
    base-devel \
    git \
    rust \
    clang \
    nasm \
    ffms2

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches

RUN cargo build --release --bin av-denoise --no-default-features --features cpu,vulkan,binary


FROM archlinux:latest AS runtime

RUN pacman -Syu --noconfirm --needed \
    ffmpeg \
    ffms2 \
    gcc-libs \
    vulkan-icd-loader \
    vulkan-radeon \
    vulkan-intel

WORKDIR /app

# CPU works everywhere. Vulkan requires host-provided devices and driver/ICD files.
COPY --from=builder /build/target/release/av-denoise /app/av-denoise

ENTRYPOINT ["/app/av-denoise"]
