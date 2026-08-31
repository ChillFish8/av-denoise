# Bump ARCH_BUILD_IMAGE, ARCH_RUNTIME_IMAGE and ARCH_SNAPSHOT together when refreshing the pin.
ARG ARCH_BUILD_IMAGE=archlinux:base-devel-20260816.0.574111@sha256:714acd1eef9ae997d95691b1c5220ada0076185b77857c1813f02de0fa83cf7b
ARG ARCH_RUNTIME_IMAGE=archlinux:base-20260816.0.574111@sha256:4bf33b21a715aac0b48ce6e9eaed4782a898eae96f88f5da3635572129c2584a
ARG ARCH_SNAPSHOT=2026/08/17

FROM ${ARCH_BUILD_IMAGE} AS builder

ARG ARCH_SNAPSHOT

# cubecl-hip-sys's build script runs hipconfig and links hiprtc and amdhip64, so HIP must be present at build time.
RUN printf 'Server = https://archive.archlinux.org/repos/%s/$repo/os/$arch\n' "${ARCH_SNAPSHOT}" > /etc/pacman.d/mirrorlist \
    && pacman -Syyuu --noconfirm --needed --disable-download-timeout \
        git rust clang nasm ffms2 rocm-hip-runtime \
    && rm -rf /var/cache/pacman/pkg/*

ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib
# hipconfig only reaches PATH through /etc/profile.d/rocm.sh, which a RUN step never sources, so PATH is set explicitly here.
ENV PATH=/opt/rocm/bin:${PATH}

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY README.md ./
COPY av-denoise-core/ ./av-denoise-core/
COPY av-denoise/ ./av-denoise/
COPY av-denoise-vs/ ./av-denoise-vs/

RUN cargo build --release -p av-denoise --no-default-features --features rocm,binary

FROM ${ARCH_RUNTIME_IMAGE} AS runtime

ARG ARCH_SNAPSHOT

RUN printf 'Server = https://archive.archlinux.org/repos/%s/$repo/os/$arch\n' "${ARCH_SNAPSHOT}" > /etc/pacman.d/mirrorlist \
    && pacman -Syyuu --noconfirm --needed --disable-download-timeout \
        ffmpeg ffms2 gcc-libs rocm-hip-runtime \
    && rm -rf /var/cache/pacman/pkg/* \

ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib
# Set explicitly for symmetry with the builder stage, so rocminfo and hipconfig are usable when debugging a container.
ENV PATH=/opt/rocm/bin:${PATH}

WORKDIR /app

COPY --from=builder /build/target/release/av-denoise /app/av-denoise

# The JIT compiles through hiprtc at runtime, and /dev/kfd and /dev/dri come from the host, so this only proves the binary's own dynamic libraries resolve in the runtime image.
RUN /app/av-denoise --help > /dev/null

ENTRYPOINT ["/app/av-denoise"]
