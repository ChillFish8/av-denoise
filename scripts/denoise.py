#!/usr/bin/env python3
import time

import click
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from PIL import Image

TILE_SIZE = 64


def load_session(model_path: str) -> ort.InferenceSession:
    """Load the ONNX model with a dynamic batch dimension."""
    model = onnx.load(model_path)
    # The model was exported with batch=1. Relax that to a symbolic dim so
    # onnxruntime accepts any batch size.
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
    print(model.graph.input, model.graph.output)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0   # 0 = use all cores within a single op
    opts.inter_op_num_threads = 1   # 0 = parallelise independent graph nodes
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    return ort.InferenceSession(model.SerializeToString(), sess_options=opts)


def tile_mask(overlap: int) -> np.ndarray:
    mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for i in range(overlap):
        ramp = i / overlap
        reverse = (overlap - 1 - i) / overlap
        mask[i, :] *= ramp
        mask[TILE_SIZE - overlap + i, :] *= reverse
        mask[:, i] *= ramp
        mask[:, TILE_SIZE - overlap + i] *= reverse
    return mask


def denoise(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    overlap = 16
    step = TILE_SIZE - overlap * 2

    pad_h = (TILE_SIZE - height % step) % step
    pad_w = (TILE_SIZE - width % step) % step

    padded = np.pad(
        image,
        [(overlap, overlap + pad_h), (overlap, overlap + pad_w), (0, 0)],
        mode="reflect",
    )
    padded_h, padded_w = padded.shape[:2]

    mask = tile_mask(overlap)
    tiles = []
    regions = []

    for y in range(0, padded_h - TILE_SIZE + 1, step):
        for x in range(0, padded_w - TILE_SIZE + 1, step):
            tiles.append(padded[y : y + TILE_SIZE, x : x + TILE_SIZE, :].transpose(2, 0, 1))

            oy = y - overlap
            ox = x - overlap
            dy0 = max(0, oy)
            dy1 = min(oy + TILE_SIZE, height)
            dx0 = max(0, ox)
            dx1 = min(ox + TILE_SIZE, width)
            sy0 = dy0 - oy
            sy1 = sy0 + (dy1 - dy0)
            sx0 = dx0 - ox
            sx1 = sx0 + (dx1 - dx0)
            regions.append((dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1))

    batch = np.stack(tiles, axis=0)  # [N, 3, 64, 64]

    start = time.perf_counter()
    results = session.run(["output"], {"input": batch})[0]  # [N, 3, 64, 64]
    elapsed = time.perf_counter() - start
    print(f"Took: {elapsed:.2}s tiles:{len(batch)}")

    output = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    mask3 = mask[:, :, np.newaxis]

    for result, (dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1) in zip(results, regions):
        r = result.transpose(1, 2, 0)  # [64, 64, 3]
        output[dy0:dy1, dx0:dx1] += r[sy0:sy1, sx0:sx1] * mask3[sy0:sy1, sx0:sx1]
        weight[dy0:dy1, dx0:dx1] += mask3[sy0:sy1, sx0:sx1]

    return output / np.maximum(weight, 1e-8)


@click.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_image", type=click.Path(dir_okay=False))
@click.option("--model", default="models/rt_ldr_small.onnx", show_default=True, help="Path to ONNX model.")
def main(input_image: str, output_image: str, model: str) -> None:
    session = load_session(model)

    img = Image.open(input_image).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    for _ in range(0, 30):
        result = denoise(session, arr)

    out = Image.fromarray((np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8))
    out.save(output_image)
    click.echo("Saved to " + output_image)


if __name__ == "__main__":
    main()
