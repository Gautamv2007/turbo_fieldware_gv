#!/usr/bin/env python3
"""Write a bounded-memory Gemma 4 vision-only safetensors sidecar."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

EXPECTED_TENSORS = 358
MAX_OUTPUT_BYTES = 1536 * 1024 * 1024
CHUNK_BYTES = 4 * 1024 * 1024
PREFIXES = ("vision_tower.", "embed_vision.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_shard", type=Path,
                        help="model-00003-of-00003.safetensors from the pinned MLX checkpoint")
    parser.add_argument("model_directory", type=Path,
                        help="completed .gturbo directory for the output sidecar")
    args = parser.parse_args()
    source = args.source_shard
    output_dir = args.model_directory
    if not source.is_file() or not output_dir.is_dir():
        parser.error("source shard and completed model directory must exist")

    with source.open("rb") as src:
        prefix = src.read(8)
        if len(prefix) != 8:
            parser.error("source shard has a truncated safetensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > 64 * 1024 * 1024:
            parser.error("source shard has an invalid safetensors header size")
        try:
            header = json.loads(src.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            parser.error(f"source shard header is invalid: {error}")
        data_start = 8 + header_size
        selected = {name: info for name, info in header.items() if name.startswith(PREFIXES)}
        if len(selected) != EXPECTED_TENSORS:
            parser.error(f"expected {EXPECTED_TENSORS} pinned vision tensors, found {len(selected)}")
        for name, shape in (
            ("vision_tower.patch_embedder.input_proj.weight", [1152, 768]),
            ("embed_vision.embedding_projection.weight", [2816, 144]),
        ):
            if name not in selected or selected[name].get("shape") != shape:
                parser.error(f"source checkpoint is incompatible at {name}")
        if selected["vision_tower.patch_embedder.input_proj.weight"].get("dtype") != "BF16":
            parser.error("vision tower dtype is not BF16")
        if selected["embed_vision.embedding_projection.weight"].get("dtype") != "U32":
            parser.error("vision projector is not the pinned 4-bit U32 weight")

        new_header = {}
        cursor = 0
        for name in sorted(selected):
            info = selected[name]
            dtype = info.get("dtype")
            shape = info.get("shape")
            if dtype not in ("BF16", "U32") or not isinstance(shape, list) or not shape \
                    or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape):
                parser.error(f"unsupported dtype or shape for {name}")
            if dtype == "U32" and name != "embed_vision.embedding_projection.weight":
                parser.error(f"unexpected packed tensor dtype for {name}")
            offsets = info.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                parser.error(f"invalid data offsets for {name}")
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                parser.error(f"invalid data offsets for {name}")
            count = end - start
            expected_bytes = 4 if dtype == "U32" else 2
            for dimension in shape:
                expected_bytes *= dimension
            if count != expected_bytes:
                parser.error(f"tensor byte length does not match its shape for {name}")
            cursor += count
            if cursor > MAX_OUTPUT_BYTES:
                parser.error("vision tensors exceed the 1.5 GiB sidecar safety limit")
            new_header[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [cursor - count, cursor],
            }
        serialized = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        serialized += b" " * ((-len(serialized)) % 8)

        weights_path = output_dir / "vision_weights.safetensors"
        checksum_path = output_dir / "vision_weights.sha256"
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".vision-weights-", delete=False) as tmp:
            temporary_path = Path(tmp.name)
            digest = hashlib.sha256()
            encoded_prefix = struct.pack("<Q", len(serialized))
            tmp.write(encoded_prefix)
            tmp.write(serialized)
            digest.update(encoded_prefix)
            digest.update(serialized)
            for name in sorted(selected):
                start, end = selected[name]["data_offsets"]
                src.seek(data_start + start)
                remaining = end - start
                while remaining:
                    block = src.read(min(CHUNK_BYTES, remaining))
                    if not block:
                        parser.error(f"tensor data is truncated for {name}")
                    tmp.write(block)
                    digest.update(block)
                    remaining -= len(block)
    os.replace(temporary_path, weights_path)
    checksum_path.write_text(digest.hexdigest() + "\n", encoding="ascii")
    print(f"Wrote {weights_path} ({cursor:,} tensor bytes)")
    print(f"SHA-256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
