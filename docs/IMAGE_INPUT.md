# Single-image input

TurboFieldfare can optionally use one image with Gemma 4 26B-A4B IT. The
existing Swift + Metal runtime still handles all text embeddings, language
model prefill, and token generation. MLX Swift loads only the Gemma vision
tower and image projector for the duration of that image request.

## Provide the optional vision weights

The normal `.gturbo` installer remains text-only. To add image support, use
`model-00003-of-00003.safetensors` from the pinned
`mlx-community/gemma-4-26b-a4b-it-4bit` checkpoint at revision
`0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. Run the standard-library-only
extractor against that shard and the completed model directory:

```bash
python3 Scripts/extract_gemma4_vision_weights.py \
  /path/to/model-00003-of-00003.safetensors \
  scratch/gemma4.gturbo
```

It copies only the 358 `vision_tower.*` and `embed_vision.*` tensors in 4 MiB
chunks, writes `vision_weights.safetensors`, and records its SHA-256 in
`vision_weights.sha256`. The extractor checks the tensor count, key shapes,
and a 1.5 GiB sidecar size ceiling. The source checkpoint file stays in place.
The sidecar is optional and kept outside the existing `.gturbo` manifest, so
the text model installation and its v1 wire format are unchanged.

The decode service requires both files and verifies the sidecar checksum
before loading it. Missing, altered, or incompatible weights produce an
“Image support unavailable” error; the attachment is never silently ignored.

## Limits and memory behavior

- One JPEG or PNG image per request; maximum encoded file size is 32 MiB.
- EXIF orientation is applied before resizing. Decoded source dimensions are
  checked before pixel buffers are allocated; the source image limit is 100
  megapixels.
- Gemma image detail can use 70, 140, 280, or 560 soft tokens. The default is
  560 for better small-object and text detail. Choose the budget from the
  Image detail menu after attaching an image. The resize uses 16-pixel patches
  and 3×3 pooling; the actual projected row count is derived from the resized
  patch grid and is included in prompt/context validation.
- Image requests have a maximum 4,096-token context. Image rows use Gemma's
  bidirectional block attention during custom Metal prefill.
- Image work is admitted only when the decode-service physical footprint leaves
  at least 2 GiB below the per-process budget. The budget is half of installed
  RAM, capped at 6 GiB. MLX is limited to 3 GiB with its reusable-buffer cache
  disabled. The combined process footprint is sampled during vision and
  language prefill; work stops 512 MiB below the budget.

These are application guardrails, not an operating-system memory limit. They
do not establish a fixed peak for every image budget or machine. A
representative measurement with the real sidecar and image is still required
before making that claim.
