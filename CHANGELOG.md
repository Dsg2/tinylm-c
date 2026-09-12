# Changelog

## Unreleased - 2026-09-12

- Add `topk` field to `Cfg` struct and `g_cfg_topk` resolved cache to track the MoE routing K baked into the checkpoint.
- Add `note_cfg_topk()` callback so each loader call updates the resolved cache, resetting the unresolved cache on a second model load in the same process.
- Update `moe_topk()` to prefer the env override `CUDALM_TOPK`, then fall back to `g_cfg_topk` (set by the loader), then default to 1.
- Extend `hdr_ints()` to recognize TLM7 magic and return 16 ints (including topk), and extend `cfg_from_hdr()` to read `topk` from the header and call `note_cfg_topk()`.
- Update `model_load()` and `model_load_full()` to handle TLQ4 (version 4) with 16 header ints and the new topk field.
- Update `gguf_load_full()` to parse the `tinylm.topk` key and store it in `topk_kv`, then apply it to `c->topk` and `note_cfg_topk()` after config loading.
- Update `load_q4_lowpeak()` to handle TLQ4 (version 4) with 16 header ints.
- Update `cmd_score`, `cmd_run`, `cmd_chat`, `cmd_quantize` to recognize TLQ4 magic and use the correct header size for quantized models.
- In `cmd_quantize`, bake the MoE routing K from env or checkpoint into the output when known, producing a TLQ4 header for self-describing MoE checkpoints.

## Unreleased - 2026-09-04

- `tinylm.c:2030` (sampling path) and `tinylm.c:2366` (cmd_chat) now read `gn->logits` directly instead of calling `mm_bt` to recompute the full fp32 head.
- `tinylm.c:1967` (`main_argmax`) similarly reads `gn->logits + n_acc*V` rather than recomputing the head.
- This eliminates the 1.09 ms/token fp32 head recomputation (21% of decode time) and is projected to yield ~+25% speedup.
- A/B test with `bench_lm.py` is planned to verify the numerics change is safe.

## Unreleased - 2026-08-26

- Replace libgomp OpenMP parallel regions on the decode path with a persistent spin-wait thread pool (`tl_for` / `tlp_*` in tinylm.c) — the pool is live only between `gen_new` and exit, training keeps OpenMP for amortised batch work. Measured 2.4x slower at 8 threads (146 → 95 tok/s) vs 3.8x faster with the pool (38.4 → 10.2 µs per GEMV call).
- Deduplicate the fp32 lm_head recomputation in `forward_chunk` and `main_argmax` — `gn->logits` is now read directly from the int8 head, eliminating the 16.8 MB fp32 embedding stream that was 21% of decode wall time.
- Improve top-k sampling from a 40x8192 selection sort to a size-k min-heap with a preallocated scratch buffer, reducing per-token work from ~327k compares plus malloc/memcpy to a heap of 40 elements.
- Add `TINYLM_THREADS` env var to disable the pool entirely, and guard `g_pool_nt` so decode-only kernels have no pragma left at all.
- Add `TINYLM_KV=q8` and `TINYLM_KVSPLIT` config options to the parser, and update `README.md` and `DECODE_PLAN.md` with the new decode performance findings.

