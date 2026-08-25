# Changelog

## Unreleased - 2026-08-26

- Replace libgomp OpenMP parallel regions on the decode path with a persistent spin-wait thread pool (`tl_for` / `tlp_*` in tinylm.c) — the pool is live only between `gen_new` and exit, training keeps OpenMP for amortised batch work. Measured 2.4x slower at 8 threads (146 → 95 tok/s) vs 3.8x faster with the pool (38.4 → 10.2 µs per GEMV call).
- Deduplicate the fp32 lm_head recomputation in `forward_chunk` and `main_argmax` — `gn->logits` is now read directly from the int8 head, eliminating the 16.8 MB fp32 embedding stream that was 21% of decode wall time.
- Improve top-k sampling from a 40x8192 selection sort to a size-k min-heap with a preallocated scratch buffer, reducing per-token work from ~327k compares plus malloc/memcpy to a heap of 40 elements.
- Add `TINYLM_THREADS` env var to disable the pool entirely, and guard `g_pool_nt` so decode-only kernels have no pragma left at all.
- Add `TINYLM_KV=q8` and `TINYLM_KVSPLIT` config options to the parser, and update `README.md` and `DECODE_PLAN.md` with the new decode performance findings.

