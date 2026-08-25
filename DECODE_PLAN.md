# tinylm decode speed — analysis and plan

> **Status: Tiers 1–3 implemented** in [tinylm.c](tinylm-c/tinylm.c).
> Measured **120 → 669 tok/s** at the 1024 window with the shipped `-flto`
> binary (interleaved A/B, medians of 3), and **119 → 753** at short context.
> Accuracy identical on arc_easy / piqa / lambada. What shipped, what it was
> worth, and what turned out not to be worth it is in
> [§7 Results](#7-results-what-shipped).

Model: `models/LS-220M-A25M-q8.bin` — TLQ2, V=8192 D=512 L=12 H=8 KV=2 hd=64
F=707 E=16 top-1, T=4096, **win=1024**, n_mtp=0. 220.7M params / ~25M active.
Machine: i5-1135G7 (Tiger Lake, 4C/8T, AVX-512 + **VNNI**), LPDDR4x.
Build: `-O3 -march=native -ffast-math -funroll-loops -fopenmp -mprefer-vector-width=256`.

Baseline: **~146 tok/s** at short context, **~140 tok/s sustained** at the 1024
window (shipped binary at 8 threads: 95-100).

---

## 1. The headline measurement: OpenMP is making decode *slower*

`gen ... 256 0.8 40`, same prompt, only `OMP_NUM_THREADS` varied:

| threads | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| tok/s | **131** | 85 | 116 | 102 | 95 |

One thread wins. `OMP_WAIT_POLICY=active`, `OMP_PROC_BIND=close/spread` and
`GOMP_SPINCOUNT=INFINITE` were all swept — none recovers it (best 4-thread
result 115 tok/s).

It is worse than "threading doesn't help". **libgomp costs ~30% even at one
thread**, because every `#pragma omp parallel for ... if(cond)` on the decode
path still calls `GOMP_parallel` and builds a one-thread team when `cond` is
false. Rebuilding the identical source *without* `-fopenmp`:

| build (1 thread, same prompt) | tok/s | rmsnorm | rope | qkv | attn |
|---|---|---|---|---|---|
| with `-fopenmp` | 146 | 0.075 s | 0.049 s | 0.217 s | 0.124 s |
| without `-fopenmp` | **192** | 0.002 s | 0.001 s | 0.100 s | 0.060 s |

RMSNorm over a 512-float vector was costing ~11.7 us per call. It is ~0.2 us of
arithmetic; the rest was team setup. Confirmed by patching *only* `rmsnorm_fwd`
and `rope_apply` to take a hand-written serial branch at `rows==1`, keeping
`-fopenmp` on: 146 -> **165 tok/s**, with the norm and rope buckets both at
0.000 s.

Sites with the same defect: `rmsnorm_fwd` (`if(rows>1)`), `rope_apply`
(`if(rows>1)`), `mm` (`if(M>1)`), `gemv_q8` (`if(par)`), and the attention loop
(`if(njob>1)`).

## 2. Where the time actually goes

`TINYLM_PROF=1`, plus extra buckets added to a scratch copy
(`_scratch/perf/tp.c`). No-OpenMP build, 210 tokens, 1.092 s = **5.20 ms/token**:

| bucket | ms/tok | % | notes |
|---|---|---|---|
| MoE experts (int8 w1/w3/w2) | 2.00 | 38 | memory bound, see §3 |
| **duplicate fp32 lm_head** | 1.09 | 21 | pure waste, see §4 |
| sampling: top-k + softmax + malloc | 0.78 | 15 | see §5 |
| qkv projections | 0.48 | 9 | 3 GEMVs, 3 activation quantizations |
| int8 lm_head (inside `forward_chunk`) | 0.41 | 8 | |
| o_proj | 0.30 | 6 | |
| attention | 0.29 | 6 | short context; **1.14 ms/tok at win=1024** |
| rmsnorm + rope + kv_store + malloc | 0.05 | 1 | (0.52 ms/tok in the OpenMP build) |

Context scaling, no-OpenMP build, `stop_eot=0`:

| generated | 128 | 512 | 1024 |
|---|---|---|---|
| tok/s | 195 | 195 | 175 |
| attention share | 4.0% | 11.2% | **20.0%** |

Attention saturates at the trained 1024 window, so 175 tok/s is the steady state
for that build (~140 for the shipped one). `TINYLM_KV=q8` is **slower** than the
f16 default at 1024 ctx (172 vs 177) — attention is latency bound at one thread,
not bandwidth bound, so halving cache bytes does not pay yet.

## 3. The int8 GEMV kernel is fine; the parallelism around it is not

Microbenchmark (`_scratch/perf/mb.c`, `mb2.c`): one expert's `w1` GEMV,
D=512 -> F=707, weights drawn from a 12x16-expert pool so nothing caches,
3,072 calls (= 256 tokens x 12 layers).

| scheme | 1 thread | 2 | 4 | 7 | 8 |
|---|---|---|---|---|---|
| `#pragma omp parallel for` | 44.5 us | 73.6 | 84.5 | — | 106.7 us |
| **persistent spin-wait pool** | 38.4 us | 19.1 | 13.2 | **10.2 us** | 10.9 |
| pool, GB/s | 10.6 | 21.3 | 30.8 | **39.8** | 37.3 |

OpenMP: **2.4x slower** at 8 threads. A pool of Win32 threads spinning on an
atomic ticket: **3.8x faster** at 7. Same kernel, same work split. The memory
system delivers ~40 GB/s; one core reaches ~10.

Also tested: accumulating in int32 across all 16 blocks and doing one float
rescale per row instead of 16 — only **+8%** (40.9 vs 44.5 us). The per-block
scale multiply is not the bottleneck, memory is. VNNI (`_mm256_dpbusd_epi32`) is
already compiled in (`__AVX512VNNI__` and `__AVX512VL__` are both defined by
`-march=native` here), so that lever is already taken.

## 4. The duplicate fp32 lm_head

`forward_chunk` ends with `mmbt_dec(gn->logits, gn->fn, ..., qemb, ...)` — the
full V x D head, in int8. `run_generation` then throws that away:

```c
/* tinylm.c:2030 - sampling path */
memset(gn->hlog,0,sizeof(float)*c->V); mm_bt(gn->hlog,fn_cur,w->emb,1,c->V,c->D);
```

That recomputes the same head in **fp32**, streaming all 16.8 MB of `w->emb`
every token — more bytes than the twelve int8 layers put together. Measured at
**1.09 ms/token, 21% of wall**. The same duplication is at `tinylm.c:2366`
(`cmd_chat`) and `tinylm.c:1967` (`main_argmax`, used by the speculative path).
`cmd_score` already does the right thing and reads `gn->logits`
(`tinylm.c:2171`).

This is the one-line-ish fix worth ~+25% on its own.

## 5. Sampling

Per token, for `topk=40` over V=8192, `run_generation` does: a `malloc(V*4)`, a
`memcpy(V*4)`, a **selection sort** — 40 passes over 8,192 floats, ~327k
compares — then three more full-V passes (threshold, max, exp/sum) and an
`fflush`. Measured ~0.78 ms/token, 15% of wall.

## 6. Byte budget and the ceiling

Per decoded token, at win=1024:

| stream | bytes |
|---|---|
| wq/wk/wv/wo, 12 layers | 7.9 MB |
| one expert x 12 layers (2·D·F + F·D) | 13.0 MB |
| q8 block scales (1 float per 32 weights) | 2.6 MB |
| fp32 routers | 0.4 MB |
| lm_head (V·D + scales) | 4.7 MB |
| KV cache, f16, 1024 positions x 12 layers | 6.3 MB |
| **total** | **~34.9 MB** (28.6 MB at short context) |

At the measured 39.8 GB/s pool ceiling that is **0.88 ms/token -> ~1,140 tok/s**
at full window, ~1,390 at short context. Today: 140. **Roughly 8x is on the
table** before any change to precision or architecture.

## 7. Results (what shipped)

Interleaved A/B, `gen ... 1024 0.8 40 0`, three runs each alternating builds so
thermal drift cancels. Both builds compiled with the same flags from the same
tree; "base" is the pre-change source.

| build | n=256 | n=1024 (full window) |
|---|---|---|
| base | 119 | 123 |
| after (same flags) | **753** | **698** |
| after, shipped `-flto` binary | 725 | **669** |
| | ~6.3× | **5.6-5.7×** |

Two separate interleaved passes, hence two n=1024 base figures (123 and 120);
the spread between them is the run-to-run noise on this laptop. Against the
~140 tok/s sustained figure this started from: **4.8×**.

Profile after, at n=1024 (1.72 s for 1024 tokens = 1.68 ms/token):

| bucket | share | was |
|---|---|---|
| MoE experts | 41.4% | 38% |
| attention | 26.2% | 20% (serial), 49% (pool, before the KV split) |
| qkv | 12.4% | 9% |
| o_proj | 11.5% | 6% |
| lm_head | 4.8% | 8% |
| other | 3.7% | 36% |

The duplicate fp32 head and the sampling loop are gone from the profile
entirely — "other" went from a third of decode to under 4%.

**Quality: unchanged.** 200 items per task, same model, base vs after:

| task | acc | acc_norm |
|---|---|---|
| arc_easy | 39.5% / 39.5% | 34.5% / 34.5% |
| piqa | 58.0% / 58.0% | 60.5% / 60.5% |
| lambada | 19.0% / 19.0% | — |

Identical on every task, lambada included — and lambada is greedy exact-match
generation, so it exercises the decode path end to end. Sampled text does
diverge token-for-token, because sampling now reads the int8 head rather than
the fp32 one (§1.1); isolating it showed the KV split is not the cause.

### Two plan items that did not survive contact

- **§3.4 `TINYLM_KV=q8`: still a loss.** The prediction was that making
  attention bandwidth-bound would flip its sign. It did not — 648 tok/s against
  664 for f16 at the 1024 window. The f16 path is vectorised through F16C
  (`_mm256_cvtph_ps`); the q8 path pays a per-row scale multiply for bytes that
  were not the constraint. f16 stays the default.
- **The KV split had to change default, or attention did not parallelize at
  all.** With `TINYLM_KVSPLIT` off (the old default) there are only `KV`=2 jobs
  at m=1, which is fewer than the pool's 7 threads, so `tl_for` ran attention
  serially — and *slower* than the old serial build, because six idle workers
  were spinning next to it. Attention at the 1024 window: 1.511 s → 0.335 s
  once the split defaulted on. This was not in the plan; it only showed up
  because the pool made the job count matter.

---

## Plan, in order of payoff per unit of risk

### Tier 1 — no new machinery

**1.1 Reuse the int8 logits instead of recomputing the head in fp32.** *(done)*
`tinylm.c:2030` and `tinylm.c:2366`: drop the `mm_bt` and sample from
`gn->logits + (m-1)*V` (row `m-1` after a prefill chunk, row 0 after a
single-token step). `main_argmax` at `tinylm.c:1967` likewise reads
`gn->logits + n_acc*V`. *Projected +25%.* Caveat: sampling then sees q8-head
logits rather than fp32 ones — a numerics change to sampled output, though it is
the same head that `cmd_score` and the greedy/speculative paths already use. A/B
with `bench_lm.py` before shipping.
Follow-on: `gen_drop_fp32` keeps 16.8 MB of fp32 `emb` alive *only* for these
call sites plus the input row lookup. Dequantize the single needed row from
`qemb` and the fp32 copy can go entirely.

**1.2 Delete every degenerate OpenMP region on the m==1 path.** *(done — superseded by §2.1: the decode-only kernels have no pragma left at all, and the four shared with training take the pool when it is live)* Give
`rmsnorm_fwd`, `rope_apply`, `mm`, `gemv_q8` and the attention loop a
hand-written serial branch instead of an `if(cond)` clause on the pragma — the
clause avoids the extra threads, not the libgomp call. *Measured +13% from two
of the five sites; the full set should be worth ~+30%.*

**1.3 Rewrite top-k.** *(done)* Quickselect (or a size-k min-heap) instead of the
40x8192 selection sort, a preallocated scratch buffer instead of the per-token
`malloc`/`memcpy`, and fuse the threshold/max/exp passes into one sweep.
*Projected +10%.*

Tier 1 alone: ~5.20 -> ~3.3 ms/token, **~300 tok/s**, still single-threaded,
with only the §1.1 caveat as a numerics change.

### Tier 2 — the real win: a persistent thread pool

**2.1 Replace libgomp on the decode path with a spin-wait pool.** *(done — `tl_for` / `tlp_*` in tinylm.c)* N-1 Win32 (or
pthread) threads created once in `gen_new`, parked on an atomic ticket, each
claiming a contiguous output-row range; the caller runs range 0 itself and spins
on a done-counter. `_scratch/perf/mb2.c` is a working ~60-line prototype.
Parallelize exactly the row-parallel GEMVs and the attention job loop:
`gemv_q8`, the fused w1/w3 sweep and w2 in `moe_forward_q8`, `mmbt_dec`, and the
`(row, kv-head, chunk)` attention loop. Everything else stays serial. Keep
OpenMP for training — this is a decode-only pool.
*Projected: the ~3.5 ms/token of GEMV + attention work at 3.5x -> ~1.0 ms; total
~1.3 ms/token short-context and ~1.6 ms at win=1024 -> **600-800 tok/s**.*

**2.2 Bound the risk.** *(done — workers pause, then yield, then nap; `TINYLM_THREADS=1` disables the pool)* A spin pool burns cores while idle. Park after N spins
(`WaitOnAddress`), and expose `TINYLM_THREADS` so 1 disables the pool entirely.

### Tier 3 — fewer bytes and fewer passes

*(done — `qmat_KN3`)* **3.1 Fuse Q, K, V into one output-major matrix** `[D+2·KD][D]`. One GEMV, one
dispatch, and — the part that matters — **one** `q8_row` of `xn` instead of
three (`gemv_q8` re-quantizes X on every call).

*(done — `qmat_KN_ilv` + `swiglu_q8`, which also fuses the SwiGLU)* **3.2 Interleave w1 and w3 rows** so the fused SwiGLU sweep walks one contiguous
stream instead of two, and add `_mm_prefetch` on the next rows. Expert weights
are cold every token — the prefetcher gets no help across an expert switch.

**3.3 Q4 experts.** *(not done — a precision change, needs a quality call)* The experts are 13.0 MB of the ~34.9 MB/token. Q4_0 with
per-32 scales halves that to ~6.5 MB and lifts the roof from ~1,140 to
~1,700 tok/s. `q4_trial.py` already exists; this is a quality call, not only a
speed one.

**3.4 Revisit `TINYLM_KV=q8`.** *(measured after the pool: still a loss, see §7)* It loses today (172 vs 177 at 1024 ctx) because
attention is latency bound at one thread. Once §2.1 makes attention bandwidth
bound, halving the 6.3 MB KV stream should flip the sign. Re-measure after the
pool lands, not before.

### Tier 4 — capability changes, not optimisations

**4.1 MTP heads.** *(not done — needs training)* The self-speculative decoder is already written and verified
byte-identical to greedy, but this checkpoint has `n_mtp=0`. Training two heads
gives 1.5-2x on greedy decode for ~0.5M extra params. Sampling paths do not
benefit.

**4.2 Batched decode (B>1).** *(not done — a serving change)* Every weight byte above is read once per token
regardless of batch, so decoding 4 sequences at once is nearly free in aggregate
throughput until the kernels turn compute bound. `forward_chunk` is
single-sequence today; this is a serving lever, not a latency one.

---

## Suggested order

1. §1.1 + §1.3 (a morning; measure with `bench_lm.py` for the q8-head A/B)
2. §1.2 (mechanical, measure each site)
3. §2.1 + §2.2 (the big one; `mb2.c` de-risks it)
4. §3.1 + §3.2, then re-measure §3.4
5. §3.3 / §4.1 as capability work

Measurement protocol used throughout: `TINYLM_PROF=1`, fixed prompt, 256+ tokens,
`stop_eot=0` for the context-scaling runs; scratch builds and logs live under
`_scratch/perf/`.
