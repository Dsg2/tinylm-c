# tinylm in C (multithreaded, zero dependencies)

A straight C port of a small PyTorch mini-Llama — same architecture
architecture (RMSNorm + RoPE + GQA/MQA + SwiGLU + tied embeddings), but the
forward pass, **the full backward pass, and AdamW are all hand-written**, and
the hot loops (GEMMs, attention, norms, optimizer) are parallelized with
**OpenMP**. No PyTorch, no BLAS, no Python at runtime — one ~50 KB `.exe`.

## Files
- `tinylm.c` — everything: model, autodiff-by-hand, training, generation, BPE loader.
- `tools/export_tokenizer.py` — one-time helper: dumps the HF `tokenizer.json` into a flat
  `tokenizer.bin` (decode table + byte map + merge ranks) the C code reads.

## Build (MSYS2 MinGW64)
```
# fast (recommended for m16+): OpenBLAS-backed GEMM, ~2x on the big presets
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp -flto -DUSE_BLAS -o tinylm.exe tinylm.c -lopenblas -lm
# portable (zero deps): register-blocked kernels. -mprefer-vector-width=256 avoids
# the Tiger Lake AVX-512 downclock (~12% faster here); drop it on non-throttling CPUs.
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp -flto -mprefer-vector-width=256 -o tinylm.exe tinylm.c -lm
```
`-DUSE_BLAS` routes the three matmul primitives (`X@W`, `X@Wᵀ`, `Xᵀ@W`) through
`cblas_sgemm` (needs `pacman -S mingw-w64-x86_64-openblas`; the `libopenblas.dll`
must be findable at runtime — it is if `C:\msys64\mingw64\bin` is on PATH). The
non-BLAS build uses `restrict` + `-march=native` AVX2/512 FMA kernels and is fully
self-contained.

## Quick start — three commands

```bash
# 0) prepare data ONCE (writes train/val.bin, tokenizer.bin, template.txt)
python tools/prepare_alpaca.py --input alpaca.json --vocab_size 2048   # -> data/alpaca
#   or: python tools/prepare_data.py --vocab_size 512                  # -> data/v512 (TinyStories)

set OMP_NUM_THREADS=8
tinylm.exe new   mychat m3                 # create a ~3M model -> models/mychat.bin
tinylm.exe train mychat data/alpaca 2   # train 2 epochs (Ctrl-C safe; re-run to resume)
tinylm.exe run   mychat -i                 # chat with it (or: run mychat "your prompt")
```

- **`new <name> <preset>`** materializes `models/<name>.bin` (untrained).
- **`train <name> <data> [epochs]`** trains it for N epochs; the model adopts the
  data's tokenizer + chat template on first train, auto-resumes, and checkpoints
  as it goes. If `<name>` is itself a preset, `new` is implied.
- **`run <name> [prompt]`** one-shot, or **`run <name> -i`** interactive. Alpaca
  models wrap your input in the instruction template automatically.

Low-level file-path commands still exist: `gen <model.bin> "<prompt>" ...` and
`pack <in> <tok.bin> <raw|alpaca> <out>`. Run `tinylm.exe help` for everything.

### Presets (params at vocab 2048; V is taken from the data's tokenizer)
```
nano ~0.2M | micro ~0.6M | m1 ~0.9M | m3 ~2.8M | m6 ~6.1M | m8 ~7.6M
```
m1 and below train comfortably on this CPU. **m3–m8 are large for a laptop CPU —
expect a few hours per epoch (less with the OpenBLAS build)**; use a small dataset
or fractional epochs (e.g. `train big data/alpaca 0.2`) to iterate, or train
overnight. They train fine; it's just slow.

### MoE — more capacity at ~the same compute
```
tinylm new mymoe m3 4      # m3 with 4 top-1 routed experts (instead of one FFN)
tinylm train mymoe data/alpaca 1
```
Each layer's SwiGLU FFN is replaced by **E experts + a router**; only the **top-1**
expert runs per token (gather/scatter, so compute ≈ a single dense FFN), with a
load-balance aux loss so experts don't collapse. Since the FFN dominates params,
`m3 4` is roughly an **8M-param model that trains at ~m3 speed** (measured: 4-expert
nano = 2.8× the params at ~88% of dense throughput). Inference is cheap too (one
expert fires). Caveat: at tiny scale MoE realizes maybe 60–80% of the quality of a
true dense model of the same total size, and routing needs the aux loss to balance —
but it's the right lever when you're compute-bound. (MoE is CPU-only; the GPU backend
stays dense.)

### Self-contained checkpoints (one file = weights + tokenizer + template)

`train` bundles the tokenizer and the chat template into every `.bin`, so `gen`
only needs the model and your prompt. For an **alpaca** model you pass just the
instruction and the `### Instruction / ### Response` wrapper is added for you:

```
tinylm.exe gen out/alpaca_m1.bin "Give three tips for staying healthy."
```

The template is chosen by `data_dir/template.txt` (`alpaca` or `raw`, written by
the prepare scripts). To upgrade an **older** checkpoint that lacks a bundled
tokenizer, pack it once:

```
tinylm.exe pack out/old.bin data/alpaca/tokenizer.bin alpaca out/old.bin
```

(The legacy 3-arg form `gen <model> <tok.bin> "<prompt>"` still works too.)

### Sliding KV cache

Generation uses a real KV cache: each new token does a single-token forward and
attends to cached keys/values instead of recomputing the whole window. On this
laptop that took m1 decoding from ~28 tok/s to **~1,200-1,800 tok/s** (tens of x).

It's a **sliding** cache: the K/V live in a ring buffer of the last `ctx` (256)
positions, so `n` can be arbitrarily large — generation never stops at 256, it
just attends to the most recent 256 tokens. RoPE is relative, so absolute
positions inside the window stay consistent without re-rotating the cache.
Pass `stop_eot=0` as the last gen arg to keep going past `<|endoftext|>`:

```
tinylm.exe gen out/alpaca_m1.bin "List tips for staying healthy." 500 0.8 40 0
```

### MTP heads + self-speculative decoding

Training adds **Medusa-style multi-token-prediction (MTP) heads** by default
(`n_mtp=2`): small extra heads that predict tokens *t+2, t+3* from the same final
hidden, with the unembedding tied to the shared embedding (so each head is just a
`d_model×d_model` matrix, ~16K params for m1). They're zero-initialised, so each
head starts as an exact copy of the main head and then specialises; they're
trained with an added (decayed-weight) cross-entropy loss. Disable with a trailing
`0`: `tinylm.exe train m1 data/alpaca 10000 32 3e-4 out/x.bin 200 0 0`.

At **greedy** decoding (`topk 1` or `temp 0`) an MTP model uses **self-speculative
decoding**: the heads draft the next `n_mtp` tokens, the main model verifies them
in one batched `forward_chunk` pass, and the longest matching prefix is accepted.
This is **exactly correctness-preserving** — output is byte-identical to plain
greedy (verified) — and reports the draft acceptance rate:

```
tinylm.exe gen out/alpaca_m1.bin "Give three tips for staying healthy." 200 0 1
# [gen] ... self-speculative: NN% draft accept, X.XX tokens/forward
TINYLM_NOSPEC=1 tinylm.exe gen ...   # force the plain path (to compare speed)
```

Reality check at this scale: the speedup is only positive when the heads draft
well, which needs a reasonably-trained model. A weak/undertrained model gets ~0%
acceptance, so speculative decoding is *slower* there (it pays for the draft +
a wider verify pass for nothing). It's correct either way; the win grows with
model quality. (Sampling — `temp>0`, `topk>1` — uses the plain path.)

Presets (`nano`/`micro`/`m1`) match the PyTorch reference implementation (plus
optional MTP heads). The
model file stores its own config, so `gen` needs only the model.

Run `tinylm.exe help` for the full reference (commands, args, presets, file
formats, examples).

## Overnight training (checkpointing + auto-resume)

Training **auto-resumes by default**: if the (auto-derived) checkpoint already
exists, `train` continues from it; otherwise it starts fresh. Each checkpoint
writes `<out>.bin` (self-contained) and `<out>.bin.opt` (AdamW state + iter), so
an interrupted run is never lost and resume is exact. So the whole overnight
loop is just the same one line, re-run:

```
set OMP_NUM_THREADS=8
tinylm.exe train m1 data/alpaca 10000      # night 1: fresh; Ctrl-C any time
tinylm.exe train m1 data/alpaca 10000      # night 2: auto-resumes
```

Override defaults positionally if needed: `train m1 data/alpaca 10000 32 3e-4
out/x.bin 200 0` (iters batch lr out save_every resume).

### Train on Alpaca data

Point the prep script at any Alpaca-format JSON (a list of
`{instruction, input, output}` records), then train:

```
python tools/prepare_alpaca.py --input alpaca_data_cleaned.json --vocab_size 2048
set OMP_NUM_THREADS=8
tinylm.exe train m1 data/alpaca 10000       # Ctrl-C safe; re-run to auto-resume
tinylm.exe gen out/alpaca_m1.bin "Give three tips for staying healthy."
```

51,760 Alpaca records at vocab 2048 come to ~13.2M tokens. Each record is
formatted with the standard `### Instruction / ### Input / ### Response`
template, which the checkpoint then carries with it. At ~2,100 tok/s (m1, the
machine below) 10k iters is roughly one overnight run.

> Reality check: at ~1M params, an Alpaca model learns the *format* and produces
> fluent, on-topic-ish English, not correct instruction-following. That needs a
> much larger model — this is the honest ceiling for the size.

## Measured on an i5-1135G7 (MinGW GCC, OpenMP, 8 threads)
- **Training:** ~4,400 tok/s fp32 (nano, batch 32, 8 threads). That's ~75% of
  the PyTorch/MKL build's 5,800 tok/s — a fair result for hand-written kernels
  with **zero dependencies**. Throughput scales with thread count (compute-bound).
- **Correctness:** val loss follows the PyTorch run almost exactly
  (both ≈5.49 at iter 100), which validates the hand-derived gradients.
- **Generation:** ~1,200-1,800 tok/s (m1, B=1) with the sliding KV cache —
  single-token forwards attending to a ring buffer of the last 256 K/V; `n` can
  exceed the context without slowdown or a quality cliff.
- **Decode, 220M/A25M int8 MoE** (12L, D=512, F=707, E=16 top-1, win=1024):
  **120 → 669 tok/s** sustained at the full 1024-token window, interleaved A/B,
  medians of 3. Short context: 119 → 753. See "Decode is not OpenMP" below.

### Decode is not OpenMP

Generation runs on its own persistent spin-wait thread pool
(`TINYLM_THREADS`), not on OpenMP. At one token per forward every parallel
region is a few hundred KB of work, and at that granularity libgomp is a **net
loss**: on the 220M model, 1 thread gave 146 tok/s, 4 gave 116, 8 gave 95.
`OMP_WAIT_POLICY`, `OMP_PROC_BIND` and `GOMP_SPINCOUNT` do not recover it.
Worse, it cost ~30% even at one thread, because `parallel for ... if(cond)`
with `cond` false still calls `GOMP_parallel` and builds a one-thread team —
RMSNorm over a 512-float vector was taking 11.7 µs against ~0.2 µs of
arithmetic.

Microbenchmark, one expert's D=512 → F=707 int8 GEMV, weights drawn from a
12×16 expert pool so nothing caches:

| | 1 thread | 4 | 7 | 8 |
|---|---|---|---|---|
| libgomp `parallel for` | 44.5 µs | 84.5 | — | 106.7 µs |
| spin-wait pool | 38.4 µs | 13.2 | **10.2 µs** | 10.9 |
| pool, effective GB/s | 10.6 | 30.8 | **39.8** | 37.3 |

Training keeps OpenMP — there a region is a whole batch and fork/join is
amortised. The two are never live at once: the pool exists only between
`gen_new` and exit, and the kernels shared with training (`mm`, `mm_bt`,
`rmsnorm_fwd`, `rope_apply`) check for it and take the pool instead of the
pragma.

Other decode changes that came with it:
- The sampler reads the logits `forward_chunk` already produced. It used to
  discard them and recompute the V×D head in **fp32**, streaming all 16.8 MB of
  the embedding every token — 21% of decode, for a result already in hand.
- Top-k is a k-element min-heap, not a k-pass selection sort (at topk=40,
  V=8192 that was ~327k compares plus a malloc and a memcpy per token).
- Q, K and V are one concatenated output-major int8 matrix: one activation
  quantization and one dispatch instead of three.
- W1 and W3 are interleaved row-by-row, so the SwiGLU walks one contiguous
  weight stream and fuses `silu(g)*u` into the same pass — no `g`/`u` buffers.

Accuracy is unchanged: arc_easy 39.5/34.5, piqa 58.0/60.5, lambada 19.0 —
identical to the pre-pool build on 200 items per task.

## Differences from the PyTorch version (intentional)
| Aspect | PyTorch | C port |
|---|---|---|
| Autodiff | autograd | hand-written backward |
| GEMM/attention | MKL + SDPA | hand-written kernels (OpenMP to train, a spin-wait pool to decode) |
| Dropout | 0.05 | disabled (deterministic) |
| Generation | KV cache | sliding KV cache (ring buffer) |
| MTP / spec decode | — | Medusa heads + self-speculative greedy decode |
| Checkpoint | weights only | weights + tokenizer + template + MTP (1 file) |
| Deps | torch, numpy, tokenizers | libc + libgomp only |

## How the gradients are organized
`map_weights()` lays params and grads over one flat buffer each, so AdamW is a
single parallel loop. `backward()` walks layers in reverse using three GEMM
primitives — `mm` (Y=X·W), `mm_bt` (dX=dY·Wᵀ), `mm_atb` (dW=Xᵀ·dY) — plus
explicit RMSNorm, RoPE, SwiGLU, and causal-attention backward. RoPE backward is
just the forward rotation with the sign of `sin` flipped (its transpose).
