/* tinylm.c - a 0.1M-1M parameter decoder-only LM in portable C, multithreaded.
 *
 * A C port of a small PyTorch mini-Llama (see README.md).
 * Same architecture: RMSNorm + RoPE + GQA/MQA + SwiGLU + tied embeddings.
 * Forward + backward + AdamW are hand-written; matmuls/attention use OpenMP.
 *
 * Build (MSYS2 MinGW64):
 *   gcc -O3 -march=native -ffast-math -fopenmp -o tinylm.exe tinylm.c -lm
 *
 * Train:    tinylm train  <preset> <data_dir>  <iters> <batch> <lr> <out.bin>
 * Generate: tinylm gen    <model.bin> <tok.bin> "<prompt>" <n> <temp> <topk>
 *
 * Notes vs the PyTorch version:
 *  - dropout is disabled (deterministic); negligible at this scale.
 *  - generation uses a sliding KV ring cache (forward_chunk). The comment that
 *    used to sit here claimed it recomputed the whole window each token; that
 *    stopped being true when the ring landed and misdirected a profiling pass.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <direct.h>      /* _mkdir (MinGW/Windows) for auto-creating out/ */
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_BLAS          /* link -lopenblas; ~2-3x faster GEMM on AVX-512 */
#include <openblas/cblas.h>
#endif
#include <immintrin.h>   /* int8 GEMV decode kernel (AVX2) */

/* ----------------------------- config ----------------------------------- */
typedef struct {
    int V, D, L, H, KV, F, T;   /* vocab, d_model, layers, heads, kv_heads, d_ff, ctx */
    int hd, KD, half;           /* head_dim, KV*hd, hd/2 (derived) */
    int eot;
    int tmpl;                   /* chat template: 0=raw, 1=alpaca (bundled in checkpoint) */
    int n_mtp;                  /* extra Medusa heads predicting t+2..t+1+n_mtp (0 = off) */
    int n_exp;                  /* MoE experts (0 = dense FFN; >=2 = top-1 routed) */
    /* Sliding-window attention (TLM5/TLQ2, written by the CUDA build). win=0 is
     * full causal attention, i.e. every TLM1-4 model. full_every>0 makes every
     * Nth layer ignore the window and attend to the whole prefix. */
    int win, full_every;
    /* TLM6/TLQ3, written by the CUDA build.
     * n_shared: always-on experts at slots n_exp..n_exp+n_shared-1, run for
     *   EVERY token in addition to the routed ones (DeepSeekMoE).
     * qknorm:   RMS-normalise Q and K per head before RoPE, learned gain of
     *   length hd shared across heads.
     * Both consume no space when zero, so every TLM1-5 checkpoint keeps the
     * exact layout it has always had. */
    int n_shared, qknorm;
    float rope_base;
} Cfg;
/* total expert slots in a layer: routed + shared (1 == the dense FFN) */
static int n_slots(const Cfg *c){ return c->n_exp>0 ? c->n_exp + c->n_shared : 1; }

/* Effective window for layer l: 0 means "attend to the whole prefix".
 * MUST match the GPU backend's lwin() exactly or the two diverge silently. */
static int cfg_lwin(const Cfg *c, int l){
    if(c->win<=0) return 0;
    if(c->full_every>0 && ((l+1)%c->full_every)==0) return 0;
    return c->win;
}

static void cfg_derive(Cfg *c) {
    c->hd = c->D / c->H;
    c->KD = c->KV * c->hd;
    c->half = c->hd / 2;
}

/* number of Medusa/MTP heads on a fresh model. Default 0 (off): each head adds
 * 3 full vocab-sized GEMMs in backward (~30% of training at V=2048) and only
 * benefits self-speculative *inference*, not training quality. Set TINYLM_NMTP=2
 * to re-enable them if you want speculative decoding. */
static int default_nmtp(void){
    const char *e=getenv("TINYLM_NMTP");
    if(e){ int v=atoi(e); return v<0?0:v; }
    return 0;
}

/* Pick a training batch that keeps activation memory ~bounded (so the big
 * presets don't OOM): B=32 for small models, scaled down for wide/deep ones.
 * Override with TINYLM_BATCH. Activation floats per sequence (T tokens) ~
 * T*(L*(6D+2KD+2F) + 5V + 10D); target ~2 GB of activations. */
static int auto_batch(const Cfg *c){
    const char *e=getenv("TINYLM_BATCH");
    if(e){ int v=atoi(e); if(v>=1) return v; }
    double per = (double)c->T * ((double)c->L*(6.0*c->D + 2.0*c->KD + 2.0*c->F)
                                 + 5.0*(double)c->V + 10.0*(double)c->D);
    int B = (int)(2.0e9 / (per*4.0));
    if(B>32) B=32; if(B<4) B=4;
    return B;
}

/* wall-clock interval (seconds) from an env var, else the default. Lets training
 * progress/val/checkpoint cadence be tuned independent of (slow) per-step time. */
static double env_sec(const char *name, double def){
    const char *e=getenv(name);
    if(e){ double v=atof(e); if(v>0) return v; }
    return def;
}

static int preset(const char *name, Cfg *c) {
    memset(c, 0, sizeof(*c));
    c->T = 256; c->rope_base = 10000.0f; c->eot = 0;
    if (!strcmp(name, "nano"))  { c->V=512;  c->D=64;  c->L=4; c->H=4; c->KV=2; c->F=176; }
    else if (!strcmp(name,"micro")){c->V=2048;c->D=96;  c->L=4; c->H=6; c->KV=2; c->F=256; }
    else if (!strcmp(name,"m1")) { c->V=2048;c->D=128; c->L=4; c->H=4; c->KV=1; c->F=336; }
    else if (!strcmp(name,"m3")) { c->V=2048;c->D=192; c->L=6; c->H=6; c->KV=2; c->F=512; }  /* ~3M */
    else if (!strcmp(name,"m6")) { c->V=2048;c->D=256; c->L=8; c->H=8; c->KV=2; c->F=688; }  /* ~6M */
    else if (!strcmp(name,"m8")) { c->V=2048;c->D=288; c->L=8; c->H=8; c->KV=2; c->F=768; }  /* ~8M */
    else if (!strcmp(name,"m16")){c->V=2048;c->D=512; c->L=5; c->H=8; c->KV=2; c->F=1408;}   /* ~15M */
    else if (!strcmp(name,"m32")){c->V=2048;c->D=512; c->L=10;c->H=8; c->KV=2; c->F=1536;}   /* ~31M */
    else if (!strcmp(name,"m70")){c->V=2048;c->D=768; c->L=11;c->H=12;c->KV=2; c->F=2048;}   /* ~69M */
    else return 0;
    cfg_derive(c);
    return 1;
}

/* ----------------------------- rng -------------------------------------- */
static uint64_t g_rng = 1234567891011ULL;
static inline uint64_t xorshift(void){ uint64_t x=g_rng; x^=x<<13; x^=x>>7; x^=x<<17; g_rng=x; return x; }
static inline float rnd_uniform(void){ return (xorshift()>>11) * (1.0f/9007199254740992.0f); }
static inline float rnd_normal(void){
    float u1 = rnd_uniform(); if (u1 < 1e-9f) u1 = 1e-9f;
    float u2 = rnd_uniform();
    return sqrtf(-2.0f*logf(u1)) * cosf(6.28318530718f*u2);
}

/* ----------------------------- alloc ------------------------------------ */
static float *fz(size_t n){ float *p = (float*)calloc(n, sizeof(float)); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }

static double wtime(void){
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock()/CLOCKS_PER_SEC;
#endif
}

/* --------------------- decode thread pool (not OpenMP) ------------------- */
/* Decode is one token at a time, so every parallel region is a few hundred KB
 * of work. Measured on an i5-1135G7 with LS-220M-A25M-q8, libgomp is a net LOSS
 * at that granularity: 1 thread 146 tok/s, 2 -> 85, 4 -> 116, 8 -> 95. Worse,
 * it costs ~30% even at one thread, because `parallel for ... if(cond)` with
 * cond false STILL calls GOMP_parallel and builds a one-thread team -- RMSNorm
 * over a 512-float vector was taking 11.7 us against ~0.2 us of arithmetic.
 * OMP_WAIT_POLICY / OMP_PROC_BIND / GOMP_SPINCOUNT do not recover it.
 *
 * Microbenchmark, one expert's D=512 -> F=707 int8 GEMV, weights drawn from a
 * 12x16 expert pool so nothing caches:
 *     libgomp parallel for : 44.5 us (1t) -> 106.7 us (8t)   2.4x SLOWER
 *     this pool            : 38.4 us (1t) ->  10.2 us (7t)   3.8x faster
 *                            10.6 GB/s    ->  39.8 GB/s
 * Same kernel, same split. The difference is entirely fork/join.
 *
 * Workers are created once, park on an atomic ticket, and each takes a
 * contiguous range of output rows; the caller runs range 0 itself and spins on
 * a done-counter, so there is no thread handoff on the critical path.
 *
 * The pool is live ONLY while generating (gen_new brings it up, atexit tears it
 * down). Training keeps OpenMP -- there a region is a whole batch and fork/join
 * is amortised. The two must never run at once or they oversubscribe the
 * machine, which is what g_pool_nt guards: the kernels shared between decode
 * and training (mm, mm_bt, rmsnorm_fwd, rope_apply) test it and take the pool
 * instead of the pragma. Decode-only kernels have no pragma left at all. */
#define TL_MAXT 64
typedef void (*tl_body)(void *ctx, int lo, int hi, int tid);
static int  g_pool_nt = 1;                  /* workers incl. the caller; 1 = off */
static tl_body       tlp_fn;
static void         *tlp_ctx;
static int           tlp_n;
static volatile long tlp_ticket = 0;        /* bumped once per dispatch */
static volatile long tlp_done   = 0;
static volatile int  tlp_stop   = 0;

#ifdef _WIN32
__declspec(dllimport) void* __stdcall CreateThread(void*, size_t,
        unsigned long (__stdcall *)(void*), void*, unsigned long, unsigned long*);
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void*, unsigned long);
__declspec(dllimport) int  __stdcall CloseHandle(void*);
__declspec(dllimport) int  __stdcall SwitchToThread(void);
__declspec(dllimport) void __stdcall Sleep(unsigned long);
typedef void *tl_thread_t;
static void tl_yield(void){ SwitchToThread(); }
static void tl_nap(void){ Sleep(1); }
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
typedef pthread_t tl_thread_t;
static void tl_yield(void){ sched_yield(); }
static void tl_nap(void){ usleep(1000); }
#endif
static tl_thread_t tlp_th[TL_MAXT];

static void tlp_run(long id){
    int n=tlp_n, nt=g_pool_nt;
    int lo=(int)(((long long)n*id)/nt), hi=(int)(((long long)n*(id+1))/nt);
    if(hi>lo) tlp_fn(tlp_ctx, lo, hi, (int)id);
}
/* Spin, then yield, then nap. A pure spin would fight the OS for a core the
 * moment generation pauses at a chat prompt; a pure sleep would put a 1-15 ms
 * wakeup on a dispatch that takes 10 us. During decode the ticket changes every
 * few microseconds, so the pause arm is the only one that ever runs. */
static void tlp_loop(long id){
    long seen=0;
    for(;;){
        long t, spins=0;
        while((t=__atomic_load_n(&tlp_ticket,__ATOMIC_ACQUIRE))==seen){
            if(__atomic_load_n(&tlp_stop,__ATOMIC_RELAXED)) return;
            if(++spins < 4000)       _mm_pause();
            else if(spins < 24000)   tl_yield();
            else                     tl_nap();
        }
        seen=t;
        if(__atomic_load_n(&tlp_stop,__ATOMIC_RELAXED)) return;
        tlp_run(id);
        __atomic_add_fetch(&tlp_done,1,__ATOMIC_RELEASE);
    }
}
#ifdef _WIN32
static unsigned long __stdcall tlp_entry(void *a){ tlp_loop((long)(intptr_t)a); return 0; }
#else
static void *tlp_entry(void *a){ tlp_loop((long)(intptr_t)a); return NULL; }
#endif

/* Fan `n` items out over the pool. Publishing (fn,ctx,n) happens-before the
 * release bump of the ticket, which the workers acquire-load, so no lock is
 * needed for the descriptor. */
static void tl_for(int n, tl_body fn, void *ctx){
    int nt=g_pool_nt;
    if(nt<=1 || n<nt){ if(n>0) fn(ctx,0,n,0); return; }
    tlp_fn=fn; tlp_ctx=ctx; tlp_n=n;
    __atomic_store_n(&tlp_done,0,__ATOMIC_RELAXED);
    __atomic_add_fetch(&tlp_ticket,1,__ATOMIC_RELEASE);
    tlp_run(0);
    while(__atomic_load_n(&tlp_done,__ATOMIC_ACQUIRE) < nt-1) _mm_pause();
}
static void tlp_shutdown(void){
    if(g_pool_nt<=1) return;
    int nt=g_pool_nt;
    __atomic_store_n(&tlp_stop,1,__ATOMIC_RELAXED);
    __atomic_add_fetch(&tlp_ticket,1,__ATOMIC_RELEASE);
    g_pool_nt=1;                             /* tl_for is serial from here on */
    for(int i=1;i<nt;i++){
#ifdef _WIN32
        WaitForSingleObject(tlp_th[i],2000); CloseHandle(tlp_th[i]);
#else
        pthread_join(tlp_th[i],NULL);
#endif
    }
}
/* TINYLM_THREADS wins, then OMP_NUM_THREADS (so existing scripts keep working),
 * else one fewer than the logical cores -- measured best at 7 of 8 here; 8 is
 * ~6% down because the last hyperthread contends with the caller. */
static int tl_default_threads(void){
    const char *e=getenv("TINYLM_THREADS");
    if(!e||!*e) e=getenv("OMP_NUM_THREADS");
    if(e&&*e){ int v=atoi(e); if(v>0) return v<TL_MAXT?v:TL_MAXT; }
    int np=0;
#ifdef _OPENMP
    np=omp_get_num_procs();
#endif
    if(np<=0){ const char *p=getenv("NUMBER_OF_PROCESSORS"); np=p?atoi(p):0; }
    if(np<=0) np=4;
    if(np>TL_MAXT) np=TL_MAXT;
    return np>2 ? np-1 : np;
}
static int exact_gemv_on(void);
static int exact_exp_on(void);
static int prof_on(void);
static void tlp_init(void){
    if(g_pool_nt>1) return;
    /* Force the lazy getenv() flags to resolve on this thread before any worker
     * exists. They are `static int v=-1` one-shots; the write is idempotent, but
     * resolving them up front keeps the workers' inner loops off that path. */
    exact_gemv_on(); exact_exp_on(); prof_on();
    int nt=tl_default_threads();
    if(nt<=1) return;
    g_pool_nt=nt;                            /* workers read this to size ranges */
    for(int i=1;i<nt;i++){
#ifdef _WIN32
        tlp_th[i]=CreateThread(NULL,0,tlp_entry,(void*)(intptr_t)i,0,NULL);
        if(!tlp_th[i]){ g_pool_nt=i; break; }
#else
        if(pthread_create(&tlp_th[i],NULL,tlp_entry,(void*)(intptr_t)i)!=0){ g_pool_nt=i; break; }
#endif
    }
    atexit(tlp_shutdown);
    if(getenv("TINYLM_PROF")) fprintf(stderr,"[tinylm] decode pool: %d threads\n", g_pool_nt);
}

/* directory containing the running executable, so named models live in one place
 * (models/ next to tinylm.exe) regardless of the current working directory. */
__declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(void*, char*, unsigned long);
static const char *exe_dir(void){
    static char dir[1024]; static int done=0;
    if(done) return dir;
    char path[1024]; unsigned long n=GetModuleFileNameA(0, path, sizeof(path));
    if(n==0 || n>=sizeof(path)){ strcpy(dir,"."); done=1; return dir; }
    path[n]=0;
    char *last=path; for(char *q=path; *q; q++) if(*q=='/'||*q=='\\') last=q;
    size_t len=(size_t)(last-path);            /* dir part (no trailing slash) */
    if(len==0 || len>=sizeof(dir)){ strcpy(dir,"."); done=1; return dir; }
    memcpy(dir,path,len); dir[len]=0; done=1; return dir;
}
/* build "<exe_dir>/models/<name>.bin" and ensure the models/ dir exists */
static void model_path(const char *name, char *buf, size_t n){
    char mdir[700]; snprintf(mdir,sizeof(mdir),"%.690s\\models", exe_dir()); _mkdir(mdir);
    snprintf(buf,n,"%.700s\\%.200s.bin", mdir, name);
}

/* ----------------------------- weights ---------------------------------- */
/* All linear weights stored row-major [in][out] so Y=X@W is the plain mm. */
/* N-gram head (bigram table for lexical prediction) */
typedef struct {
    int R, De, order, group, ngrp;   /* rows, embed dim, 2=bigram, group size, ngroups */
    float *Wng;                      /* projection De*D */
    signed char *d;                  /* q4 digits (packed into (nt+1)/2 bytes); NULL when streaming */
    float *s;                        /* per-group scales */
    float sref;                      /* reference scale */
    FILE *fs;                        /* open handle to .opt.ng when streaming digits from disk */
    long long doff;                  /* byte offset of the digit block within the file */
} Ng;

typedef struct {
    float *emb;                              /* V*D (tied with lm_head)      */
    float **an1, **wq, **wk, **wv, **wo;     /* per layer                    */
    float **an2, **w1, **w3, **w2;
    float **qn, **kn;                        /* QK-norm gains, hd each (NULL if off) */
    float *nf;                               /* final norm gain  D           */
    float **mtp;                             /* n_mtp Medusa heads, each D*D */
    float **wr;                              /* MoE router per layer, D*n_exp (NULL if dense) */
    Ng ng;                                   /* n-gram bigram head (inference only) */
} Weights;

static size_t param_count(const Cfg *c){
    int E = n_slots(c);                        /* routed + shared (dense = 1) */
    size_t per = (size_t)c->D                 /* an1 */
        + (size_t)c->D*c->D                   /* wq  */
        + (size_t)c->D*c->KD                  /* wk  */
        + (size_t)c->D*c->KD                  /* wv  */
        + (size_t)c->D*c->D                   /* wo  */
        + (c->qknorm ? 2*(size_t)c->hd : 0)   /* qn, kn */
        + (size_t)c->D                        /* an2 */
        + (c->n_exp>0 ? (size_t)c->D*c->n_exp : 0)   /* router */
        + (size_t)E*(2*(size_t)c->D*c->F + (size_t)c->F*c->D); /* E expert FFNs (w1,w3,w2) */
    return (size_t)c->V*c->D + (size_t)c->L*per + (size_t)c->D
         + (size_t)c->n_mtp*c->D*c->D;        /* Medusa heads */
}

/* map a flat buffer into named weight pointers (used for params and grads) */
static void map_weights(const Cfg *c, float *base, Weights *w){
    w->an1=malloc(sizeof(float*)*c->L); w->wq=malloc(sizeof(float*)*c->L);
    w->wk =malloc(sizeof(float*)*c->L); w->wv=malloc(sizeof(float*)*c->L);
    w->wo =malloc(sizeof(float*)*c->L); w->an2=malloc(sizeof(float*)*c->L);
    w->w1 =malloc(sizeof(float*)*c->L); w->w3=malloc(sizeof(float*)*c->L);
    w->w2 =malloc(sizeof(float*)*c->L); w->wr=malloc(sizeof(float*)*c->L);
    w->qn =malloc(sizeof(float*)*c->L); w->kn=malloc(sizeof(float*)*c->L);
    int E = n_slots(c);
    float *p = base;
    w->emb = p; p += (size_t)c->V*c->D;
    for (int l=0;l<c->L;l++){
        w->an1[l]=p; p+=c->D;
        w->wq[l]=p;  p+=(size_t)c->D*c->D;
        w->wk[l]=p;  p+=(size_t)c->D*c->KD;
        w->wv[l]=p;  p+=(size_t)c->D*c->KD;
        w->wo[l]=p;  p+=(size_t)c->D*c->D;
        /* qn/kn sit between wo and an2 and take no space when qknorm is off --
         * matching cudalm's calc_offsets exactly. Get this wrong and every
         * weight after it shifts, which reads as garbage rather than an error. */
        if(c->qknorm){ w->qn[l]=p; p+=c->hd; w->kn[l]=p; p+=c->hd; }
        else { w->qn[l]=NULL; w->kn[l]=NULL; }
        w->an2[l]=p; p+=c->D;
        if(c->n_exp>0){ w->wr[l]=p; p+=(size_t)c->D*c->n_exp; } else w->wr[l]=NULL;
        w->w1[l]=p;  p+=(size_t)E*c->D*c->F;   /* E experts, contiguous: expert e at w1[l]+e*D*F */
        w->w3[l]=p;  p+=(size_t)E*c->D*c->F;
        w->w2[l]=p;  p+=(size_t)E*c->F*c->D;   /* expert e at w2[l]+e*F*D */
    }
    w->nf = p; p += c->D;
    if(c->n_mtp>0){
        w->mtp = malloc(sizeof(float*)*c->n_mtp);
        for(int k=0;k<c->n_mtp;k++){ w->mtp[k]=p; p+=(size_t)c->D*c->D; }
    } else w->mtp = NULL;
}

static void init_params(const Cfg *c, Weights *w){
    /* embedding + matrices ~ N(0,0.02); norms = 1; residual projs scaled */
    size_t emb_n=(size_t)c->V*c->D;
    for(size_t i=0;i<emb_n;i++) w->emb[i]=0.02f*rnd_normal();
    float scaled = 0.02f/sqrtf(2.0f*c->L);
    int E = n_slots(c);
    for(int l=0;l<c->L;l++){
        for(int i=0;i<c->D;i++){ w->an1[l][i]=1.0f; w->an2[l][i]=1.0f; }
        size_t dd=(size_t)c->D*c->D, dkd=(size_t)c->D*c->KD;
        size_t edfn=(size_t)E*c->D*c->F, efd=(size_t)E*c->F*c->D;
        for(size_t i=0;i<dd;i++)  w->wq[l][i]=0.02f*rnd_normal();
        for(size_t i=0;i<dkd;i++) w->wk[l][i]=0.02f*rnd_normal();
        for(size_t i=0;i<dkd;i++) w->wv[l][i]=0.02f*rnd_normal();
        for(size_t i=0;i<dd;i++)  w->wo[l][i]=scaled*rnd_normal();   /* residual proj */
        if(c->qknorm) for(int i=0;i<c->hd;i++){ w->qn[l][i]=1.0f; w->kn[l][i]=1.0f; }
        for(size_t i=0;i<edfn;i++) w->w1[l][i]=0.02f*rnd_normal();   /* E experts */
        for(size_t i=0;i<edfn;i++) w->w3[l][i]=0.02f*rnd_normal();
        for(size_t i=0;i<efd;i++)  w->w2[l][i]=scaled*rnd_normal();  /* residual proj */
        if(c->n_exp>0) for(size_t i=0;i<(size_t)c->D*c->n_exp;i++) w->wr[l][i]=0.02f*rnd_normal(); /* router */
    }
    for(int i=0;i<c->D;i++) w->nf[i]=1.0f;
    /* Medusa heads zero-init: hk = fn + silu(fn@0) = fn, so each head starts as
       an exact copy of the main head and then specializes. */
    for(int k=0;k<c->n_mtp;k++){ size_t dd=(size_t)c->D*c->D; for(size_t i=0;i<dd;i++) w->mtp[k][i]=0.0f; }
}

/* ----------------------------- matmuls ---------------------------------- */
/* The portable (non-BLAS) kernels are register-blocked over 4 rows so that each
 * streamed element of the shared operand feeds 4 independent FMA accumulators
 * (4x the arithmetic per load + ILP). The contiguous inner loop vectorizes to
 * AVX2/512 FMA. Mathematically identical to the naive ikj form (verified by
 * `tinylm selftest`); only low-order fp bits differ under -ffast-math. */
typedef struct { float *Y; const float *X,*W; int M,K,N; } MMJob;
static void mm_range(float *restrict Y, const float *restrict X, const float *restrict W,
                     int M, int K, int N, int blo, int bhi);
static void mmbt_range(float *restrict dX, const float *restrict dY, const float *restrict W,
                       int M, int K, int N, int lo, int hi);
static void mm_body(void *p,int lo,int hi,int tid){
    MMJob *j=(MMJob*)p; (void)tid; mm_range(j->Y,j->X,j->W,j->M,j->K,j->N,lo,hi);
}
static void mmbt_body(void *p,int lo,int hi,int tid){
    MMJob *j=(MMJob*)p; (void)tid; mmbt_range(j->Y,j->X,j->W,j->M,j->K,j->N,lo,hi);
}
/* Y[M*N] = X[M*K] @ W[K*N]                      (set) */
static void mm(float *restrict Y, const float *restrict X, const float *restrict W,
               int M, int K, int N){
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,X,K,W,N,0.0f,Y,N); return;
#endif
    int nblk=(M+3)/4;
    MMJob j={Y,X,W,M,K,N};
    if(nblk==1 || g_pool_nt>1){ tl_for(nblk, mm_body, &j); return; }
    #pragma omp parallel for schedule(static)
    for(int b=0;b<nblk;b++) mm_range(Y,X,W,M,K,N,b,b+1);
}
/* One 4-row block range of `mm`, so the same body serves the serial path, the
 * pool and the OpenMP training path without being written three times. */
static void mm_range(float *restrict Y, const float *restrict X, const float *restrict W,
                     int M, int K, int N, int blo, int bhi){
    for(int m0=blo*4; m0<M && m0<bhi*4; m0+=4){
        int mb = M-m0<4 ? M-m0 : 4;
        if(mb==4){
            const float *restrict x0=X+(size_t)m0*K, *restrict x1=x0+K,
                        *restrict x2=x1+K, *restrict x3=x2+K;
            float *restrict y0=Y+(size_t)m0*N, *restrict y1=y0+N,
                  *restrict y2=y1+N, *restrict y3=y2+N;
            for(int n=0;n<N;n++){ y0[n]=0.0f; y1[n]=0.0f; y2[n]=0.0f; y3[n]=0.0f; }
            for(int k=0;k<K;k++){
                float a0=x0[k],a1=x1[k],a2=x2[k],a3=x3[k];
                const float *restrict wr=W+(size_t)k*N;
                for(int n=0;n<N;n++){ float wv=wr[n];
                    y0[n]+=a0*wv; y1[n]+=a1*wv; y2[n]+=a2*wv; y3[n]+=a3*wv; }
            }
        } else {
            for(int mi=0;mi<mb;mi++){
                float *restrict yr=Y+(size_t)(m0+mi)*N;
                const float *restrict xr=X+(size_t)(m0+mi)*K;
                for(int n=0;n<N;n++) yr[n]=0.0f;
                for(int k=0;k<K;k++){ float xk=xr[k]; const float *restrict wr=W+(size_t)k*N;
                    for(int n=0;n<N;n++) yr[n]+=xk*wr[n]; }
            }
        }
    }
}
/* dX[M*K] += dY[M*N] @ W^T,  W is [K*N]         (accumulate) */
static void mm_bt(float *restrict dX, const float *restrict dY, const float *restrict W,
                  int M, int K, int N){
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,K,N,1.0f,dY,N,W,N,1.0f,dX,K); return;
#endif
    MMJob j={dX,dY,W,M,K,N};
    if(M==1 || g_pool_nt>1){ tl_for(M, mmbt_body, &j); return; }
    #pragma omp parallel for schedule(static)
    for(int m=0;m<M;m++) mmbt_range(dX,dY,W,M,K,N,m,m+1);
}
static void mmbt_range(float *restrict dX, const float *restrict dY, const float *restrict W,
                       int M, int K, int N, int lo, int hi){
    (void)M;
    for(int m=lo;m<hi;m++){
        const float *restrict dy = dY + (size_t)m*N;
        float *restrict dx = dX + (size_t)m*K;
        int k=0;
        for(;k+4<=K;k+=4){      /* reuse dy[n] across 4 W rows -> 4 accumulators */
            const float *restrict w0=W+(size_t)k*N, *restrict w1=w0+N,
                        *restrict w2=w1+N, *restrict w3=w2+N;
            float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
            for(int n=0;n<N;n++){ float d=dy[n];
                s0+=d*w0[n]; s1+=d*w1[n]; s2+=d*w2[n]; s3+=d*w3[n]; }
            dx[k]+=s0; dx[k+1]+=s1; dx[k+2]+=s2; dx[k+3]+=s3;
        }
        for(;k<K;k++){ const float *restrict wr=W+(size_t)k*N; float s=0.0f;
            for(int n=0;n<N;n++) s+=dy[n]*wr[n]; dx[k]+=s; }
    }
}
/* dW[K*N] += X[M*K]^T @ dY[M*N]                 (accumulate) */
static void mm_atb(float *restrict dW, const float *restrict X, const float *restrict dY,
                   int M, int K, int N){
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,K,N,M,1.0f,X,K,dY,N,1.0f,dW,N); return;
#endif
    #pragma omp parallel for schedule(static)
    for(int k0=0;k0<K;k0+=4){
        int kb = K-k0<4 ? K-k0 : 4;
        if(kb==4){             /* reuse dy[n] across 4 dW rows -> 4x per load */
            float *restrict d0=dW+(size_t)k0*N, *restrict d1=d0+N,
                  *restrict d2=d1+N, *restrict d3=d2+N;
            for(int m=0;m<M;m++){
                const float *restrict xr=X+(size_t)m*K;
                float a0=xr[k0],a1=xr[k0+1],a2=xr[k0+2],a3=xr[k0+3];
                const float *restrict dy=dY+(size_t)m*N;
                for(int n=0;n<N;n++){ float d=dy[n];
                    d0[n]+=a0*d; d1[n]+=a1*d; d2[n]+=a2*d; d3[n]+=a3*d; }
            }
        } else {
            for(int kk=k0;kk<k0+kb;kk++){
                float *restrict dw=dW+(size_t)kk*N;
                for(int m=0;m<M;m++){ float xk=X[(size_t)m*K+kk];
                    const float *restrict dy=dY+(size_t)m*N;
                    for(int n=0;n<N;n++) dw[n]+=xk*dy[n]; }
            }
        }
    }
}

/* ----------------------------- rmsnorm ---------------------------------- */
typedef struct { const float *x,*g; float *out,*rinv; int D; } RmsJob;
static void rms_range(const float *x, const float *g, float *out, float *rinv,
                      int D, int lo, int hi){
    for(int r=lo;r<hi;r++){
        const float *xr=x+(size_t)r*D; float *or_=out+(size_t)r*D;
        float ms=0.0f; for(int i=0;i<D;i++) ms+=xr[i]*xr[i];
        float s=1.0f/sqrtf(ms/D + 1e-5f); rinv[r]=s;
        for(int i=0;i<D;i++) or_[i]=xr[i]*s*g[i];
    }
}
static void rms_body(void *p,int lo,int hi,int tid){
    RmsJob *j=(RmsJob*)p; (void)tid; rms_range(j->x,j->g,j->out,j->rinv,j->D,lo,hi);
}
static void rmsnorm_fwd(const float *x, const float *g, float *out, float *rinv,
                        int rows, int D){
    /* rows==1 is the decode case and MUST NOT touch libgomp: an `if(rows>1)`
       clause still calls GOMP_parallel for a one-thread team, which measured
       11.7 us per call against ~0.2 us of arithmetic. */
    if(rows==1){ rms_range(x,g,out,rinv,D,0,1); return; }
    RmsJob j={x,g,out,rinv,D};
    if(g_pool_nt>1){ tl_for(rows, rms_body, &j); return; }
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++) rms_range(x,g,out,rinv,D,r,r+1);
}
/* given dy (grad of out), add to dx and dg */
static void rmsnorm_bwd(const float *x, const float *g, const float *rinv,
                        const float *dy, float *dx, float *dg, int rows, int D){
    /* dg accumulated across rows -> needs reduction; do per-row into thread-local then atomic */
    #pragma omp parallel
    {
        float *dg_local = fz(D);
        #pragma omp for schedule(static)
        for(int r=0;r<rows;r++){
            const float *xr=x+(size_t)r*D, *dyr=dy+(size_t)r*D;
            float *dxr=dx+(size_t)r*D;
            float s=rinv[r];
            float c=0.0f;
            for(int i=0;i<D;i++) c += xr[i]*g[i]*dyr[i];
            float coef = s*s*s/D * c;
            for(int i=0;i<D;i++){
                dxr[i] += s*g[i]*dyr[i] - coef*xr[i];
                dg_local[i] += (xr[i]*s)*dyr[i];
            }
        }
        #pragma omp critical
        { for(int i=0;i<D;i++) dg[i]+=dg_local[i]; }
        free(dg_local);
    }
}

/* --------------------------- QK-norm ------------------------------------
 * RMS-normalise each head's hd-vector in place and apply a learned gain of
 * length hd shared across heads, before RoPE. A head's values are contiguous
 * inside a [rows, nheads*hd] buffer, so this is exactly cudalm's k_rmsnorm
 * invoked with rows*nheads rows of width hd -- same eps (1e-5 on the mean
 * square), same order of operations. Any divergence here is a silent
 * train/inference mismatch, not a crash. */
static void qknorm_apply(float *buf, const float *gain, int rows, int nheads, int hd){
    int R = rows*nheads;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<R;r++){
        float *v = buf + (size_t)r*hd;
        float ss=0; for(int i=0;i<hd;i++) ss += v[i]*v[i];
        float s = 1.0f/sqrtf(ss/hd + 1e-5f);
        for(int i=0;i<hd;i++) v[i] = v[i]*s*gain[i];
    }
}

/* ----------------------------- rope ------------------------------------- */
static float *rope_cos, *rope_sin;            /* [maxpos*half] */
static int    rope_max;                        /* positions available */
/* maxpos can exceed c->T (sliding-window generation needs absolute positions
 * beyond the training context). */
static void rope_init(const Cfg *c, int maxpos){
    if(maxpos < c->T) maxpos = c->T;
    if(maxpos <= rope_max && rope_cos) return;     /* table already big enough */
    free(rope_cos); free(rope_sin);
    rope_max = maxpos;
    rope_cos=fz((size_t)maxpos*c->half); rope_sin=fz((size_t)maxpos*c->half);
    for(int t=0;t<maxpos;t++) for(int i=0;i<c->half;i++){
        float inv = powf(c->rope_base, -(2.0f*i)/c->hd);
        float a=t*inv; rope_cos[t*c->half+i]=cosf(a); rope_sin[t*c->half+i]=sinf(a);
    }
}
/* apply rope in place to a [rows][nheads*hd] buffer; rows indexed by absolute pos via pos[] */
typedef struct { float *buf; const int *pos; int nheads,hd,half,back; } RopeJob;
static void rope_range(float *buf, const int *pos, int nheads, int hd, int half, int back,
                       int lo, int hi){
    for(int r=lo;r<hi;r++){
        int t=pos[r];
        for(int h=0;h<nheads;h++){
            float *v = buf + (size_t)r*nheads*hd + (size_t)h*hd;
            for(int i=0;i<half;i++){
                float c=rope_cos[t*half+i], s=rope_sin[t*half+i];
                float a=v[i], b=v[i+half];
                if(!back){ v[i]=a*c-b*s; v[i+half]=b*c+a*s; }
                else     { v[i]=a*c+b*s; v[i+half]=-a*s+b*c; }   /* transpose */
            }
        }
    }
}
static void rope_body(void *p,int lo,int hi,int tid){
    RopeJob *j=(RopeJob*)p; (void)tid;
    rope_range(j->buf,j->pos,j->nheads,j->hd,j->half,j->back,lo,hi);
}
static void rope_apply(float *buf, const int *pos, int rows, int nheads, int hd, int half, int back){
    if(rows==1){ rope_range(buf,pos,nheads,hd,half,back,0,1); return; }  /* see rmsnorm_fwd */
    RopeJob j={buf,pos,nheads,hd,half,back};
    if(g_pool_nt>1){ tl_for(rows, rope_body, &j); return; }
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++) rope_range(buf,pos,nheads,hd,half,back,r,r+1);
}

/* ----------------------------- activations ------------------------------ */
typedef struct {
    int B, T, N;                              /* N = B*T */
    float **a_in,**a_norm,*a_rinv_buf,**a_rinv;
    float **q,**k,**v,**att_ctx;
    float **f_in,**f_norm,*f_rinv_buf,**f_rinv;
    float **g,**u;
    float *x_final,*fn_norm,*fn_rinv,*logits,*probs;
    /* scratch grads */
    float *dout,*dact,*dg,*du,*dfnorm,*dfin,*datt,*dq,*dk,*dv,*dnorm,*dain,*dlogits,*dfn;
    int *pos;                                 /* abs position per row */
    int **assign; float **gate,**rprobs;      /* MoE routing per layer (n_exp>0) */
} Acts;

static Acts *acts_new(const Cfg *c, int B){
    Acts *a=calloc(1,sizeof(Acts));
    a->B=B; a->T=c->T; a->N=B*c->T; int N=a->N, D=c->D, F=c->F, KD=c->KD, V=c->V, L=c->L;
    #define LA(name) a->name=malloc(sizeof(float*)*L)
    LA(a_in);LA(a_norm);LA(a_rinv);LA(q);LA(k);LA(v);LA(att_ctx);LA(f_in);LA(f_norm);LA(f_rinv);LA(g);LA(u);
    #undef LA
    a->a_rinv_buf=fz((size_t)N*L); a->f_rinv_buf=fz((size_t)N*L);
    for(int l=0;l<L;l++){
        a->a_in[l]=fz((size_t)N*D); a->a_norm[l]=fz((size_t)N*D);
        a->a_rinv[l]=a->a_rinv_buf+(size_t)N*l;
        a->q[l]=fz((size_t)N*D); a->k[l]=fz((size_t)N*KD); a->v[l]=fz((size_t)N*KD);
        a->att_ctx[l]=fz((size_t)N*D);
        a->f_in[l]=fz((size_t)N*D); a->f_norm[l]=fz((size_t)N*D);
        a->f_rinv[l]=a->f_rinv_buf+(size_t)N*l;
        a->g[l]=fz((size_t)N*F); a->u[l]=fz((size_t)N*F);
    }
    if(c->n_exp>0){
        a->assign=malloc(sizeof(int*)*L); a->gate=malloc(sizeof(float*)*L); a->rprobs=malloc(sizeof(float*)*L);
        for(int l=0;l<L;l++){ a->assign[l]=malloc(sizeof(int)*N); a->gate[l]=fz((size_t)N);
            a->rprobs[l]=fz((size_t)N*c->n_exp); }
    }
    a->x_final=fz((size_t)N*D); a->fn_norm=fz((size_t)N*D); a->fn_rinv=fz((size_t)N);
    a->logits=fz((size_t)N*V); a->probs=fz((size_t)N*V);
    a->dout=fz((size_t)N*D); a->dact=fz((size_t)N*F); a->dg=fz((size_t)N*F); a->du=fz((size_t)N*F);
    a->dfnorm=fz((size_t)N*D); a->dfin=fz((size_t)N*D); a->datt=fz((size_t)N*D);
    a->dq=fz((size_t)N*D); a->dk=fz((size_t)N*KD); a->dv=fz((size_t)N*KD);
    a->dnorm=fz((size_t)N*D); a->dain=fz((size_t)N*D); a->dlogits=fz((size_t)N*V); a->dfn=fz((size_t)N*D);
    a->pos=malloc(sizeof(int)*N);
    return a;
}

/* ----------------------------- attention -------------------------------- */
static void attn_fwd(const Cfg *c, Acts *a, int l){
    const float *q=a->q[l], *k=a->k[l], *v=a->v[l]; float *out=a->att_ctx[l];
    int T=c->T,H=c->H,KV=c->KV,hd=c->hd,D=c->D,KD=c->KD,B=a->B;
    float scale=1.0f/sqrtf((float)hd); int grp=H/KV;
    int win=cfg_lwin(c,l);                    /* 0 = full causal */
    #pragma omp parallel
    {
        float *probs=malloc(sizeof(float)*T);     /* one scratch per thread */
        #pragma omp for schedule(static)
        for(int bh=0;bh<B*H;bh++){
            int b=bh/H, h=bh%H, kvh=h/grp;
            for(int t=0;t<T;t++){
                const float *qp=q+((size_t)(b*T+t)*D)+(size_t)h*hd;
                int lo=(win>0 && t>=win)?t-win+1:0;   /* sliding window */
                float mx=-1e30f;
                for(int s=lo;s<=t;s++){
                    const float *kp=k+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    float dot=0; for(int d=0;d<hd;d++) dot+=qp[d]*kp[d];
                    dot*=scale; probs[s]=dot; if(dot>mx)mx=dot;
                }
                float sum=0; for(int s=lo;s<=t;s++){ probs[s]=expf(probs[s]-mx); sum+=probs[s]; }
                float inv=1.0f/sum;
                float *op=out+((size_t)(b*T+t)*D)+(size_t)h*hd;
                for(int d=0;d<hd;d++) op[d]=0;
                for(int s=lo;s<=t;s++){
                    float p=probs[s]*inv;
                    const float *vp=v+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    for(int d=0;d<hd;d++) op[d]+=p*vp[d];
                }
            }
        }
        free(probs);
    }
}
/* backward: parallel over batch only so shared kv-heads don't race */
static void attn_bwd(const Cfg *c, Acts *a, int l){
    const float *q=a->q[l], *k=a->k[l], *v=a->v[l], *datt=a->datt;
    float *dq=a->dq, *dk=a->dk, *dv=a->dv;
    int T=c->T,H=c->H,KV=c->KV,hd=c->hd,D=c->D,KD=c->KD,B=a->B;
    float scale=1.0f/sqrtf((float)hd); int grp=H/KV;
    int win=cfg_lwin(c,l);                    /* 0 = full causal */
    #pragma omp parallel
    {
        float *probs=malloc(sizeof(float)*T);     /* per-thread scratch */
        float *dpr =malloc(sizeof(float)*T);
        #pragma omp for schedule(static)
        for(int b=0;b<B;b++){
        for(int h=0;h<H;h++){
            int kvh=h/grp;
            for(int t=0;t<T;t++){
                const float *qp=q+((size_t)(b*T+t)*D)+(size_t)h*hd;
                int lo=(win>0 && t>=win)?t-win+1:0;   /* same window as attn_fwd */
                /* recompute softmax */
                float mx=-1e30f;
                for(int s=lo;s<=t;s++){
                    const float *kp=k+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    float dot=0; for(int d=0;d<hd;d++) dot+=qp[d]*kp[d];
                    dot*=scale; probs[s]=dot; if(dot>mx)mx=dot;
                }
                float sum=0; for(int s=lo;s<=t;s++){ probs[s]=expf(probs[s]-mx); sum+=probs[s]; }
                float inv=1.0f/sum; for(int s=lo;s<=t;s++) probs[s]*=inv;
                const float *dap=datt+((size_t)(b*T+t)*D)+(size_t)h*hd;
                /* d_probs and d_v */
                for(int s=lo;s<=t;s++){
                    const float *vp=v+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    float dp=0; for(int d=0;d<hd;d++) dp+=dap[d]*vp[d];
                    dpr[s]=dp;
                    float *dvp=dv+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    for(int d=0;d<hd;d++) dvp[d]+=probs[s]*dap[d];
                }
                /* softmax backward */
                float dotpd=0; for(int s=lo;s<=t;s++) dotpd+=probs[s]*dpr[s];
                float *dqp=dq+((size_t)(b*T+t)*D)+(size_t)h*hd;
                for(int s=lo;s<=t;s++){
                    float dsc=probs[s]*(dpr[s]-dotpd)*scale;
                    const float *kp=k+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    float *dkp=dk+((size_t)(b*T+s)*KD)+(size_t)kvh*hd;
                    for(int d=0;d<hd;d++){ dqp[d]+=dsc*kp[d]; dkp[d]+=dsc*qp[d]; }
                }
            }
        }
        }
        free(probs); free(dpr);
    }
}

/* ----------------------------- forward ---------------------------------- */
static inline float siluf(float z){ return z/(1.0f+expf(-z)); }

/* ------------------------------ MoE (top-K) ----------------------------- */
/* Experts routed per token. K is NOT stored in the checkpoint (cudalm reads it
 * from CUDALM_TOPK at train time and never serializes it), so inference must be
 * told the same value the model was trained with. Default 1; the served TLM4
 * checkpoint was trained CUDALM_TOPK=2, so set that env to reproduce it. */
static int g_moe_topk=0;   /* 0 = unresolved */
static int moe_topk(int E){
    if(!g_moe_topk){ const char *s=getenv("CUDALM_TOPK"); int k=s?atoi(s):1; g_moe_topk=k<1?1:k; }
    int k=g_moe_topk; return k>E?E:k;
}
/* Index of the k-th largest prob (k=0 = max), ties broken by lowest index --
 * matches cudalm's strict-greater argmax-with-masking in k_router_topk. */
static inline int moe_kth(const float *pr, int E, int k){
    for(int e=0;e<E;e++){
        float v=pr[e]; int rank=0;
        for(int e2=0;e2<E;e2++) if(pr[e2]>v || (pr[e2]==v && e2<e)) rank++;
        if(rank==k) return e;
    }
    return 0;
}
/* Shared scratch for the gather/scatter expert GEMMs (grows as needed). */
static float *me_X,*me_g,*me_u,*me_a,*me_y,*me_dy,*me_da,*me_dg,*me_du,*me_dX,*me_rt,*me_drt;
static int *me_idx; static size_t me_cap=0;
/* int8 activation scratch for the fused single-token expert path */
static int8_t *me_xq=NULL,*me_aq=NULL; static float *me_xs=NULL,*me_as=NULL; static int me_qcap=0;
static void me_qensure(int D,int F){
    int need=(D>F?D:F); if(need<=me_qcap) return;
    int nb=(need+31)/32;
    me_xq=realloc(me_xq,(size_t)nb*32); me_aq=realloc(me_aq,(size_t)nb*32);
    me_xs=realloc(me_xs,sizeof(float)*nb); me_as=realloc(me_as,sizeof(float)*nb);
    me_qcap=nb*32;
}
static void me_ensure(int N,int D,int F,int E){
    if((size_t)N<=me_cap) return;
    size_t nd=(size_t)N*D, nf=(size_t)N*F, ne=(size_t)N*E;
    me_X=realloc(me_X,nd*4); me_g=realloc(me_g,nf*4); me_u=realloc(me_u,nf*4); me_a=realloc(me_a,nf*4);
    me_y=realloc(me_y,nd*4); me_dy=realloc(me_dy,nd*4); me_da=realloc(me_da,nf*4);
    me_dg=realloc(me_dg,nf*4); me_du=realloc(me_du,nf*4); me_dX=realloc(me_dX,nd*4);
    me_rt=realloc(me_rt,ne*4); me_drt=realloc(me_drt,ne*4); me_idx=realloc(me_idx,(size_t)N*4);
    me_cap=N;
}
#define MOE_AUX 0.01f   /* load-balance aux-loss weight */

/* MoE FFN: out[n] = fin[n] + sum_{k<K} pr[e_k]*Expert_{e_k}(fnorm[n]), where
 * e_0..e_{K-1} are the K highest-prob experts and pr is the softmax over ALL E
 * (no top-K renormalization -- gates stay the raw normalized probs, matching
 * cudalm). Writes routing (assign,gate,rprobs) for the backward; assign/gate
 * hold the LAST slot on return, so backward is only exact for K==1 (the
 * training default -- this model is trained on cudalm, not here). */
static void moe_forward(const Cfg *c, const Weights *w, int l, const float *fnorm,
                        const float *fin, float *out, int N, int *assign, float *gate, float *rprobs){
    int D=c->D,F=c->F,E=c->n_exp, K=moe_topk(E);
    me_ensure(N,D,F,E);
    mm(me_rt, fnorm, w->wr[l], N, D, E);                 /* router logits [N,E] */
    #pragma omp parallel for schedule(static)
    for(int n=0;n<N;n++){
        float *r=me_rt+(size_t)n*E, *pr=rprobs+(size_t)n*E;
        float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx)mx=r[e];
        float s=0; for(int e=0;e<E;e++){ pr[e]=expf(r[e]-mx); s+=pr[e]; }
        float inv=1.0f/s; for(int e=0;e<E;e++) pr[e]*=inv;
    }
    memcpy(out, fin, sizeof(float)*(size_t)N*D);          /* residual */
    for(int k=0;k<K;k++){
        #pragma omp parallel for schedule(static)
        for(int n=0;n<N;n++){ int sel=moe_kth(rprobs+(size_t)n*E,E,k);
            assign[n]=sel; gate[n]=rprobs[(size_t)n*E+sel]; }
        for(int e=0;e<E;e++){
            int ne=0; for(int n=0;n<N;n++) if(assign[n]==e) me_idx[ne++]=n;   /* serial index build (cheap) */
            if(!ne) continue;
            #pragma omp parallel for schedule(static)
            for(int j=0;j<ne;j++) memcpy(me_X+(size_t)j*D, fnorm+(size_t)me_idx[j]*D, sizeof(float)*D);
            const float *w1=w->w1[l]+(size_t)e*D*F, *w3=w->w3[l]+(size_t)e*D*F, *w2=w->w2[l]+(size_t)e*F*D;
            mm(me_g, me_X, w1, ne, D, F);
            mm(me_u, me_X, w3, ne, D, F);
            #pragma omp parallel for schedule(static)
            for(size_t i=0;i<(size_t)ne*F;i++) me_a[i]=siluf(me_g[i])*me_u[i];
            mm(me_y, me_a, w2, ne, F, D);
            #pragma omp parallel for schedule(static)
            for(int j=0;j<ne;j++){ int n=me_idx[j]; float gt=gate[n]; float *o=out+(size_t)n*D, *y=me_y+(size_t)j*D;
                for(int d=0;d<D;d++) o[d]+=gt*y[d]; }
        }
    }
}

/* MoE FFN backward: dout = grad of out. Produces d_fnorm (set) and accumulates
 * expert FFN grads + router grad into gr. */
static void moe_backward(const Cfg *c, const Weights *w, Weights *gr, int l, const float *fnorm,
                         const float *dout, float *d_fnorm, const int *assign, const float *gate,
                         const float *rprobs, int N){
    int D=c->D,F=c->F,E=c->n_exp;
    me_ensure(N,D,F,E);
    memset(d_fnorm,0,sizeof(float)*(size_t)N*D);
    float *d_gate=calloc(N,4);
    /* per-expert FFN backward via gather/scatter */
    for(int e=0;e<E;e++){
        int ne=0; for(int n=0;n<N;n++) if(assign[n]==e) me_idx[ne++]=n;
        if(!ne) continue;
        const float *w1=w->w1[l]+(size_t)e*D*F, *w3=w->w3[l]+(size_t)e*D*F, *w2=w->w2[l]+(size_t)e*F*D;
        float *gw1=gr->w1[l]+(size_t)e*D*F, *gw3=gr->w3[l]+(size_t)e*D*F, *gw2=gr->w2[l]+(size_t)e*F*D;
        /* gather Xe, recompute ge,ue,act,ye */
        #pragma omp parallel for schedule(static)
        for(int j=0;j<ne;j++) memcpy(me_X+(size_t)j*D, fnorm+(size_t)me_idx[j]*D, sizeof(float)*D);
        mm(me_g, me_X, w1, ne, D, F); mm(me_u, me_X, w3, ne, D, F);
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<(size_t)ne*F;i++) me_a[i]=siluf(me_g[i])*me_u[i];
        mm(me_y, me_a, w2, ne, F, D);                       /* ye (pre-gate) */
        /* d_gate and gathered d_ye = gate*dout */
        #pragma omp parallel for schedule(static)
        for(int j=0;j<ne;j++){ int n=me_idx[j]; const float *doR=dout+(size_t)n*D, *yR=me_y+(size_t)j*D;
            float dg=0; for(int d=0;d<D;d++) dg+=doR[d]*yR[d]; d_gate[n]=dg;
            float gt=gate[n]; float *dyR=me_dy+(size_t)j*D; for(int d=0;d<D;d++) dyR[d]=gt*doR[d]; }
        /* dW2 += act^T@dYe ; d_act = dYe@w2^T */
        mm_atb(gw2, me_a, me_dy, ne, F, D);
        memset(me_da,0,sizeof(float)*(size_t)ne*F); mm_bt(me_da, me_dy, w2, ne, F, D);
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<(size_t)ne*F;i++){ float z=me_g[i]; float sg=1.0f/(1.0f+expf(-z)); float si=z*sg;
            float ds=sg+z*sg*(1.0f-sg); me_dg[i]=me_da[i]*me_u[i]*ds; me_du[i]=me_da[i]*si; }
        mm_atb(gw1, me_X, me_dg, ne, D, F); mm_atb(gw3, me_X, me_du, ne, D, F);
        memset(me_dX,0,sizeof(float)*(size_t)ne*D); mm_bt(me_dX, me_dg, w1, ne, D, F); mm_bt(me_dX, me_du, w3, ne, D, F);
        #pragma omp parallel for schedule(static)
        for(int j=0;j<ne;j++){ int n=me_idx[j]; float *df=d_fnorm+(size_t)n*D, *dx=me_dX+(size_t)j*D;
            for(int d=0;d<D;d++) df[d]+=dx[d]; }
    }
    /* router backward: gate path + load-balance aux; then softmax back -> d_rt */
    int *cnt=calloc(E,sizeof(int)); for(int n=0;n<N;n++) cnt[assign[n]]++;
    float invN=1.0f/N;
    #pragma omp parallel for schedule(static)
    for(int n=0;n<N;n++){
        const float *pr=rprobs+(size_t)n*E; float *drt=me_drt+(size_t)n*E;
        /* d w.r.t rprobs[n,j]: aux for all j, gate only for the chosen expert */
        float dp[64]; for(int e=0;e<E;e++) dp[e]=MOE_AUX*E*(cnt[e]*invN)*invN;  /* aux: a*E*f_e/N */
        dp[assign[n]] += d_gate[n];
        float dot=0; for(int e=0;e<E;e++) dot+=pr[e]*dp[e];
        for(int e=0;e<E;e++) drt[e]=pr[e]*(dp[e]-dot);          /* softmax backward */
    }
    mm_bt(d_fnorm, me_drt, w->wr[l], N, D, E);                 /* d_fnorm += d_rt @ wr^T */
    mm_atb(gr->wr[l], fnorm, me_drt, N, D, E);                 /* dWr += fnorm^T @ d_rt */
    free(d_gate); free(cnt);
}
/* fills acts; if targets!=NULL also computes probs+loss (returns mean loss) */
/* QLM_ACT8: per-token int8 fake-quant of the matmul inputs (BitNet-a8 style).
 * Rounds in place and keeps fp32 arithmetic, so it measures the QUALITY cost of
 * int8 activations without needing an int8 GEMM. Backward is straight-through:
 * the gradient of round() is taken as 1, which is why no backward change is
 * needed. Covers the inputs to wq/wk/wv (a_norm), w1/w3 (f_norm) and w2 (the
 * SwiGLU product); wo's input (att_ctx) is left alone because attn_bwd reads it.
 */
static int g_act8 = 0;
static void qlm_act8(float *X, int rows, int cols){
    #pragma omp parallel for schedule(static)
    for(int n=0;n<rows;n++){
        float *x=X+(size_t)n*cols, m=0.0f;
        for(int i=0;i<cols;i++){ float t=fabsf(x[i]); if(t>m)m=t; }
        if(!(m>0.0f)) continue;
        float sc=m/127.0f, inv=1.0f/sc;
        for(int i=0;i<cols;i++) x[i]=sc*(float)((int)lrintf(x[i]*inv));
    }
}

static float forward(const Cfg *c, const Weights *w, Acts *a,
                     const int *tok, const int *targets){
    int N=a->N, D=c->D, F=c->F, KD=c->KD, V=c->V, T=c->T;
    /* embedding -> a_in[0] */
    for(int n=0;n<N;n++){
        const float *e=w->emb+(size_t)tok[n]*D;
        memcpy(a->a_in[0]+(size_t)n*D, e, sizeof(float)*D);
        a->pos[n]= n % T;
    }
    for(int l=0;l<c->L;l++){
        float *xin = a->a_in[l];
        rmsnorm_fwd(xin, w->an1[l], a->a_norm[l], a->a_rinv[l], N, D);
        if(g_act8) qlm_act8(a->a_norm[l], N, D);
        mm(a->q[l], a->a_norm[l], w->wq[l], N, D, D);
        mm(a->k[l], a->a_norm[l], w->wk[l], N, D, KD);
        mm(a->v[l], a->a_norm[l], w->wv[l], N, D, KD);
        if(c->qknorm){
            qknorm_apply(a->q[l], w->qn[l], N, c->H,  c->hd);
            qknorm_apply(a->k[l], w->kn[l], N, c->KV, c->hd);
        }
        rope_apply(a->q[l], a->pos, N, c->H,  c->hd, c->half, 0);
        rope_apply(a->k[l], a->pos, N, c->KV, c->hd, c->half, 0);
        attn_fwd(c, a, l);
        /* f_in = a_in + att_ctx@Wo */
        float *attn_proj = a->f_in[l];
        mm(attn_proj, a->att_ctx[l], w->wo[l], N, D, D);
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<(size_t)N*D;i++) attn_proj[i]+=xin[i];
        rmsnorm_fwd(a->f_in[l], w->an2[l], a->f_norm[l], a->f_rinv[l], N, D);
        if(g_act8) qlm_act8(a->f_norm[l], N, D);
        float *out = (l+1<c->L)? a->a_in[l+1] : a->x_final;
        if(c->n_exp>0){
            moe_forward(c,w,l, a->f_norm[l], a->f_in[l], out, N, a->assign[l], a->gate[l], a->rprobs[l]);
        } else {
            mm(a->g[l], a->f_norm[l], w->w1[l], N, D, F);
            mm(a->u[l], a->f_norm[l], w->w3[l], N, D, F);
            float *act=a->dact;
            #pragma omp parallel for schedule(static)
            for(size_t i=0;i<(size_t)N*F;i++) act[i]=siluf(a->g[l][i])*a->u[l][i];
            if(g_act8) qlm_act8(act, N, F);
            mm(out, act, w->w2[l], N, F, D);
            #pragma omp parallel for schedule(static)
            for(size_t i=0;i<(size_t)N*D;i++) out[i]+=a->f_in[l][i];
        }
    }
    rmsnorm_fwd(a->x_final, w->nf, a->fn_norm, a->fn_rinv, N, D);
    /* logits = fn @ emb^T : logits[n,vv]=sum_d fn[n,d]*emb[vv,d] */
    memset(a->logits, 0, sizeof(float)*(size_t)N*V);
    mm_bt(a->logits, a->fn_norm, w->emb, N, V, D);
    if(!targets) return 0.0f;
    /* softmax + CE */
    double loss=0.0;
    #pragma omp parallel for reduction(+:loss) schedule(static)
    for(int n=0;n<N;n++){
        float *lg=a->logits+(size_t)n*V, *pr=a->probs+(size_t)n*V;
        float mx=-1e30f; for(int i=0;i<V;i++) if(lg[i]>mx)mx=lg[i];
        float sum=0; for(int i=0;i<V;i++){ pr[i]=expf(lg[i]-mx); sum+=pr[i]; }
        float inv=1.0f/sum; for(int i=0;i<V;i++) pr[i]*=inv;
        loss += -log((double)pr[targets[n]] + 1e-12);
    }
    return (float)(loss/N);
}

/* ----------------------------- MTP head gradient ------------------------ */
/* Medusa heads: hk = fn + silu(fn@Wk); logits_k = hk @ emb^T (tied). Each head k
 * predicts the token (k+2) ahead. Accumulates grads into a->dfn (so it folds into
 * the trunk before the final-norm backward), gr->emb (tied), and gr->mtp[k].
 * Reuses backward scratch that is free at this point. */
static void mtp_grad(const Cfg *c, const Weights *w, Weights *gr, Acts *a, int *const *tgt_mtp){
    int N=a->N, D=c->D, V=c->V;
    const float *fn=a->fn_norm;
    float *t1=a->datt, *hk=a->dnorm, *dhk=a->dq, *dt1=a->dain;   /* N*D scratch */
    float *lg=a->logits, *pr=a->probs, *dl=a->dlogits;          /* N*V scratch */
    for(int k=0;k<c->n_mtp;k++){
        float wk = 0.2f * powf(0.8f, (float)k);                 /* head loss weight */
        const int *tgt=tgt_mtp[k];
        mm(t1, fn, w->mtp[k], N, D, D);                         /* t1 = fn@Wk */
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<(size_t)N*D;i++) hk[i]=fn[i]+siluf(t1[i]);  /* hk = fn+silu(t1) */
        memset(lg,0,sizeof(float)*(size_t)N*V);
        mm_bt(lg, hk, w->emb, N, V, D);                         /* logits_k = hk@emb^T */
        #pragma omp parallel for schedule(static)
        for(int n=0;n<N;n++){
            float *l=lg+(size_t)n*V, *p=pr+(size_t)n*V, *d=dl+(size_t)n*V;
            float mx=-1e30f; for(int i=0;i<V;i++) if(l[i]>mx)mx=l[i];
            float s=0; for(int i=0;i<V;i++){ p[i]=expf(l[i]-mx); s+=p[i]; }
            float inv=1.0f/s, scale=wk/N;
            for(int i=0;i<V;i++) d[i]=p[i]*inv*scale;
            d[tgt[n]] -= scale;
        }
        mm_atb(gr->emb, dl, hk, N, V, D);                      /* dEmb += dl^T@hk */
        memset(dhk,0,sizeof(float)*(size_t)N*D);
        mm(dhk, dl, w->emb, N, V, D);                          /* dhk = dl@emb */
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<(size_t)N*D;i++) a->dfn[i]+=dhk[i];   /* residual path */
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N*D;i++){ float z=t1[i]; float sig=1.0f/(1.0f+expf(-z));
            dt1[i]=dhk[i]*(sig+z*sig*(1.0f-sig)); }
        mm_atb(gr->mtp[k], fn, dt1, N, D, D);                 /* dWk += fn^T@dt1 */
        mm_bt(a->dfn, dt1, w->mtp[k], N, D, D);               /* dfn += dt1@Wk^T */
    }
}

/* ----------------------------- backward --------------------------------- */
static void backward(const Cfg *c, const Weights *w, Weights *gr, Acts *a,
                     const int *tok, const int *targets, int *const *tgt_mtp){
    int N=a->N, D=c->D, F=c->F, KD=c->KD, V=c->V;
    /* dlogits = (probs - onehot)/N */
    #pragma omp parallel for schedule(static)
    for(int n=0;n<N;n++){
        float *dl=a->dlogits+(size_t)n*V, *pr=a->probs+(size_t)n*V;
        float invN=1.0f/N;
        for(int i=0;i<V;i++) dl[i]=pr[i]*invN;
        dl[targets[n]] -= invN;
    }
    /* logits = fn @ emb^T :  dfn += dlogits@emb ; dEmb += dlogits^T @ fn */
    memset(a->dfn, 0, sizeof(float)*(size_t)N*D);
    mm(a->dfn, a->dlogits, w->emb, N, V, D);          /* dfn[n,d]=sum_v dl*emb */
    mm_atb(gr->emb, a->dlogits, a->fn_norm, N, V, D); /* dEmb[v,d]+=sum_n dl*fn */
    /* MTP heads contribute to dfn / gr->emb / gr->mtp before the trunk backward */
    if(c->n_mtp>0 && tgt_mtp) mtp_grad(c,w,gr,a,tgt_mtp);
    /* final rmsnorm backward -> dout (grad of x_final) */
    memset(a->dout, 0, sizeof(float)*(size_t)N*D);
    rmsnorm_bwd(a->x_final, w->nf, a->fn_rinv, a->dfn, a->dout, gr->nf, N, D);

    for(int l=c->L-1;l>=0;l--){
        /* dout = grad of layer output `out` = f_in + ffn(f_norm) */
        /* ffn backward -> a->dfnorm (grad of f_norm) */
        if(c->n_exp>0){
            moe_backward(c,w,gr,l, a->f_norm[l], a->dout, a->dfnorm, a->assign[l], a->gate[l], a->rprobs[l], N);
        } else {
            float *acttmp=a->du;
            #pragma omp parallel for schedule(static)
            for(size_t i=0;i<(size_t)N*F;i++) acttmp[i]=siluf(a->g[l][i])*a->u[l][i];
            if(g_act8) qlm_act8(acttmp, N, F);
            mm_atb(gr->w2[l], acttmp, a->dout, N, F, D);       /* dW2 += act^T@dout */
            memset(a->dact,0,sizeof(float)*(size_t)N*F);
            mm_bt(a->dact, a->dout, w->w2[l], N, F, D);        /* d_act */
            #pragma omp parallel for schedule(static)
            for(size_t i=0;i<(size_t)N*F;i++){
                float z=a->g[l][i]; float sig=1.0f/(1.0f+expf(-z));
                float silu=z*sig; float dsilu=sig+z*sig*(1.0f-sig);
                a->dg[i]=a->dact[i]*a->u[l][i]*dsilu;
                a->du[i]=a->dact[i]*silu;
            }
            memset(a->dfnorm,0,sizeof(float)*(size_t)N*D);
            mm_bt(a->dfnorm, a->dg, w->w1[l], N, D, F);
            mm_bt(a->dfnorm, a->du, w->w3[l], N, D, F);
            mm_atb(gr->w1[l], a->f_norm[l], a->dg, N, D, F);
            mm_atb(gr->w3[l], a->f_norm[l], a->du, N, D, F);
        }
        /* dfin = dout (residual) + rmsnorm2_back(f_in, dfnorm) */
        memcpy(a->dfin, a->dout, sizeof(float)*(size_t)N*D);
        rmsnorm_bwd(a->f_in[l], w->an2[l], a->f_rinv[l], a->dfnorm, a->dfin, gr->an2[l], N, D);
        /* attn_proj = att_ctx@Wo ; d_att_ctx, dWo */
        memset(a->datt,0,sizeof(float)*(size_t)N*D);
        mm_bt(a->datt, a->dfin, w->wo[l], N, D, D);
        mm_atb(gr->wo[l], a->att_ctx[l], a->dfin, N, D, D);
        /* attention backward -> dq,dk,dv */
        memset(a->dq,0,sizeof(float)*(size_t)N*D);
        memset(a->dk,0,sizeof(float)*(size_t)N*KD);
        memset(a->dv,0,sizeof(float)*(size_t)N*KD);
        attn_bwd(c, a, l);
        /* rope backward on dq (H) and dk (KV) */
        rope_apply(a->dq, a->pos, N, c->H,  c->hd, c->half, 1);
        rope_apply(a->dk, a->pos, N, c->KV, c->hd, c->half, 1);
        /* qkv proj backward */
        memset(a->dnorm,0,sizeof(float)*(size_t)N*D);
        mm_bt(a->dnorm, a->dq, w->wq[l], N, D, D);
        mm_bt(a->dnorm, a->dk, w->wk[l], N, D, KD);
        mm_bt(a->dnorm, a->dv, w->wv[l], N, D, KD);
        mm_atb(gr->wq[l], a->a_norm[l], a->dq, N, D, D);
        mm_atb(gr->wk[l], a->a_norm[l], a->dk, N, D, KD);
        mm_atb(gr->wv[l], a->a_norm[l], a->dv, N, D, KD);
        /* rmsnorm1 backward: dain = dfin(residual) + norm_back */
        memcpy(a->dain, a->dfin, sizeof(float)*(size_t)N*D);
        rmsnorm_bwd(a->a_in[l], w->an1[l], a->a_rinv[l], a->dnorm, a->dain, gr->an1[l], N, D);
        /* pass to previous layer */
        memcpy(a->dout, a->dain, sizeof(float)*(size_t)N*D);
    }
    /* embedding backward: dout is grad of a_in[0]=emb rows (single-threaded scatter) */
    for(int n=0;n<N;n++){
        float *de=gr->emb+(size_t)tok[n]*D;
        const float *d=a->dout+(size_t)n*D;
        for(int i=0;i<D;i++) de[i]+=d[i];
    }
}

/* ----------------------------- adamw ------------------------------------ */
/* gscale folds gradient clipping in (caller passes 1/gradnorm when clipping),
 * saving a separate full pass over the param array each step. */
static void adamw(float *P, float *G, float *M, float *Vv, size_t n,
                  float lr, float b1, float b2, float wd, int t, float gscale){
    float bc1=1.0f-powf(b1,t), bc2=1.0f-powf(b2,t);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<(long long)n;i++){
        float g=G[i]*gscale;
        M[i]=b1*M[i]+(1-b1)*g;
        Vv[i]=b2*Vv[i]+(1-b2)*g*g;
        float mh=M[i]/bc1, vh=Vv[i]/bc2;
        P[i]-=lr*(mh/(sqrtf(vh)+1e-8f) + wd*P[i]);
    }
}
static float global_gradnorm(const float *G, size_t n){
    double s=0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for(long long i=0;i<(long long)n;i++) s+=(double)G[i]*G[i];
    return (float)sqrt(s);
}

/* ===================== qlm: all-integer training ========================= *
 * Experimental training method (see qlm-plan.md). For every 2D weight matrix
 * (wq/wk/wv/wo and every expert w1/w3/w2) the ONLY persistent state is integer:
 *
 *   d[i]  int8 in [-dmax,+dmax]   the weight digit  (dmax 1 = ternary, 7 = 4-bit)
 *   r[i]  int8 in [-64,+64]       sub-digit residual -- an error-feedback
 *                                 accumulator, so updates smaller than one digit
 *                                 are not lost to rounding. This is the job fp32
 *                                 master weights normally do, in 1 byte not 4.
 *   s[g]  fp32                    one scale per group of QLM_GROUP elements
 *
 * The float weight the forward pass sees is the HARD quantization w = s*d
 * (i.e. a genuinely 2-4 bit model, deployable as-is). The residual is latent
 * only. Straight-through estimation means the gradient wrt the latent equals
 * the gradient wrt w, so backward() is untouched -- this is a training-method
 * change, not an architecture change.
 *
 * The update is Integer Lion: Lion's output is sign(.), already an integer, so
 * a step is a plain integer increment of r and no gradient scale is ever needed
 * (which is what makes Adam impossible here and Lion natural).
 *
 * P (the flat fp32 parameter buffer) is a RENDERED VIEW of (d,r,s) for these
 * spans -- rewritten from the digits after every step, never itself updated.
 * On this CPU trial it is still allocated; on the GPU it would not be, which is
 * where the 7x memory saving comes from. `qlm footprint` reports both numbers.
 */

typedef struct { size_t off, n; } QSpan;      /* span of the flat P buffer */
typedef struct { size_t qi, pof; int n, sp; } QGrp;  /* one scale group     */

typedef struct {
    int on, dither, dmax, group, mombits, lrmode, servo_every;
    float fstar, f_hi, f_lo;
    int    nqs, nfs;
    QSpan *qs, *fs;          /* quantized spans; float spans (the complement) */
    size_t nq, nf, np, ngrp;
    QGrp  *g;
    int8_t *d, *r;
    float  *s, *sref;        /* sref: per-span reference scale, frozen at init */
    int8_t *m8; float *m8s;  /* int8 momentum (mombits==8); else M is used     */
    int    adam;             /* 1 = Integer AdamW instead of Integer Lion      */
    uint8_t *v8; float *v8s; /* int8 second moment, stored as sqrt(v)          */
    long long n_carry, n_sat, n_dead, n_servo_up, n_servo_dn;
    double sum_absr;
} Qlm;

static double g_max_sec = 0.0;   /* TINYLM_MAX_SECONDS: wall-clock training cap */
static int    g_qadam = 0;       /* QLM_OPT=qadam: Integer AdamW, not Integer Lion */
static Qlm  g_qlm;
static Qlm *g_q = NULL;          /* NULL => stock AdamW path, bit-for-bit */
static int  g_opt_lion = 0;      /* fp32 Lion on everything (the middle rung) */
static float g_wd_over = -1.0f;

static inline uint32_t qhash(uint64_t x){
    x += 0x9E3779B97F4A7C15ULL; x ^= x>>30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x>>27; x *= 0x94D049BB133111EBULL; x ^= x>>31; return (uint32_t)x;
}
/* Stochastic rounding: floor(v), plus 1 with probability frac(v). Keyed on
 * (index, step) rather than a global RNG so it is thread-safe under OpenMP AND
 * reproducible across runs -- which is what makes the A/B protocol meaningful. */
static inline int qsr(float v, uint64_t i, uint64_t t){
    float f = floorf(v);
    uint32_t h = qhash(i*0x2545F4914F6CDD1DULL + t*0x9E3779B97F4A7C15ULL);
    return (int)f + (((h>>8)*(1.0f/16777216.0f) < (v-f)) ? 1 : 0);
}
static inline int qclampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

/* quantize one group of floats into (d,r) at scale sg */
static void qgrp_encode(const float *W, int n, float sg, int dmax, int8_t *d, int8_t *r){
    float inv = 1.0f/sg;
    for(int j=0;j<n;j++){
        float t = W[j]*inv;
        int dd = qclampi((int)lrintf(t), -dmax, dmax);
        int rr = qclampi((int)lrintf((t-(float)dd)*128.0f), -64, 64);
        d[j]=(int8_t)dd; r[j]=(int8_t)rr;
    }
}
/* scale for a group: absmean for ternary (BitNet b1.58), absmax/dmax above it */
static float qgrp_scale(const float *W, int n, int dmax){
    double amax=0, asum=0;
    for(int j=0;j<n;j++){ double a=fabs((double)W[j]); if(a>amax)amax=a; asum+=a; }
    float sg = (dmax==1) ? (float)(asum/n) : (float)(amax/dmax);
    if(!(sg>0)) sg=1e-8f;
    return sg;
}

static void qspan_add(Qlm *q, ptrdiff_t off, size_t n){
    q->qs[q->nqs].off=(size_t)off; q->qs[q->nqs].n=n; q->nqs++;
}

/* Build the span table, quantize P in place, and freeze the per-span reference
 * scale used by the `rel` LR mode. Returns 0 if qlm is not enabled. */
static int qlm_init(Qlm *q, const Cfg *c, const Weights *w, float *P, size_t np){
    memset(q,0,sizeof(*q));
    const char *e;
    /* read the mode HERE: the memset above clears anything the caller set, and
     * the second-moment buffer is sized below, so a flag assigned afterwards
     * arrives too late and leaves v8 NULL (segfault on the first step). */
    { const char *o=getenv("QLM_OPT"); q->adam = (o && !strcmp(o,"qadam")); }
    int bits = (e=getenv("QLM_BITS")) ? atoi(e) : 4;
    q->dmax  = bits<=2?1 : bits==3?3 : bits==4?7 : 127;
    q->group = (e=getenv("QLM_GROUP")) ? atoi(e) : 128;
    if(q->group<16) q->group=16; if(q->group>512) q->group=512;
    q->mombits = (e=getenv("QLM_MOM")) ? atoi(e) : 32;
    if(q->mombits!=8) q->mombits=32;
    q->lrmode  = (e=getenv("QLM_LRMODE")) ? (!strcmp(e,"abs")?1:0) : 0;
    q->dither  = (e=getenv("QLM_DITHER")) ? atoi(e) : 0;
    q->servo_every = (e=getenv("QLM_SERVO")) ? atoi(e) : 100;
    /* Target rail fraction. For 4-bit an absmax-ish scale rails ~2% of a
     * Gaussian; for ternary the rail IS the normal state (absmean ternary puts
     * roughly 2/3 of weights at +-1), so a single constant here would drive the
     * servo the wrong way at dmax=1. */
    /* Init uses absmax/dmax, which rails exactly one element per group, so the
     * target has to be ~1/group or the servo shrinks every scale on its first
     * tick (observed: 808 of 1440 groups scaled down at once). Ternary is the
     * exception -- there the rail IS the normal state. */
    q->fstar = (e=getenv("QLM_FSTAR")) ? (float)atof(e)
             : (q->dmax==1 ? 0.60f : 1.5f/(float)q->group);
    q->f_hi = q->fstar*1.5f; q->f_lo = q->fstar*0.5f;

    int E = n_slots(c);
    q->qs = malloc(sizeof(QSpan)*(size_t)c->L*7);
    for(int l=0;l<c->L;l++){
        qspan_add(q, w->wq[l]-P, (size_t)c->D*c->D);
        qspan_add(q, w->wk[l]-P, (size_t)c->D*c->KD);
        qspan_add(q, w->wv[l]-P, (size_t)c->D*c->KD);
        qspan_add(q, w->wo[l]-P, (size_t)c->D*c->D);
        qspan_add(q, w->w1[l]-P, (size_t)E*c->D*c->F);
        qspan_add(q, w->w3[l]-P, (size_t)E*c->D*c->F);
        qspan_add(q, w->w2[l]-P, (size_t)E*c->F*c->D);
    }
    /* spans come out of map_weights in ascending order; the float spans are the
     * gaps (embeddings, both norms, the router, the final norm, MTP heads) */
    q->np = np;
    q->fs = malloc(sizeof(QSpan)*(q->nqs+2));
    size_t cur=0;
    for(int i=0;i<q->nqs;i++){
        if(q->qs[i].off>cur){ q->fs[q->nfs].off=cur; q->fs[q->nfs].n=q->qs[i].off-cur; q->nfs++; }
        cur = q->qs[i].off + q->qs[i].n;
        q->nq += q->qs[i].n;
    }
    if(cur<np){ q->fs[q->nfs].off=cur; q->fs[q->nfs].n=np-cur; q->nfs++; }
    for(int i=0;i<q->nfs;i++) q->nf += q->fs[i].n;

    for(int i=0;i<q->nqs;i++) q->ngrp += (q->qs[i].n + q->group - 1)/q->group;
    q->g    = malloc(sizeof(QGrp)*q->ngrp);
    q->d    = malloc(q->nq); q->r = malloc(q->nq);
    q->s    = malloc(sizeof(float)*q->ngrp);
    q->sref = malloc(sizeof(float)*q->nqs);
    if(q->mombits==8){ q->m8=calloc(q->nq,1); q->m8s=calloc(q->ngrp,sizeof(float));
        if(q->adam){ q->v8=calloc(q->nq,1); q->v8s=calloc(q->ngrp,sizeof(float)); } }

    size_t qi=0, gi=0;
    for(int i=0;i<q->nqs;i++){
        double ssum=0; size_t g0=gi;
        for(size_t o=0;o<q->qs[i].n;o+=q->group){
            int n = (int)((q->qs[i].n-o < (size_t)q->group) ? q->qs[i].n-o : (size_t)q->group);
            q->g[gi].qi=qi; q->g[gi].pof=q->qs[i].off+o; q->g[gi].n=n; q->g[gi].sp=i;
            float sg = qgrp_scale(P+q->g[gi].pof, n, q->dmax);
            q->s[gi]=sg; ssum+=sg;
            qgrp_encode(P+q->g[gi].pof, n, sg, q->dmax, q->d+qi, q->r+qi);
            qi+=n; gi++;
        }
        q->sref[i] = (float)(ssum/(double)(gi-g0));
    }
    q->on=1;
    return 1;
}

/* rewrite the quantized spans of P from the digits. This is the only way P is
 * ever written for those spans -- it is a view, not a master copy. */
static void qlm_render(const Qlm *q, float *P){
    #pragma omp parallel for schedule(static)
    for(long long gi=0; gi<(long long)q->ngrp; gi++){
        const QGrp *g=&q->g[gi]; float sg=q->s[gi];
        if(q->dither) for(int j=0;j<g->n;j++)
            P[g->pof+j] = sg*((float)q->d[g->qi+j] + (float)q->r[g->qi+j]*(1.0f/128.0f));
        else for(int j=0;j<g->n;j++)
            P[g->pof+j] = sg*(float)q->d[g->qi+j];
    }
}

/* plain fp32 Lion over a span (used for the un-quantized parameters, and for
 * the whole model in QLM_OPT=lion). M is the existing AdamW momentum buffer. */
static void lion_span(float *P, const float *G, float *M, size_t off, size_t n,
                      float lr, float b1, float b2, float wd, float gscale){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<(long long)n;i++){
        size_t k=off+(size_t)i;
        float g=G[k]*gscale;
        float cc=b1*M[k]+(1.0f-b1)*g;
        float u = cc>0.0f?1.0f:(cc<0.0f?-1.0f:0.0f);
        P[k] -= lr*(u + wd*P[k]);
        M[k] = b2*M[k]+(1.0f-b2)*g;
    }
}

/* Integer Lion over the quantized spans + render. */
/* One element of the update. Written as a macro so ADAM and MOM8 arrive as
 * compile-time constants: previously both were branches on `q->` fields INSIDE
 * the inner loop, which alone stops any vectorizer. Everything else here exists
 * to keep the loop a straight-line, unit-stride body:
 *   - the carry `while` loops become a single branchless correction, which is
 *     exact because idl is clamped to +-64 so |rr| <= 128 and one carry always
 *     suffices (rr=128 -> c=1 -> rr=0; rr=64 -> c=0, matching `while(rr>64)`)
 *   - ONE sqrt instead of two: sqrt(vv/bc2) == sqrt(vv) * (1/sqrt(bc2)), and
 *     sqrt(vv) is needed anyway to store the second moment in the sqrt domain
 *   - the bias-correction divides become multiplies by hoisted reciprocals
 *   - `dither` and the momentum scales are hoisted out of the loop
 * Stats accumulate as integers (|rr| <= 64, so the sum is exact). */
#define QLM_BODY(ADAM, MOM8)                                                    \
    for(int j=0;j<n;j++){                                                       \
        size_t qi=qi0+(size_t)j, k=pof+(size_t)j;                               \
        float gr=G[k]*gscale;                                                   \
        float mv = MOM8 ? (float)m8[qi]*ms_old : M[k];                          \
        float u, mn;                                                            \
        if(ADAM){                                                               \
            float vv;                                                           \
            if(MOM8){ float sv=(float)v8[qi]*vs_old; vv=sv*sv; }                \
            else      vv=Vv[k];                                                 \
            mn = b1*mv + (1.0f-b1)*gr;                                          \
            vv = b2*vv + (1.0f-b2)*gr*gr;                                       \
            float rv = sqrtf(vv);                                               \
            u  = (mn*inv_bc1) / (rv*inv_sqrt_bc2 + 1e-8f);                      \
            if(MOM8){ tv[j]=rv; if(rv>vmax)vmax=rv; } else Vv[k]=vv;            \
        } else {                                                                \
            float cc = b1*mv + (1.0f-b1)*gr;                                    \
            u  = (float)((cc>0.0f)-(cc<0.0f));                                  \
            mn = b2*mv + (1.0f-b2)*gr;                                          \
        }                                                                       \
        if(MOM8){ tmp[j]=mn; float a=fabsf(mn); if(a>amax)amax=a; }             \
        else M[k]=mn;                                                           \
        int dd=(int)dq[qi];                                                     \
        int idl = qsr(-(L*u + dec*(float)dd), qi, (uint64_t)t);                 \
        int rr = (int)rq[qi] + idl;                                             \
        /* exact multi-carry, no loop and no clamp: the unique c with           \
         * rr-128c in [-64,64], matching `while(rr>64){rr-=128;}` at the        \
         * boundary (rr==64 must NOT carry). An earlier version clamped idl to  \
         * +-64 and assumed one carry always sufficed -- wrong: L is ~48 LSBs   \
         * at these learning rates and Adam's u exceeds 1 while v is small, so  \
         * real steps do overflow a single carry. */                            \
        int c = (rr > 64) ?  (((rr - 65) >> 7) + 1)                             \
              : (rr < -64) ? -((((-rr) - 65) >> 7) + 1) : 0;                    \
        rr -= c*128; dd += c;                                                   \
        ncar += (c<0?-c:c);                                                     \
        dd = dd<-dmax?-dmax:(dd>dmax?dmax:dd);                                  \
        nsat += (dd==dmax)|(dd==-dmax);                                         \
        ndead += (dd==0);                                                       \
        sabs += (rr<0?-rr:rr);                                                  \
        dq[qi]=(int8_t)dd; rq[qi]=(int8_t)rr;                                   \
        P[k] = dith ? sg*((float)dd + (float)rr*(1.0f/128.0f)) : sg*(float)dd;  \
    }

static void qlm_step(Qlm *q, float *restrict P, const float *restrict G,
                     float *restrict M, float *restrict Vv,
                     float lr, float b1, float b2, float wd, float gscale, int t){
    /* Integer AdamW. The plan claimed Adam could not drive an integer
     * accumulator without reintroducing a float gradient scale -- that was
     * wrong: the per-group scale s IS that scale, and is already stored. Adam's
     * update converts to LSBs exactly the way Lion's does, the only difference
     * being that `upd` is m_hat/(sqrt(v_hat)+eps) instead of sign(). Everything
     * downstream (carry, clamp, decay, render) is unchanged. */
    const int   adam = q->adam, mom8 = (q->mombits==8), dith = q->dither;
    const int   dmax = q->dmax;
    const float bc1 = adam ? 1.0f-powf(b1,(float)t) : 1.0f;
    const float bc2 = adam ? 1.0f-powf(b2,(float)t) : 1.0f;
    const float inv_bc1 = 1.0f/bc1, inv_sqrt_bc2 = 1.0f/sqrtf(bc2);
    /* Weight update in weight units is  dw = -lr*(u + wd*w),  w = s*d, and one
     * LSB of r is s/128, so in LSB units:  dr = -(lr*128/s)*u - lr*wd*128*d.
     * The decay term is scale-free; only the sign term needs a scale. */
    const float dec = lr*wd*128.0f;
    int8_t  *restrict dq = q->d, *restrict rq = q->r, *restrict m8 = q->m8;
    uint8_t *restrict v8 = q->v8;
    long long carry=0, sat=0, dead=0, absr=0;
    #pragma omp parallel for schedule(static) reduction(+:carry,sat,dead,absr)
    for(long long gi=0; gi<(long long)q->ngrp; gi++){
        const QGrp *g=&q->g[gi];
        const float sg=q->s[gi];
        const int   n=g->n;
        const size_t qi0=g->qi, pof=g->pof;
        /* both scales are overwritten after the loop, so capture the values the
         * update must read (the previous step's) up front */
        const float ms_old = mom8 ? q->m8s[gi] : 0.0f;
        const float vs_old = (mom8 && adam) ? q->v8s[gi] : 0.0f;
        /* `rel` (default): the step is a fixed number of LSBs per tensor, so the
         * optimizer really does move digits at a uniform rate and the effective
         * fp learning rate follows each group's own scale. `abs` reproduces
         * float Lion's uniform-in-weight-units step exactly. */
        const float L = lr*128.0f/(q->lrmode ? sg : q->sref[g->sp]);
        float tmp[512], tv[512]; float amax=0.0f, vmax=0.0f;
        long long ncar=0, nsat=0, ndead=0, sabs=0;

        if(adam){ if(mom8) { QLM_BODY(1,1) } else { QLM_BODY(1,0) } }
        else    { if(mom8) { QLM_BODY(0,1) } else { QLM_BODY(0,0) } }

        carry+=ncar; sat+=nsat; dead+=ndead; absr+=sabs;

        if(mom8){                             /* blockwise int8 momentum, SR */
            float ms = amax>0 ? amax/127.0f : 1e-12f;
            q->m8s[gi]=ms;
            float inv=1.0f/ms;
            for(int j=0;j<n;j++)
                m8[qi0+(size_t)j] = (int8_t)qclampi(qsr(tmp[j]*inv, qi0+(size_t)j, (uint64_t)t+0x5EEDULL), -127,127);
            if(adam){
                float vs = vmax>0 ? vmax/255.0f : 1e-12f;
                q->v8s[gi]=vs; float vinv=1.0f/vs;
                for(int j=0;j<n;j++)
                    v8[qi0+(size_t)j] = (uint8_t)qclampi(qsr(tv[j]*vinv, qi0+(size_t)j, (uint64_t)t+0xBEEFULL), 0,255);
            }
        }
    }
    q->n_carry=carry; q->n_sat=sat; q->n_dead=dead; q->sum_absr=(double)absr;
}

/* Scale servo. With no latent float weights there is nothing to recompute
 * mean|W| from, so each group's scale is steered by how often it rails. */
static void qlm_servo(Qlm *q){
    long long up=0, dn=0;
    #pragma omp parallel for schedule(static) reduction(+:up,dn)
    for(long long gi=0; gi<(long long)q->ngrp; gi++){
        const QGrp *g=&q->g[gi];
        int rail=0;
        for(int j=0;j<g->n;j++){ int dd=q->d[g->qi+j]; if(dd==q->dmax||dd==-q->dmax) rail++; }
        float f=(float)rail/(float)g->n, sg=q->s[gi], sn;
        if(f>q->f_hi){ sn=sg*1.09051f; up++; }        /* 2^(1/8) */
        else if(f<q->f_lo){ sn=sg/1.09051f; dn++; }
        else continue;
        /* redistribute the digits at the new scale -- an integer operation, it
         * never materializes a float master weight */
        float conv=sg/sn;
        for(int j=0;j<g->n;j++){
            size_t qi=g->qi+j;
            float t=((float)q->d[qi] + (float)q->r[qi]*(1.0f/128.0f))*conv;
            int dd=qclampi((int)lrintf(t), -q->dmax, q->dmax);
            int rr=qclampi((int)lrintf((t-(float)dd)*128.0f), -64, 64);
            q->d[qi]=(int8_t)dd; q->r[qi]=(int8_t)rr;
        }
        q->s[gi]=sn;
    }
    q->n_servo_up=up; q->n_servo_dn=dn;
}

/* one optimizer step for the whole model */
static void qlm_update(Qlm *q, float *P, const float *G, float *M, float *Vv, size_t np,
                       float lr, float b1, float b2, float wd, float gscale, int t){
    if(!q || !q->on){                       /* fp32 Lion on everything */
        lion_span(P,G,M,0,np,lr,b1,b2,wd,gscale);
        return;
    }
    /* un-quantized spans (embeddings, norms, router, MTP) follow the same
     * optimizer family as the quantized ones, so the comparison isolates the
     * representation and not a mixture of two update rules */
    for(int i=0;i<q->nfs;i++){
        if(q->adam) adamw(P+q->fs[i].off,(float*)G+q->fs[i].off,M+q->fs[i].off,
                          Vv+q->fs[i].off,q->fs[i].n,lr,b1,b2,wd,t,gscale);
        else        lion_span(P,G,M,q->fs[i].off,q->fs[i].n,lr,b1,b2,wd,gscale);
    }
    qlm_step(q,P,G,M,Vv,lr,b1,b2,wd,gscale,t);
    if(q->servo_every>0 && (t % q->servo_every)==0) qlm_servo(q);
}

static void qlm_report(const Qlm *q, size_t np){
    /* bits needed for one digit in [-dmax,+dmax]: 2*dmax+1 levels. The old
     * form special-cased only dmax 1 and 7, so dmax=3 (3-bit) was reported as
     * a full byte and overstated the state by ~0.45 B/param. */
    double dbits = ceil(log2((double)(2*q->dmax+1)));
    double packed = (double)q->nq*(dbits/8.0)                            /* digits */
                  + (double)q->nq                                        /* residual */
                  + (double)q->ngrp*2.0                                  /* fp16 scales */
                  + (double)q->nq*(q->mombits==8?1.0:4.0)*(q->adam?2.0:1.0)  /* m (+v for Adam) */
                  + (double)q->nf*8.0;                                   /* fp32 float spans + mom */
    printf("[qlm] optimizer=%s\n", q->adam?"Integer AdamW":"Integer Lion");
    printf("[qlm] %zu of %zu params quantized (%.1f%%) in %zu groups | bits=%d group=%d mom=%d lr=%s%s\n",
           q->nq, np, 100.0*q->nq/np, q->ngrp,
           q->dmax==1?2:q->dmax==3?3:q->dmax==7?4:8, q->group, q->mombits,
           q->lrmode?"abs":"rel", q->dither?" DITHER":"");
    printf("[qlm] training state %.2f B/param packed (%.2f MB) vs AdamW's 16.00 B/param (%.2f MB)\n",
           packed/np, packed/1048576.0, np*16.0/1048576.0);
}
static void qlm_stats(const Qlm *q){
    printf("[qlm] carry %.2f%%/step | rail %.1f%% | zero %.1f%% | mean|r| %.1f | servo +%lld/-%lld\n",
           100.0*(double)q->n_carry/(double)q->nq, 100.0*(double)q->n_sat/(double)q->nq,
           100.0*(double)q->n_dead/(double)q->nq, q->sum_absr/(double)q->nq,
           q->n_servo_up, q->n_servo_dn);
}

static void qlm_save(const char *path, const Qlm *q){
    FILE *f=fopen(path,"wb"); if(!f) return;
    int hdr[6]={q->dmax,q->group,q->mombits,(int)q->nqs,(int)0,(int)0};
    fwrite("QLM1",1,4,f); fwrite(hdr,4,6,f);
    uint64_t nq=q->nq, ng=q->ngrp; fwrite(&nq,8,1,f); fwrite(&ng,8,1,f);
    fwrite(q->d,1,q->nq,f); fwrite(q->r,1,q->nq,f);
    fwrite(q->s,sizeof(float),q->ngrp,f);
    if(q->mombits==8){ fwrite(q->m8,1,q->nq,f); fwrite(q->m8s,sizeof(float),q->ngrp,f); }
    fclose(f);
}
static int qlm_load(const char *path, Qlm *q){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    char m[4]; int hdr[6]; uint64_t nq,ng;
    if(fread(m,1,4,f)!=4 || memcmp(m,"QLM1",4)){ fclose(f); return 0; }
    if(fread(hdr,4,6,f)!=6 || fread(&nq,8,1,f)!=1 || fread(&ng,8,1,f)!=1){ fclose(f); return 0; }
    if(hdr[0]!=q->dmax || hdr[1]!=q->group || hdr[2]!=q->mombits
       || nq!=q->nq || ng!=q->ngrp){
        fprintf(stderr,"[qlm] %s has a different shape (bits/group/mom) -- ignoring\n",path);
        fclose(f); return 0; }
    if(fread(q->d,1,q->nq,f)!=q->nq || fread(q->r,1,q->nq,f)!=q->nq
       || fread(q->s,sizeof(float),q->ngrp,f)!=q->ngrp){ fclose(f); return 0; }
    if(q->mombits==8){ if(fread(q->m8,1,q->nq,f)!=q->nq) { fclose(f); return 0; }
                       if(fread(q->m8s,sizeof(float),q->ngrp,f)!=q->ngrp){ fclose(f); return 0; } }
    fclose(f); return 1;
}

/* ----------------------------- data ------------------------------------- */
static uint16_t *load_u16(const char *path, size_t *count){
    FILE *f=fopen(path,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",path); exit(1); }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint16_t *d=malloc(sz); if(fread(d,1,sz,f)!=(size_t)sz){fprintf(stderr,"read fail\n");exit(1);} fclose(f);
    *count=sz/2; return d;
}

/* ----------------------------- tokenizer -------------------------------- */
typedef struct {
    int V, eot, nmerges;
    uint8_t **dec; int *declen;               /* decode table */
    int base[256];
    /* sparse merge map (gen only): open-addressing hash keyed by a*V+b */
    uint64_t *mkey; int *mrank, *mnew; size_t mcap;
} Tok;

/* probe the sparse merge table; returns slot index (empty slot has mrank<0) */
static inline size_t tok_slot(const Tok *tk, uint64_t key){
    size_t mask=tk->mcap-1, i=(size_t)(key*0x9E3779B97F4A7C15ULL)&mask;
    while(tk->mrank[i]>=0 && tk->mkey[i]!=key) i=(i+1)&mask;
    return i;
}

/* read an entire file into a malloc'd buffer */
static uint8_t *read_file(const char *path, size_t *len){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *b=malloc(sz>0?sz:1);
    if(fread(b,1,sz,f)!=(size_t)sz){ fclose(f); free(b); return NULL; }
    fclose(f); *len=sz; return b;
}

/* sequential reader: memory (f==NULL) or file-backed streaming (f set, buf/cap
 * a refill window). File-backed avoids slurping the whole model into RAM during
 * load -- see load_q4_lowpeak. Small sequential reads only; n must be <= cap. */
typedef struct { const uint8_t *p, *end; FILE *f; uint8_t *buf; size_t cap; } Rd;
static int rd_refill(Rd *r, size_t need){
    size_t rem=(size_t)(r->end - r->p);
    if(rem) memmove(r->buf, r->p, rem);
    size_t got=fread(r->buf+rem, 1, r->cap-rem, r->f);
    r->p=r->buf; r->end=r->buf+rem+got;
    return (size_t)(r->end - r->p) >= need;
}
static int rd(Rd *r, void *dst, size_t n){
    if(r->f && (size_t)(r->end - r->p) < n && !rd_refill(r,n)) return 0;
    if(r->p + n > r->end) return 0;
    memcpy(dst, r->p, n); r->p += n; return 1;
}
static int rd_skip(Rd *r, size_t n){
    if(r->p + n > r->end) return 0;
    r->p += n; return 1;
}

/* parse a TLTK tokenizer blob from memory (copies what it keeps) */
static Tok *tok_parse(const uint8_t *data, size_t len, int full){
    Rd r={data, data+len}; char magic[4];
    if(!rd(&r,magic,4) || memcmp(magic,"TLTK",4)){ fprintf(stderr,"bad tok magic\n"); exit(1); }
    Tok *tk=calloc(1,sizeof(Tok));
    if(!rd(&r,&tk->V,4)||!rd(&r,&tk->eot,4)||!rd(&r,&tk->nmerges,4)){fprintf(stderr,"tok hdr\n");exit(1);}
    tk->dec=malloc(sizeof(uint8_t*)*tk->V); tk->declen=malloc(sizeof(int)*tk->V);
    for(int i=0;i<tk->V;i++){ int l; if(!rd(&r,&l,4)){fprintf(stderr,"tok dec\n");exit(1);}
        tk->declen[i]=l; tk->dec[i]=malloc(l>0?l:1);
        if(l>0 && !rd(&r,tk->dec[i],l)){fprintf(stderr,"tok dec2\n");exit(1);} }
    if(!rd(&r,tk->base,4*256)){fprintf(stderr,"tok base\n");exit(1);}
    if(full){
        size_t cap=16; while(cap < (size_t)tk->nmerges*2) cap<<=1;
        tk->mcap=cap; tk->mkey=malloc(sizeof(uint64_t)*cap);
        tk->mrank=malloc(sizeof(int)*cap); tk->mnew=malloc(sizeof(int)*cap);
        for(size_t i=0;i<cap;i++){ tk->mrank[i]=-1; tk->mnew[i]=-1; }
        for(int m=0;m<tk->nmerges;m++){ int a,b,nw;
            if(!rd(&r,&a,4)||!rd(&r,&b,4)||!rd(&r,&nw,4)){fprintf(stderr,"tok merge\n");exit(1);}
            uint64_t key=(uint64_t)a*tk->V+b; size_t i=tok_slot(tk,key);
            if(tk->mrank[i]<0){ tk->mkey[i]=key; tk->mrank[i]=m; tk->mnew[i]=nw; } }
    }
    return tk;
}
static Tok *tok_load(const char *path, int full){
    size_t len; uint8_t *buf=read_file(path,&len);
    if(!buf){ fprintf(stderr,"cannot open %s\n",path); exit(1); }
    Tok *tk=tok_parse(buf,len,full); free(buf); return tk;
}
static int *tok_encode(const Tok *tk, const char *s, int *out_n){
    int n=strlen(s); int *ids=malloc(sizeof(int)*(n>0?n:1)); int cnt=0;
    for(int i=0;i<n;i++){ int b=(unsigned char)s[i]; int id=tk->base[b]; if(id<0) id=tk->base[(int)' ']; ids[cnt++]=id; }
    for(;;){ int best=-1, bi=-1;
        for(int i=0;i+1<cnt;i++){ size_t s=tok_slot(tk,(uint64_t)ids[i]*tk->V+ids[i+1]); int r=tk->mrank[s];
            if(r>=0 && (best<0 || r<best)){ best=r; bi=i; } }
        if(bi<0) break;
        size_t s=tok_slot(tk,(uint64_t)ids[bi]*tk->V+ids[bi+1]); ids[bi]=tk->mnew[s];
        for(int j=bi+1;j+1<cnt;j++) ids[j]=ids[j+1];
        cnt--;
    }
    *out_n=cnt; return ids;
}
static void tok_print(const Tok *tk, int id){
    if(id<0||id>=tk->V) return;
    fwrite(tk->dec[id],1,tk->declen[id],stdout);
}

/* ----------------------------- save/load model -------------------------- */
/* header sizes by magic: TLM1=8 ints, TLM2=9 (+tmpl), TLM3=10 (+n_mtp) */
static int hdr_ints(const char *m, int *v){  /* *v = version 1..6, returns #ints */
    if(!memcmp(m,"TLM6",4)){ *v=6; return 15; }   /* adds n_shared, qknorm */
    if(!memcmp(m,"TLM5",4)){ *v=5; return 13; }   /* adds win, full_every */
    if(!memcmp(m,"TLM4",4)){ *v=4; return 11; }
    if(!memcmp(m,"TLM3",4)){ *v=3; return 10; }
    if(!memcmp(m,"TLM2",4)){ *v=2; return 9;  }
    if(!memcmp(m,"TLM1",4)){ *v=1; return 8;  }
    *v=0; return 0;
}
/* Callers MUST size their header buffer for the largest hdr_ints() result (13).
 * An int hdr[11] here reads 52 bytes into 44 and corrupts the stack -- that
 * exact bug produced a 0xC0000409 in the CUDA build when TLM5 was added. */
#define HDR_MAX 16
static void cfg_from_hdr(Cfg *c, const int *hdr, int ver){
    c->V=hdr[0];c->D=hdr[1];c->L=hdr[2];c->H=hdr[3];c->KV=hdr[4];c->F=hdr[5];c->T=hdr[6];c->eot=hdr[7];
    c->tmpl  = ver>=2 ? hdr[8] : 0;
    c->n_mtp = ver>=3 ? hdr[9] : 0;
    c->n_exp = ver>=4 ? hdr[10] : 0;
    c->win        = ver>=5 ? hdr[11] : 0;
    c->full_every = ver>=5 ? hdr[12] : 0;
    c->n_shared   = ver>=6 ? hdr[13] : 0;
    c->qknorm     = ver>=6 ? hdr[14] : 0;
    c->rope_base=10000.0f; cfg_derive(c);
}
/* ---- fp16 <-> fp32 (portable software; weights are small normals) ---------- */
static uint16_t f32_to_f16(float x){
    uint32_t f; memcpy(&f,&x,4);
    uint32_t sign=(f>>16)&0x8000; int32_t exp=(int32_t)((f>>23)&0xff)-112; uint32_t mant=f&0x7fffff;
    if(((f>>23)&0xff)==0xff) return (uint16_t)(sign|0x7c00|(mant?0x200:0));  /* inf/nan */
    if(exp>=0x1f) return (uint16_t)(sign|0x7c00);                            /* overflow -> inf */
    if(exp<=0){                                                             /* subnormal/zero */
        if(exp<-10) return (uint16_t)sign;
        mant|=0x800000; int sh=14-exp; uint32_t h=mant>>sh;
        uint32_t rem=mant&(((uint32_t)1<<sh)-1), half=(uint32_t)1<<(sh-1);
        if(rem>half||(rem==half&&(h&1))) h++;
        return (uint16_t)(sign|h);
    }
    uint16_t h=(uint16_t)(sign|((uint32_t)exp<<10)|(mant>>13));
    uint32_t rem=mant&0x1fff;
    if(rem>0x1000||(rem==0x1000&&(h&1))) h++;     /* round-to-even; carry into exp is fine */
    return h;
}
static float f16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1f, mant=h&0x3ff, f;
    if(exp==0){
        if(mant==0) f=sign;
        else{ exp=113; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3ff; f=sign|(exp<<23)|(mant<<13); }
    } else if(exp==0x1f){ f=sign|0x7f800000|(mant<<13); }
    else f=sign|((exp+112)<<23)|(mant<<13);
    float out; memcpy(&out,&f,4); return out;
}
/* Selective q4 (stored qtype=3): quantize only the QAT-quantized spans
   (wq,wk,wv,wo and every expert w1/w3/w2, per layer) to 4-bit; store the gaps
   (embeddings, norms, router, final norm, MTP heads) as fp16. Quantizing those
   sensitive gaps to q4 -- norms sit near 1.0, the router decides top-k experts --
   destroys generation even though bulk weight error stays ~1%. This mirrors
   qlm_prepare's span selection exactly so PTQ matches the QAT layout. */
typedef struct { size_t off, n; } Q4Span;
/* Build the quantized spans in ascending param order (must match map_weights). */
static int q4_spans(const Cfg *c, Q4Span *qs){       /* qs holds >= L*7 */
    int E=n_slots(c), k=0; size_t p=(size_t)c->V*c->D;   /* skip emb */
    for(int l=0;l<c->L;l++){
        p+=c->D;                                                              /* an1 */
        qs[k].off=p; qs[k].n=(size_t)c->D*c->D;  k++; p+=(size_t)c->D*c->D;   /* wq */
        qs[k].off=p; qs[k].n=(size_t)c->D*c->KD; k++; p+=(size_t)c->D*c->KD;  /* wk */
        qs[k].off=p; qs[k].n=(size_t)c->D*c->KD; k++; p+=(size_t)c->D*c->KD;  /* wv */
        qs[k].off=p; qs[k].n=(size_t)c->D*c->D;  k++; p+=(size_t)c->D*c->D;   /* wo */
        if(c->qknorm){ p+=c->hd; p+=c->hd; }                                  /* qn,kn (fp) */
        p+=c->D;                                                              /* an2 */
        if(c->n_exp>0) p+=(size_t)c->D*c->n_exp;                              /* router (fp) */
        qs[k].off=p; qs[k].n=(size_t)E*c->D*c->F; k++; p+=(size_t)E*c->D*c->F;/* w1 */
        qs[k].off=p; qs[k].n=(size_t)E*c->D*c->F; k++; p+=(size_t)E*c->D*c->F;/* w3 */
        qs[k].off=p; qs[k].n=(size_t)E*c->F*c->D; k++; p+=(size_t)E*c->F*c->D;/* w2 */
    }
    return k;   /* nf + mtp stay in the trailing fp gap */
}
static uint8_t *q4_selective_encode(const float *P, size_t n, const Cfg *c, int group, size_t *out_len){
    Q4Span qs[1024]; int nq=q4_spans(c,qs);
    uint8_t *out=malloc(n*2+64); size_t o=0, cur=0;
    for(int i=0;i<nq;i++){
        for(size_t k=cur;k<qs[i].off;k++){ uint16_t h=f32_to_f16(P[k]); memcpy(out+o,&h,2); o+=2; }
        size_t so=qs[i].off, sn=qs[i].n;
        for(size_t j=0;j<sn;j+=group){
            int g=(j+(size_t)group<=sn)?group:(int)(sn-j);
            float amax=0; for(int t=0;t<g;t++){ float a=fabsf(P[so+j+t]); if(a>amax)amax=a; }
            float scale=amax>0?amax/7.0f:1.0f;
            uint16_t sh=f32_to_f16(scale); memcpy(out+o,&sh,2); o+=2;
            for(int t=0;t<g;t+=2){
                int v0=(int)lrintf(P[so+j+t]/scale); if(v0>7)v0=7; if(v0<-7)v0=-7;
                int v1=0; if(t+1<g){ v1=(int)lrintf(P[so+j+t+1]/scale); if(v1>7)v1=7; if(v1<-7)v1=-7; }
                out[o++]=(uint8_t)((v0&0xF)|((v1&0xF)<<4));
            }
        }
        cur=so+sn;
    }
    for(size_t k=cur;k<n;k++){ uint16_t h=f32_to_f16(P[k]); memcpy(out+o,&h,2); o+=2; }
    *out_len=o; return out;
}
static int q4_selective_decode(Rd *r, float *P, size_t n, const Cfg *c, int group){
    Q4Span qs[1024]; int nq=q4_spans(c,qs); size_t cur=0;
    for(int i=0;i<nq;i++){
        for(size_t k=cur;k<qs[i].off;k++){ uint16_t h; if(!rd(r,&h,2))return 0; P[k]=f16_to_f32(h); }
        size_t so=qs[i].off, sn=qs[i].n;
        for(size_t j=0;j<sn;j+=group){
            int g=(j+(size_t)group<=sn)?group:(int)(sn-j);
            uint16_t sh; if(!rd(r,&sh,2))return 0; float scale=f16_to_f32(sh);
            for(int t=0;t<g;t+=2){
                uint8_t b; if(!rd(r,&b,1))return 0;
                int lo=b&0xF; if(lo>=8)lo-=16; P[so+j+t]=(float)lo*scale;
                if(t+1<g){ int hi=(b>>4)&0xF; if(hi>=8)hi-=16; P[so+j+t+1]=(float)hi*scale; }
            }
        }
        cur=so+sn;
    }
    for(size_t k=cur;k<n;k++){ uint16_t h; if(!rd(r,&h,2))return 0; P[k]=f16_to_f32(h); }
    return 1;
}
/* Quantize P[n] -> malloc'd byte buffer (*out_len). qtype 0=fp16 (2B/w),
   1=q8 (Q8_0-style block: fp16 scale + int8 per `group` weights). */
static uint8_t *quantize_params(const float *P, size_t n, int qtype, int group, size_t *out_len){
    if(qtype==0){
        uint16_t *q=malloc(n*2>0?n*2:2);
        for(size_t i=0;i<n;i++) q[i]=f32_to_f16(P[i]);
        *out_len=n*2; return (uint8_t*)q;
    }
    if(qtype==2){  /* q4: fp16 scale + 4-bit symmetric [-7,7], 2 weights/byte (matches QAT/ngram) */
        size_t nb=(n+group-1)/group; uint8_t *q=malloc(nb*2+(n+1)/2+8); size_t o=0;
        for(size_t i=0;i<n;i+=group){
            int g=(i+(size_t)group<=n)?group:(int)(n-i);
            float amax=0; for(int j=0;j<g;j++){ float a=fabsf(P[i+j]); if(a>amax)amax=a; }
            float scale=amax>0?amax/7.0f:1.0f;
            uint16_t sh=f32_to_f16(scale); memcpy(q+o,&sh,2); o+=2;
            for(int j=0;j<g;j+=2){
                int v0=(int)lrintf(P[i+j]/scale); if(v0>7)v0=7; if(v0<-7)v0=-7;
                int v1=0; if(j+1<g){ v1=(int)lrintf(P[i+j+1]/scale); if(v1>7)v1=7; if(v1<-7)v1=-7; }
                q[o++]=(uint8_t)((v0&0xF)|((v1&0xF)<<4));
            }
        }
        *out_len=o; return q;
    }
    size_t nb=(n+group-1)/group; uint8_t *q=malloc(nb*2+n+1); size_t o=0;
    for(size_t i=0;i<n;i+=group){
        int g=(i+(size_t)group<=n)?group:(int)(n-i);
        float amax=0; for(int j=0;j<g;j++){ float a=fabsf(P[i+j]); if(a>amax)amax=a; }
        float scale=amax>0?amax/127.0f:1.0f;
        uint16_t sh=f32_to_f16(scale); memcpy(q+o,&sh,2); o+=2;
        for(int j=0;j<g;j++){ int v=(int)lrintf(P[i+j]/scale); if(v>127)v=127; if(v<-127)v=-127;
            q[o++]=(uint8_t)(int8_t)v; }
    }
    *out_len=o; return q;
}
/* Dequantize from reader r into P[n] (called at load; result is plain fp32). */
static int dequantize_params(Rd *r, float *P, size_t n, int qtype, int group){
    if(qtype==0){
        for(size_t i=0;i<n;i++){ uint16_t h; if(!rd(r,&h,2)) return 0; P[i]=f16_to_f32(h); }
        return 1;
    }
    if(qtype==2){  /* q4: fp16 scale + 4-bit symmetric, 2 weights/byte */
        for(size_t i=0;i<n;i+=group){
            int g=(i+(size_t)group<=n)?group:(int)(n-i);
            uint16_t sh; if(!rd(r,&sh,2)) return 0; float scale=f16_to_f32(sh);
            for(int j=0;j<g;j+=2){
                uint8_t b; if(!rd(r,&b,1)) return 0;
                int lo=b&0xF; if(lo>=8)lo-=16; P[i+j]=(float)lo*scale;
                if(j+1<g){ int hi=(b>>4)&0xF; if(hi>=8)hi-=16; P[i+j+1]=(float)hi*scale; }
            }
        }
        return 1;
    }
    for(size_t i=0;i<n;i+=group){
        int g=(i+(size_t)group<=n)?group:(int)(n-i);
        uint16_t sh; if(!rd(r,&sh,2)) return 0; float scale=f16_to_f32(sh);
        for(int j=0;j<g;j++){ int8_t b; if(!rd(r,&b,1)) return 0; P[i+j]=(float)b*scale; }
    }
    return 1;
}

/* Self-contained bundle (TLM4): header(11) + params + embedded tokenizer blob. */
static void model_save_bundle(const char *path, const Cfg *c, const float *P, size_t n,
                              const uint8_t *tok_blob, int tok_len){
    FILE *f=fopen(path,"wb"); if(!f){fprintf(stderr,"cannot write %s\n",path);exit(1);}
    /* TLM5 only when there is a window to record, so full-attention models stay
     * TLM4 and remain readable by older builds. Writing TLM4 unconditionally
     * would silently drop win/full_every on every save. */
    if(c->win>0){
        int hdr[13]={c->V,c->D,c->L,c->H,c->KV,c->F,c->T,c->eot,c->tmpl,c->n_mtp,c->n_exp,
                     c->win,c->full_every};
        fwrite("TLM5",1,4,f); fwrite(hdr,4,13,f);
    } else {
        int hdr[11]={c->V,c->D,c->L,c->H,c->KV,c->F,c->T,c->eot,c->tmpl,c->n_mtp,c->n_exp};
        fwrite("TLM4",1,4,f); fwrite(hdr,4,11,f);
    }
    fwrite(P,sizeof(float),n,f);
    fwrite(&tok_len,4,1,f);
    if(tok_len>0 && tok_blob) fwrite(tok_blob,1,tok_len,f);
    fclose(f);
}
/* load params (+config) only; reads TLM1-4 and TLQ1 (dequantized to fp32). */
static float *model_load(const char *path, Cfg *c){
    size_t len; uint8_t *buf=read_file(path,&len);
    if(!buf){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    Rd r={buf,buf+len}; char m[4]; rd(&r,m,4);
    /* TLQ1 mirrors TLM4 (11 ints), TLQ2 mirrors TLM5 (13, carries win). */
    int qz = !memcmp(m,"TLQ1",4) ? 1 : (!memcmp(m,"TLQ2",4) ? 2 :
             (!memcmp(m,"TLQ3",4) ? 3 : 0));
    if(qz){
        int nh = qz==1 ? 11 : (qz==2 ? 13 : 15), qver = qz==1 ? 4 : (qz==2 ? 5 : 6);
        int hdr[HDR_MAX]; rd(&r,hdr,4*nh); cfg_from_hdr(c,hdr,qver);
        int qtype=0,group=64; rd(&r,&qtype,4); rd(&r,&group,4);
        size_t n=param_count(c); float *P=fz(n);
        int ok = qtype==3 ? q4_selective_decode(&r,P,n,c,group)
                          : dequantize_params(&r,P,n,qtype,group);
        if(!ok){fprintf(stderr,"quant body\n");exit(1);}
        free(buf); return P;
    }
    int ver, nh=hdr_ints(m,&ver);
    if(!nh){fprintf(stderr,"bad model magic\n");exit(1);}
    int hdr[HDR_MAX]; rd(&r,hdr,4*nh); cfg_from_hdr(c,hdr,ver);
    size_t n=param_count(c); float *P=fz(n);
    if(!rd(&r,P,sizeof(float)*n)){fprintf(stderr,"model body\n");exit(1);}
    free(buf); return P;
}
/* Shared experts change what the FFN COMPUTES, not just the layout: slots
 * n_exp..n_exp+n_shared-1 run for every token in addition to the routed one.
 * The layout above accounts for them so param_count and the file size agree,
 * but neither forward path applies them yet -- and the q8 decode path would
 * additionally need its own pre-quantised copies. Loading such a model and
 * quietly skipping those experts would produce plausible-looking garbage, so
 * this refuses instead. QK-norm IS implemented; only n_shared is missing. */
static void check_supported(const Cfg *c, const char *path){
    if(c->n_shared>0){
        fprintf(stderr,
            "%s uses %d shared expert(s), which the CPU build does not implement yet.\n"
            "  The forward pass would silently omit them. Run it with cudalm instead.\n",
            path, c->n_shared);
        exit(1);
    }
}
/* load params AND the embedded tokenizer. Handles TLM2-6 and TLQ1-3. TLM1 -> tk NULL. */
static float *model_load_full(const char *path, Cfg *c, Tok **tk_out){
    size_t len; uint8_t *buf=read_file(path,&len);
    if(!buf){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    Rd r={buf,buf+len}; char m[4]; rd(&r,m,4);
    *tk_out=NULL;
    int qz = !memcmp(m,"TLQ1",4) ? 1 : (!memcmp(m,"TLQ2",4) ? 2 :
             (!memcmp(m,"TLQ3",4) ? 3 : 0));
    if(qz){                                                    /* quantized checkpoint */
        int nh = qz==1 ? 11 : (qz==2 ? 13 : 15), qver = qz==1 ? 4 : (qz==2 ? 5 : 6);
        int hdr[HDR_MAX]; rd(&r,hdr,4*nh); cfg_from_hdr(c,hdr,qver);
        int qtype=0,group=64; rd(&r,&qtype,4); rd(&r,&group,4);
        size_t n=param_count(c); float *P=fz(n);
        int ok = qtype==3 ? q4_selective_decode(&r,P,n,c,group)
                          : dequantize_params(&r,P,n,qtype,group);
        if(!ok){fprintf(stderr,"quant body\n");exit(1);}
        check_supported(c,path);
        int tl=0; if(rd(&r,&tl,4) && tl>0){ *tk_out=tok_parse(r.p,tl,1); }
        free(buf); return P;
    }
    int ver, nh=hdr_ints(m,&ver);
    if(!nh){fprintf(stderr,"bad model magic\n");exit(1);}
    int hdr[HDR_MAX]; rd(&r,hdr,4*nh);
    cfg_from_hdr(c,hdr,ver);
    check_supported(c,path);
    size_t n=param_count(c); float *P=fz(n);
    if(!rd(&r,P,sizeof(float)*n)){fprintf(stderr,"model body\n");exit(1);}
    if(ver>=2){ int tl=0; if(rd(&r,&tl,4) && tl>0){ *tk_out=tok_parse(r.p,tl,1); } }
    free(buf); return P;
}
/* FNV-1a hash for n-gram key, matching cudalm.cu */
static unsigned int ng_hash(int t_prev, int t_cur, int R){
    if(t_prev<0) return (unsigned int)-1;  /* skip if no prior token */
    unsigned int h=2166136261u;
    h=(h^(unsigned)t_prev)*16777619u;
    h=(h^(unsigned)t_cur)*16777619u;
    h^=h>>15;
    return h%R;
}

/* Unpack 4-bit digit with sign-extension */
static int q4_unpack(const signed char *d, unsigned long long i){
    unsigned char b=(unsigned char)d[i>>1];
    int v=(i&1) ? (b>>4) : (b&0xF);
    return v>=8 ? v-16 : v;
}

/* Dequantize one contiguous De-nibble row (base index 0) with a single scale. */
static void ng_render_bytes(const signed char *rowb, float sg, int De, float *row){
    for(int j=0;j<De;j++) row[j] = sg * (float)q4_unpack(rowb, (unsigned long long)j);
}
/* Gather one row from the q4 table (RAM or disk) and dequantize into row[De].
   When ng->d is NULL the De/2 packed bytes are read from ng->fs on demand;
   scratch (>= De/2 bytes) backs the disk read. */
static void ng_q4_render_row(const Ng *ng, int De, unsigned int key, float *row, signed char *scratch){
    if(key==(unsigned int)-1){ memset(row,0,De*sizeof(float)); return; }
    int gi=key/ng->group; if(gi>=ng->ngrp) gi=ng->ngrp-1;
    float sg=ng->s[gi];
    if(ng->d){ ng_render_bytes(ng->d + (unsigned long long)key*(De/2), sg, De, row); return; }
    /* streaming: rows are De nibbles = De/2 bytes, byte-aligned since De is even */
    long long off = ng->doff + (long long)key*(De/2);
    if(_fseeki64(ng->fs, off, SEEK_SET)!=0 || fread(scratch, 1, De/2, ng->fs)!=(size_t)(De/2)){
        memset(row,0,De*sizeof(float)); return;
    }
    ng_render_bytes(scratch, sg, De, row);
}

/* N-gram forward: hash bigram (tok[t-1], tok[t]) -> gather -> project Wng -> add to activation */
static void ngram_forward(const Cfg *c, const Ng *ng, const int *tokens, int m, int start_pos,
                          float *x){
    if(!ng->Wng) return;  /* n-gram not loaded */
    int V=c->V, D=c->D, De=ng->De, R=ng->R;
    float *buf=(float*)malloc(De*sizeof(float));
    signed char *scratch=(signed char*)malloc(De/2);  /* backs disk read when streaming */
    for(int j=0;j<m;j++){
        /* get prior token (at position start_pos+j-1 in the full sequence) */
        int t_prev=-1;
        if(start_pos+j>0){
            if(j>0) t_prev=tokens[j-1];
            /* else: prior token is outside this chunk, would need external state */
        }
        int t_cur=tokens[j]; if(t_cur<0||t_cur>=V) t_cur=-1;
        unsigned int key=(t_cur>=0) ? ng_hash(t_prev,t_cur,R) : (unsigned int)-1;
        ng_q4_render_row(ng, De, key, buf, scratch);
        /* project buf[De] through Wng[De x D] and add to x[D] */
        float *xj=x+(size_t)j*D;
        for(int d=0;d<D;d++){
            float sum=0.0f;
            for(int e=0;e<De;e++) sum += buf[e]*ng->Wng[e*D + d];
            xj[d] += sum;
        }
    }
    free(buf); free(scratch);
}

/* Streaming loader (TINYLM_NGSTREAM): keep Wng+scales in RAM (~4.5MB), leave the
   ~64MB digit block on disk and fetch De/2 bytes per row in ngram_forward. */
static int ngram_load_stream(const char *path, const Cfg *c, Ng *ng){
    FILE *f=fopen(path,"rb");
    if(!f){fprintf(stderr,"[ng] stream: no file %s\n",path); return 0;}
    char m[4];
    if(fread(m,1,4,f)!=4 || memcmp(m,"NGR1",4)){fprintf(stderr,"[ng] stream: bad magic\n"); fclose(f); return 0;}
    int hdr[5];
    if(fread(hdr,4,5,f)!=5){fprintf(stderr,"[ng] stream: hdr\n"); fclose(f); return 0;}
    int R=hdr[0], De=hdr[1], order=hdr[2], q4=hdr[3];
    if(R<=0||R>=(1<<24)||De<=0||De>=1024||order!=2||q4!=1||(De&1)){
        fprintf(stderr,"[ng] stream: bad header R=%d De=%d order=%d q4=%d\n",R,De,order,q4);
        fclose(f); return 0;
    }
    int D=c->D; size_t nw=(size_t)De*D, nt=(size_t)R*De, nd=(nt+1)/2;
    ng->R=R; ng->De=De; ng->order=order; ng->group=128; ng->ngrp=(int)((nt+127)/128);
    size_t ng4=(size_t)ng->ngrp*4;
    ng->Wng=(float*)malloc(nw*sizeof(float));
    ng->s=(float*)malloc(ng4);
    ng->d=NULL;
    if(!ng->Wng||!ng->s){fprintf(stderr,"[ng] stream: oom\n"); goto fail;}
    if(fread(ng->Wng,sizeof(float),nw,f)!=nw){fprintf(stderr,"[ng] stream: Wng read\n"); goto fail;}

    /* file size, and bytes remaining after Wng, to distinguish slim vs full layout */
    long long after_wng=_ftelli64(f);
    if(_fseeki64(f,0,SEEK_END)!=0){fprintf(stderr,"[ng] stream: seek end\n"); goto fail;}
    long long fsz=_ftelli64(f);
    long long left=fsz-after_wng;
    long long slim_left=(long long)nd + (long long)ng4 + 4;
    if(left==slim_left){
        ng->doff=after_wng;                                   /* digits immediately follow Wng */
        if(_fseeki64(f, after_wng+(long long)nd, SEEK_SET)!=0){fprintf(stderr,"[ng] stream: seek scales\n"); goto fail;}
        if(fread(ng->s,1,ng4,f)!=ng4){fprintf(stderr,"[ng] stream: scales\n"); goto fail;}
        if(fread(&ng->sref,4,1,f)!=1){fprintf(stderr,"[ng] stream: sref\n"); goto fail;}
    } else {
        /* full layout: WM/WV bf16 (4*nw) + sref + digits + residual+m8+v8 (3*nt) + scales... */
        ng->doff=after_wng + (long long)(4*nw) + 4;
        long long scales_off=ng->doff + (long long)nd + 3*(long long)nt;
        if(_fseeki64(f, scales_off, SEEK_SET)!=0){fprintf(stderr,"[ng] stream: seek scales(full)\n"); goto fail;}
        if(fread(ng->s,1,ng4,f)!=ng4){fprintf(stderr,"[ng] stream: scales(full)\n"); goto fail;}
        ng->sref=1.0f;
    }
    ng->fs=f;
    fprintf(stderr,"[ng] stream: digits on disk (%.1fMB) at off=%lld, Wng+scales in RAM (%.1fMB), sref=%g\n",
            (double)nd/1e6, ng->doff, (double)(nw*4+ng4)/1e6, ng->sref);
    return 1;
fail: fclose(f); free(ng->Wng); free(ng->s); memset(ng,0,sizeof(*ng)); return 0;
}

/* Load n-gram table from .opt.ng sidecar (bigram embedding + q4 quantization) */
static int ngram_load(const char *base_path, const Cfg *c, Ng *ng){
    fprintf(stderr,"[ng] ngram_load called with base_path=%s\n", base_path); fflush(stderr);
    size_t blen=strlen(base_path);
    if(blen+5>=256) {fprintf(stderr,"[ng] path too long\n"); fflush(stderr); return 0;}
    char path[256]; strcpy(path,base_path); strcpy(path+blen,".opt.ng");
    fprintf(stderr,"[ng] trying to load: %s\n", path); fflush(stderr);

    if(getenv("TINYLM_NGSTREAM")) return ngram_load_stream(path, c, ng);

    size_t len; uint8_t *buf=read_file(path,&len);
    fprintf(stderr,"[ng] read_file returned: buf=%p len=%zu\n", (void*)buf, len); fflush(stderr);
    if(!buf) {fprintf(stderr,"[ng] no file, returning 0\n"); fflush(stderr); return 0;}  /* no n-gram file, silently ok */

    Rd r={buf,buf+len}; char m[4]; rd(&r,m,4);
    if(memcmp(m,"NGR1",4)){fprintf(stderr,"bad ngram magic\n"); fflush(stderr); free(buf); return 0;}

    int hdr[5]; rd(&r,hdr,4*5);  /* R, De, order, q4, iter */
    int R=hdr[0], De=hdr[1], order=hdr[2], q4=hdr[3];
    if(R<=0||R>=(1<<24)||De<=0||De>=1024||order!=2||q4!=1){
        fprintf(stderr,"[ng] bad header: R=%d De=%d order=%d q4=%d\n",R,De,order,q4);
        free(buf); return 0;
    }

    int D=c->D; size_t nw=(size_t)De*D, nt=(size_t)R*De;
    ng->R=R; ng->De=De; ng->order=order; ng->group=128; ng->ngrp=(int)((nt+127)/128);
    ng->Wng=(float*)malloc(nw*sizeof(float));
    ng->d=(signed char*)malloc((nt+1)/2);
    ng->s=(float*)malloc((size_t)ng->ngrp*sizeof(float));

    fprintf(stderr,"[ng] reading: nw=%zu nt=%zu nd=%zu file_len=%zu\n",
            nw*sizeof(float), nt, (nt+1)/2, (size_t)(r.end-r.p)); fflush(stderr);

    if(!rd(&r,ng->Wng,nw*sizeof(float))){
        fprintf(stderr,"[ng] Wng read failed: ptr=%zu end=%zu\n", (size_t)(r.p-buf), (size_t)(r.end-buf));
        goto fail;
    }
    fprintf(stderr,"[ng] Wng read ok, at pos %zu\n", (size_t)(r.p-buf));

    /* Two on-disk layouts are supported, distinguished by bytes remaining after Wng:
       SLIM (inference export, slim_ng.py): digits, scales, sref
       FULL (cudalm .opt.ng optimizer checkpoint):
             WM/WV bf16 (4*nw) + sref + digits + residual(nt) + m8(nt) + v8(nt)
             + scales(ngrp*4) + ms(ngrp*4) + vs(ngrp*4)                                */
    size_t nd=(nt+1)/2, ng4=(size_t)ng->ngrp*4;
    size_t left=r.end-r.p;
    size_t slim_left=nd + ng4 + 4;
    if(left==slim_left){
        fprintf(stderr,"[ng] slim layout (%zu bytes): digits, scales, sref\n", left);
        if(!rd(&r,ng->d,nd)){fprintf(stderr,"[ng] digits read\n"); goto fail;}
        if(!rd(&r,ng->s,ng4)){fprintf(stderr,"[ng] scales read\n"); goto fail;}
        if(!rd(&r,&ng->sref,sizeof(float))){fprintf(stderr,"[ng] sref read\n"); goto fail;}
    } else {
        fprintf(stderr,"[ng] full layout (%zu bytes): skipping optimizer state\n", left);
        if(!rd_skip(&r, 4*nw)){fprintf(stderr,"[ng] WM/WV skip\n"); goto fail;}   /* bf16 WM,WV */
        if(!rd_skip(&r, 4)){fprintf(stderr,"[ng] sref skip\n"); goto fail;}        /* sref */
        if(!rd(&r,ng->d,nd)){fprintf(stderr,"[ng] digits read\n"); goto fail;}
        if(!rd_skip(&r, nt)){fprintf(stderr,"[ng] residual skip\n"); goto fail;}
        if(!rd_skip(&r, nt)){fprintf(stderr,"[ng] m8 skip\n"); goto fail;}
        if(!rd_skip(&r, nt)){fprintf(stderr,"[ng] v8 skip\n"); goto fail;}
        if(!rd(&r,ng->s,ng4)){fprintf(stderr,"[ng] scales read\n"); goto fail;}
        ng->sref = 1.0f;                                                           /* ms,vs unused */
    }
    fprintf(stderr,"[ng] loaded: digits %.1fMB, scales %d groups, sref=%g\n",
            (double)nd/1e6, ng->ngrp, ng->sref);

    free(buf); return 1;
fail: free(buf); free(ng->Wng); free(ng->d); free(ng->s); memset(ng,0,sizeof(*ng)); return 0;
}
/* AdamW state sidecar (<out>.opt): magic, iter, then M[n], V[n] */
static void opt_save(const char *path, int iter, const float *M, const float *Vv, size_t n){
    FILE *f=fopen(path,"wb"); if(!f) return;
    fwrite("TOPT",1,4,f); fwrite(&iter,4,1,f);
    fwrite(M,sizeof(float),n,f); fwrite(Vv,sizeof(float),n,f); fclose(f);
}
static int opt_load(const char *path, float *M, float *Vv, size_t n){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    char m[4]; int iter=0;
    if(fread(m,1,4,f)!=4 || memcmp(m,"TOPT",4)){ fclose(f); return 0; }
    if(fread(&iter,4,1,f)!=1){ fclose(f); return 0; }
    if(fread(M,sizeof(float),n,f)!=n || fread(Vv,sizeof(float),n,f)!=n){ fclose(f); return 0; }
    fclose(f); return iter;
}

/* default peak LR (smaller models tolerate higher LR); falls back by width */
static float default_lr_for(const Cfg *c){
    if(c->D<=64)  return 6e-4f;
    if(c->D<=96)  return 4e-4f;
    if(c->D<=192) return 3e-4f;
    if(c->D<=256) return 2.5e-4f;
    return 2e-4f;
}

/* ----------------------------- train loop ------------------------------- */
/* human-readable duration into a caller buffer (e.g. "45s", "12.3m", "3.7d") */
static char *fmt_dur(double s, char *buf, size_t n){
    if(s<0) s=0;
    if(s<90)          snprintf(buf,n,"%.0fs", s);
    else if(s<5400)   snprintf(buf,n,"%.1fm", s/60.0);
    else if(s<172800) snprintf(buf,n,"%.1fh", s/3600.0);
    else              snprintf(buf,n,"%.1fd", s/86400.0);
    return buf;
}
/* Shared training loop: AdamW + cosine schedule, periodic + final checkpoint.
 * Saves a self-contained TLM3 bundle to `out` and AdamW state to `optpath`. */
static void train_loop(const Cfg *c, Weights *w, Weights *gr,
                       float *P, float *G, float *M, float *Vv, size_t np,
                       const uint16_t *tr, size_t ntr, const uint16_t *vl, size_t nvl,
                       int B, int iters, float lr, int start_iter,
                       const char *out, const char *optpath,
                       const uint8_t *tok_blob, int tok_len){
    Acts *a=acts_new(c,B);
    int N=B*c->T; int *tok=malloc(sizeof(int)*N), *tgt=malloc(sizeof(int)*N);
    int **tgt_mtp = c->n_mtp>0 ? malloc(sizeof(int*)*c->n_mtp) : NULL;
    for(int k=0;k<c->n_mtp;k++) tgt_mtp[k]=malloc(sizeof(int)*N);
    int spe=(int)(ntr/((size_t)B*c->T)); if(spe<1)spe=1;
    double wall0=wtime();
    /* time-based cadences (seconds), so progress shows on a wall-clock rhythm
       no matter how slow a single step is. Tunable via env. */
    double print_sec = env_sec("TINYLM_PRINT_SEC", 15.0);   /* cheap per-step loss/ETA line */
    double val_sec   = env_sec("TINYLM_VAL_SEC", 600.0);    /* expensive val eval (4 batches) */
    double save_sec  = env_sec("TINYLM_SAVE_SEC", 300.0);   /* checkpoint to disk */
    double last_print=wall0, last_val=wall0, last_save=wall0;
    int timeup=0;
    float beta1=0.9f,beta2=0.95f,wd=0.1f, warmup=200.0f;
    { const char *e=getenv("TINYLM_WARMUP"); if(e){ float v=(float)atof(e); if(v>=1) warmup=v; } }
    if(g_opt_lion && !g_qadam){ beta2=0.99f; }        /* Lion's momentum beta */
    if(g_wd_over>=0) wd=g_wd_over;
    float tr_loss=0.0f;
    for(int it=start_iter; it<=iters; it++){
        float cur_lr;
        if(it<warmup) cur_lr=lr*(it+1)/warmup;
        else { float r=(it-warmup)/(float)(iters>warmup?iters-warmup:1); if(r>1)r=1;
               cur_lr=0.1f*lr + 0.5f*(1+cosf(3.14159265f*r))*(lr-0.1f*lr); }
        double wall=wtime();
        if(it==start_iter || it==iters || wall-last_val>=val_sec){   /* val: time-based + endpoints */
            /* 4 batches = 8k tokens is far too small a sample to resolve the
             * ~0.05-nat effects an A/B cares about; measured run-to-run spread
             * at vb=4 was 0.09 nats on identical configs. Raise it for A/Bs. */
            float vloss=0; int vb=4;
            { const char *e=getenv("TINYLM_VALB"); if(e){ int v=atoi(e); if(v>=1) vb=v; } }
            for(int q=0;q<vb;q++){
                for(int i=0;i<B;i++){ size_t st=(xorshift()%(nvl-c->T-1));
                    for(int j=0;j<c->T;j++){ tok[i*c->T+j]=vl[st+j]; tgt[i*c->T+j]=vl[st+j+1]; } }
                vloss+=forward(c,w,a,tok,tgt);
            }
            vloss/=vb; wall=wtime();
            char elb[24];
            printf("iter %5d (ep %.2f) | val %.4f | lr %.2e | %s\n",
                   it, (double)it/spe, vloss, cur_lr, fmt_dur(wall-wall0,elb,sizeof elb));
            if(g_q && g_q->on && it>start_iter) qlm_stats(g_q);
            fflush(stdout); last_val=wall; last_print=wall;
        }
        if(it==iters) break;
        /* Wall-clock cap, so two models of different SIZE can be compared over
         * the same amount of COMPUTE rather than the same iteration count.
         * Mirrors CUDALM_MAX_SECONDS on the GPU backend.
         * `it--; continue;` re-enters this iteration with iters==it, which makes
         * the val block at the top fire once before the loop exits -- so the
         * final val is taken AFTER the cap and its cost is not charged to the
         * training budget. That matters when comparing models of different size,
         * where evaluation cost differs several-fold. */
        if(g_max_sec>0 && !timeup && wtime()-wall0>=g_max_sec){
            printf("[train] TINYLM_MAX_SECONDS reached at iter %d\n", it); fflush(stdout);
            timeup=1; iters=it; it--; continue;
        }
        size_t span=(size_t)c->T + c->n_mtp + 2;
        for(int i=0;i<B;i++){ size_t st=(xorshift()%(ntr-span));
            for(int j=0;j<c->T;j++){ int r=i*c->T+j; tok[r]=tr[st+j]; tgt[r]=tr[st+j+1];
                for(int k=0;k<c->n_mtp;k++) tgt_mtp[k][r]=tr[st+j+2+k]; } }
        tr_loss=forward(c,w,a,tok,tgt);          /* training-batch loss (free; already computed) */
        memset(G,0,sizeof(float)*np);
        backward(c,w,gr,a,tok,tgt,tgt_mtp);
        float gnm=global_gradnorm(G,np);
        float gscale = gnm>1.0f ? 1.0f/gnm : 1.0f;   /* clip folded into adamw */
        if(g_opt_lion) qlm_update(g_q,P,G,M,Vv,np,cur_lr,beta1,beta2,wd,gscale,it+1);
        else           adamw(P,G,M,Vv,np,cur_lr,beta1,beta2,wd,it+1,gscale);
        wall=wtime();
        if(it>start_iter && wall-last_print>=print_sec){   /* cheap progress: loss + tok/s + ETA */
            double done=wall-wall0, sps=done/(it-start_iter), eta=(iters-it)*sps;
            char elb[24], etb[24];
            printf("iter %5d (ep %.2f) | loss %.4f | %.0f tok/s | %.1f%% | %s | ETA %s\n",
                   it, (double)it/spe, tr_loss, (double)N*(it-start_iter)/done, 100.0*it/iters,
                   fmt_dur(done,elb,sizeof elb), fmt_dur(eta,etb,sizeof etb));
            fflush(stdout); last_print=wall;
        }
        if(it>start_iter && wall-last_save>=save_sec){     /* checkpoint: time-based */
            model_save_bundle(out,c,P,np,tok_blob,tok_len);
            opt_save(optpath,it,M,Vv,np);
            if(g_q && g_q->on){ char qp[1200]; snprintf(qp,sizeof qp,"%s.q",optpath); qlm_save(qp,g_q); }
            printf("[ckpt] saved %s @ iter %d (ep %.2f)\n", out, it, (double)it/spe); fflush(stdout);
            last_save=wtime();   /* exclude the disk-write time from the next interval */
        }
    }
    model_save_bundle(out,c,P,np,tok_blob,tok_len);
    opt_save(optpath,iters,M,Vv,np);
    if(g_q && g_q->on){ char qp[1200]; snprintf(qp,sizeof qp,"%s.q",optpath); qlm_save(qp,g_q); }
}

/* read data_dir's tokenizer.bin (blob+V+eot), template, and train/val token streams */
static int load_dataset(const char *ddir, int *V, int *eot, int *tmpl,
                        uint8_t **tok_blob, int *tok_len,
                        uint16_t **tr, size_t *ntr, uint16_t **vl, size_t *nvl){
    char p[1024];
    snprintf(p,sizeof(p),"%s/tokenizer.bin",ddir);
    FILE *e=fopen(p,"rb"); if(!e){ fprintf(stderr,"no tokenizer.bin in %s (run a prepare_* script)\n",ddir); return 0; } fclose(e);
    Tok *dtk=tok_load(p,0); *V=dtk->V; *eot=dtk->eot;
    size_t tlz; *tok_blob=read_file(p,&tlz); *tok_len=*tok_blob?(int)tlz:0;
    snprintf(p,sizeof(p),"%s/template.txt",ddir);
    *tmpl=0; { size_t tl; uint8_t*tb=read_file(p,&tl); if(tb){ *tmpl=(tl>=6 && !memcmp(tb,"alpaca",6))?1:0; free(tb);} }
    snprintf(p,sizeof(p),"%s/train.bin",ddir); *tr=load_u16(p,ntr);
    snprintf(p,sizeof(p),"%s/val.bin",ddir);   *vl=load_u16(p,nvl);
    return 1;
}

/* ----------------------------- train (named) ---------------------------- */
static int cmd_train(int argc, char **argv){
    if(argc<4){ fprintf(stderr,
        "usage: tinylm train <name> <data_dir> [epochs]\n"
        "  <name>     a model from 'tinylm new <name> <preset>'. If <name> is itself\n"
        "             a preset and no such model exists, it is created automatically.\n"
        "  <data_dir> folder with train.bin, val.bin, tokenizer.bin (a prepare_* run)\n"
        "  epochs     passes over the data (default 1; fractional ok). Total target -\n"
        "             re-running continues toward it (auto-resume). Ctrl-C is safe.\n"
        "  e.g.  tinylm new mychat m3   then   tinylm train mychat ../data/alpaca 2\n"); return 1; }
    const char *name=argv[2], *ddir=argv[3];
    float epochs = argc>4 ? atof(argv[4]) : 1.0f;
    int B=32;
    char mpath[1024], optpath[1100];
    model_path(name, mpath, sizeof(mpath));
    snprintf(optpath,sizeof(optpath),"%s.opt",mpath);

    int dataV,dataEot,tmpl,tok_len; uint8_t *tok_blob; uint16_t *tr,*vl; size_t ntr,nvl;
    if(!load_dataset(ddir,&dataV,&dataEot,&tmpl,&tok_blob,&tok_len,&tr,&ntr,&vl,&nvl)) return 1;

    Cfg c; int exists=0, trained=0; float *loaded=NULL;
    { FILE*e=fopen(mpath,"rb"); if(e){ exists=1; fclose(e);} }
    if(exists){ Tok *mtk=NULL; loaded=model_load_full(mpath,&c,&mtk); trained=(mtk!=NULL); }
    else {
        if(!preset(name,&c)){ fprintf(stderr,
            "model 'models/%s.bin' not found and '%s' is not a preset.\n"
            "  create it:  tinylm new %s <preset>     (presets: nano micro m1 m3 m6 m8)\n", name,name,name);
            return 1; }
        c.n_mtp=default_nmtp();
    }
    int fresh = !exists || !trained;
    if(!fresh && c.V!=dataV){
        fprintf(stderr,"error: model vocab %d != data vocab %d (different tokenizer).\n"
                       "Use a fresh model for this data:  tinylm new <name> <preset>\n", c.V, dataV);
        return 1; }
    if(c.qknorm || c.n_shared>0){
        fprintf(stderr,"this checkpoint uses %s%s%s, which the CPU backward does not\n"
                       "  implement -- training it here would produce wrong gradients for\n"
                       "  those parameters. Train it with cudalm; inference here is fine.\n",
                c.qknorm?"QK-norm":"", (c.qknorm&&c.n_shared>0)?" and ":"",
                c.n_shared>0?"shared experts":"");
        exit(1);
    }
    c.V=dataV; c.eot=dataEot; c.tmpl=tmpl; cfg_derive(&c);
    B=auto_batch(&c);          /* memory-aware: 32 for small models, less for big */

    float lr=default_lr_for(&c);
    /* QLM_OPT: adamw (stock, default) | lion (fp32 Lion, isolates the optimizer)
     *        | qlion (Integer Lion over low-bit digit weights -- the experiment) */
    const char *qopt=getenv("QLM_OPT"); if(!qopt) qopt="adamw";
    int use_qadam = !strcmp(qopt,"qadam");
    int use_qlion = !strcmp(qopt,"qlion") || use_qadam;
    { const char *e=getenv("QLM_ACT8"); g_act8 = e ? atoi(e) : 0; }
    if(g_act8) printf("[qlm] int8 activations (fake-quant, STE) ON\n");
    g_opt_lion = use_qlion || !strcmp(qopt,"lion");
    /* LR scale and weight decay apply to EVERY optimizer, so the baseline can be
     * tuned on the same grid as the experiment -- comparing a swept Lion against
     * an unswept AdamW would be worthless. */
    { const char *e;
      if((e=getenv("QLM_LRSCALE"))) lr *= (float)atof(e);
      else if(g_opt_lion && !use_qadam) lr *= 0.1f;
      if((e=getenv("QLM_WD")))      g_wd_over = (float)atof(e);
      else if(g_opt_lion && !use_qadam) g_wd_over = 1.0f; }
    g_qadam = use_qadam;
    g_max_sec = env_sec("TINYLM_MAX_SECONDS", 0.0);
    rope_init(&c, c.T);
    size_t np=param_count(&c);
    float *P=fz(np),*G=fz(np),*M=fz(np),*Vv=fz(np);
    Weights w,gr; map_weights(&c,P,&w); map_weights(&c,G,&gr); memset(&w.ng,0,sizeof(w.ng)); memset(&gr.ng,0,sizeof(gr.ng));
    int start_iter=0;
    if(fresh){ init_params(&c,&w); if(loaded) free(loaded); }
    else { memcpy(P,loaded,sizeof(float)*np); free(loaded); start_iter=opt_load(optpath,M,Vv,np); }
    if(use_qlion){
        qlm_init(&g_qlm,&c,&w,P,np); g_q=&g_qlm;   /* reads QLM_OPT itself */
        char qp[1200]; snprintf(qp,sizeof qp,"%s.q",optpath);
        if(!fresh && qlm_load(qp,g_q)) printf("[qlm] resumed integer state from %s\n",qp);
        qlm_render(g_q,P);            /* P is a view of the digits from here on */
        qlm_report(g_q,np);
    }

    int spe=(int)(ntr/((size_t)B*c.T)); if(spe<1)spe=1;
    int iters=(int)(epochs*spe + 0.5f); if(iters<1)iters=1;
    if(iters<start_iter) iters=start_iter;
    int nthreads=g_pool_nt;
    #ifdef _OPENMP
    nthreads=g_pool_nt>1?g_pool_nt:omp_get_max_threads();
    #endif
    printf("[train] optimizer=%s\n", qopt);
    printf("[train] model=%s %s params=%zu threads=%d n_mtp=%d batch=%d\n",
           name, fresh?"(fresh)":"(resume)", np, nthreads, c.n_mtp, B);
    printf("[train] data=%s V=%d tok=%zu | epochs=%.2f -> %d iters (steps/epoch=%d) start=%d lr=%g\n",
           ddir, c.V, ntr, epochs, iters, spe, start_iter, lr);
    printf("[train] cadence: progress %.0fs | val %.0fs | checkpoint %.0fs (TINYLM_PRINT_SEC/VAL_SEC/SAVE_SEC)\n",
           env_sec("TINYLM_PRINT_SEC",15.0), env_sec("TINYLM_VAL_SEC",600.0), env_sec("TINYLM_SAVE_SEC",300.0));
    fflush(stdout);

    train_loop(&c,&w,&gr,P,G,M,Vv,np, tr,ntr,vl,nvl, B, iters, lr, start_iter,
               mpath, optpath, tok_blob, tok_len);
    printf("[train] '%s' now at %.2f epochs -> %s\n", name, (double)iters/spe, mpath);
    printf("[next]  tinylm run %s -i\n", name);
    return 0;
}

/* ----------------------------- new -------------------------------------- */
static int cmd_new(int argc, char **argv){
    if(argc<4){ fprintf(stderr,
        "usage: tinylm new <name> <preset> [n_experts]\n"
        "  presets: nano (~0.2M) micro (~0.6M) m1 (~0.9M) m3 (~3M) m6 (~6M) m8 (~8M)\n"
        "  n_experts: optional MoE - top-1 routed experts (>=2). More capacity at\n"
        "             ~the same compute (FFN is replaced by E experts; 1 runs/token).\n"
        "  creates models/<name>.bin (untrained). Then: tinylm train <name> <data> <epochs>\n");
        return 1; }
    const char *name=argv[2], *ps=argv[3];
    Cfg c; if(!preset(ps,&c)){ fprintf(stderr,"unknown preset '%s' (nano micro m1 m3 m6 m8)\n",ps); return 1; }
    c.n_mtp=default_nmtp(); c.n_exp = argc>4 ? atoi(argv[4]) : 0; if(c.n_exp==1) c.n_exp=0; cfg_derive(&c);
    char path[1024]; model_path(name, path, sizeof(path));
    { FILE*e=fopen(path,"rb"); if(e){ fclose(e); fprintf(stderr,"model '%s' already exists (%s)\n",name,path); return 1; } }
    rope_init(&c, c.T);
    size_t np=param_count(&c); float *P=fz(np);
    Weights w; map_weights(&c,P,&w); init_params(&c,&w);
    model_save_bundle(path,&c,P,np,NULL,0);     /* untrained: no tokenizer yet */
    printf("[new] created %s  preset=%s params=%zu (D=%d L=%d H=%d kv=%d F=%d n_mtp=%d n_exp=%d)\n",
           path, ps, np, c.D,c.L,c.H,c.KV,c.F,c.n_mtp,c.n_exp);
    printf("[next] tinylm train %s <data_dir> <epochs>\n", name);
    return 0;
}

/* ----------------------------- int8 GEMV (decode) ----------------------- *
 * llama.cpp/ggml-style int8 matrix-vector for the M=1 decode hot path: weights
 * and activation are quantized to Q8 (blocks of 32), dot via AVX2 maddubs + the
 * sign trick for signed int8. Weights are pre-quantized once at gen-load into an
 * output-major [N][K] layout (emb is already [V][D]; the [K][N] linear weights
 * are transposed). Memory-bound GEMV, so ~3-4x the fp32 OpenBLAS path.          */
static inline int dot_i8_32(const int8_t *a, const int8_t *b){
    __m256i va=_mm256_loadu_si256((const __m256i*)a), vb=_mm256_loadu_si256((const __m256i*)b);
    __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(va,va), _mm256_sign_epi8(vb,va));
    __m256i s=_mm256_madd_epi16(p,_mm256_set1_epi16(1));
    __m128i t=_mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s,1));
    t=_mm_hadd_epi32(t,t); t=_mm_hadd_epi32(t,t); return _mm_cvtsi128_si32(t);
}
/* One full int8 row dot: sum_b scale_b * <w_b, x_b> over nb blocks of 32.
 *
 * The old form called dot_i8_32 per block, and that helper ends in a full
 * horizontal reduction (two hadds + extract) to hand back a scalar. At K=512
 * that is 16 reductions, ~160 cycles, against ~16 cycles of actual multiply-add
 * -- the reduction cost an order of magnitude more than the arithmetic.
 *
 * Here the per-block int32 lane sums are NOT reduced. They are converted to
 * float, scaled by the block's scale, and accumulated into a vector; a single
 * horizontal reduction happens once per row. Same arithmetic, 1/16th the
 * reduction. VPDPBUSD (AVX512-VNNI) folds each block into one instruction where
 * available; the AVX2 maddubs/madd pair is the fallback and is still correct.
 *
 * dpbusd is unsigned x signed, so the operands are folded through sign_epi8:
 * |w| as the unsigned side and x carrying w's sign, which preserves the product.
 *
 * NOT bit-exact with the old scalar accumulation -- the lane order differs. */
static int exact_gemv_on(void){
    static int v=-1;
    if(v<0){ const char *e=getenv("TINYLM_EXACT_GEMV"); v = e && *e!='0'; }
    return v;
}
/* Phase 2B: true 4-bit weight path (env TINYLM_Q4). Weights are stored packed
 * 2/byte, 16 bytes per 32-element K-block, one fp32 scale per block -- the same
 * block layout as int8, half the weight bytes. Activations stay int8, so the
 * dot unpacks the nibbles to signed int8 in-register and reuses the identical
 * maddubs/dpbusd accumulation as qdot_row. int8 remains the default; this is a
 * quality/RAM A/B. Byte packing: byte t = (w[2t]&0xF) | ((w[2t+1]&0xF)<<4),
 * symmetric [-7,7], sign via two's complement (n>=8 ? n-16 : n). */
int g_q4=0;
static void q4_row(const float *x, int K, uint8_t *q, float *s){
    int nb=(K+31)/32;
    for(int b=0;b<nb;b++){
        int off=b*32, g=(off+32<=K)?32:(K-off); float amax=0;
        for(int j=0;j<g;j++){ float a=fabsf(x[off+j]); if(a>amax)amax=a; }
        float scale=amax>0?amax/7.0f:1.0f, inv=amax>0?7.0f/amax:0.0f; s[b]=scale;
        int8_t t[32];
        for(int j=0;j<g;j++){ int v=(int)lrintf(x[off+j]*inv); if(v>7)v=7; if(v<-7)v=-7; t[j]=(int8_t)v; }
        for(int j=g;j<32;j++) t[j]=0;
        uint8_t *o=q+(size_t)b*16;
        for(int p=0;p<16;p++) o[p]=(uint8_t)((t[2*p]&0x0F)|((t[2*p+1]&0x0F)<<4));
    }
}
static inline float q4dot_row(const uint8_t *wr, const float *ws,
                              const int8_t *xq, const float *xs, int nb){
    if(exact_gemv_on()){
        float a=0.0f;
        for(int b=0;b<nb;b++){
            const uint8_t *o=wr+(size_t)b*16; int8_t t[32];
            for(int p=0;p<16;p++){ int lo=o[p]&0xF, hi=o[p]>>4;
                if(lo>=8)lo-=16; if(hi>=8)hi-=16; t[2*p]=(int8_t)lo; t[2*p+1]=(int8_t)hi; }
            a += ws[b]*xs[b]*(float)dot_i8_32(t, xq+(size_t)b*32);
        }
        return a;
    }
    const __m128i m0f=_mm_set1_epi8(0x0F), eight=_mm_set1_epi8(8);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<nb;b++){
        __m128i v=_mm_loadu_si128((const __m128i*)(wr+(size_t)b*16));
        __m128i lo=_mm_and_si128(v,m0f);
        __m128i hi=_mm_and_si128(_mm_srli_epi16(v,4),m0f);
        lo=_mm_sub_epi8(_mm_xor_si128(lo,eight),eight);   /* (n^8)-8 == n>=8?n-16:n */
        hi=_mm_sub_epi8(_mm_xor_si128(hi,eight),eight);
        __m128i b0=_mm_unpacklo_epi8(lo,hi), b1=_mm_unpackhi_epi8(lo,hi);
        __m256i va=_mm256_set_m128i(b1,b0);               /* lanes w0..w31 in order */
        __m256i vb=_mm256_loadu_si256((const __m256i*)(xq+(size_t)b*32));
        __m256i ua=_mm256_sign_epi8(va,va), sb=_mm256_sign_epi8(vb,va);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        __m256i s=_mm256_dpbusd_epi32(_mm256_setzero_si256(), ua, sb);
#else
        __m256i s=_mm256_madd_epi16(_mm256_maddubs_epi16(ua,sb), _mm256_set1_epi16(1));
#endif
        acc=_mm256_fmadd_ps(_mm256_set1_ps(ws[b]*xs[b]), _mm256_cvtepi32_ps(s), acc);
    }
    __m128 r=_mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc,1));
    r=_mm_hadd_ps(r,r); r=_mm_hadd_ps(r,r);
    return _mm_cvtss_f32(r);
}
static inline float qdot_row(const int8_t *wr, const float *ws,
                             const int8_t *xq, const float *xs, int nb){
    if(g_q4) return q4dot_row((const uint8_t*)wr, ws, xq, xs, nb);
    if(exact_gemv_on()){        /* original per-block scalar order, for bit-exact replay */
        float a=0.0f;
        for(int b=0;b<nb;b++)
            a += ws[b]*xs[b]*(float)dot_i8_32(wr+(size_t)b*32, xq+(size_t)b*32);
        return a;
    }
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<nb;b++){
        __m256i va=_mm256_loadu_si256((const __m256i*)(wr+(size_t)b*32));
        __m256i vb=_mm256_loadu_si256((const __m256i*)(xq+(size_t)b*32));
        __m256i ua=_mm256_sign_epi8(va,va), sb=_mm256_sign_epi8(vb,va);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        __m256i s=_mm256_dpbusd_epi32(_mm256_setzero_si256(), ua, sb);
#else
        __m256i s=_mm256_madd_epi16(_mm256_maddubs_epi16(ua,sb), _mm256_set1_epi16(1));
#endif
        acc=_mm256_fmadd_ps(_mm256_set1_ps(ws[b]*xs[b]), _mm256_cvtepi32_ps(s), acc);
    }
    __m128 lo=_mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc,1));
    lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo);
    return _mm_cvtss_f32(lo);
}
/* quantize x[K] to Q8 blocks of 32 (last block zero-padded to 32) */
static void q8_row(const float *x, int K, int8_t *q, float *s){
    int nb=(K+31)/32;
    for(int b=0;b<nb;b++){
        int off=b*32, g=(off+32<=K)?32:(K-off); float amax=0;
        for(int j=0;j<g;j++){ float a=fabsf(x[off+j]); if(a>amax)amax=a; }
        float scale=amax>0?amax/127.0f:1.0f, inv=amax>0?127.0f/amax:0.0f; s[b]=scale;
        for(int j=0;j<g;j++){ int v=(int)lrintf(x[off+j]*inv); if(v>127)v=127; if(v<-127)v=-127; q[off+j]=(int8_t)v; }
        for(int j=g;j<32;j++) q[off+j]=0;
    }
}
typedef struct { int8_t *q; float *s; } QMat;    /* q:[N][rowbytes]  s:[N][nb] */
/* Weight row stride: int8 keeps 32 bytes/block, q4 packs 2/byte -> 16. The .q
 * buffer holds int8 or packed-nibbles depending on g_q4; quant_row picks. */
static inline int rowbytes(int nb){ return g_q4 ? nb*16 : nb*32; }
static inline void quant_row(const float *x,int K,int8_t *q,float *s){
    if(g_q4) q4_row(x,K,(uint8_t*)q,s); else q8_row(x,K,q,s);
}
static QMat qmat_KN(const float *W, int K, int N){   /* W[K][N] (mm reduces K) -> transpose+quant */
    int nb=(K+31)/32, kp=rowbytes(nb); QMat m;
    m.q=malloc((size_t)N*kp); m.s=malloc(sizeof(float)*(size_t)N*nb);
    float *col=malloc(sizeof(float)*K);
    for(int n=0;n<N;n++){ for(int k=0;k<K;k++) col[k]=W[(size_t)k*N+n];
        quant_row(col,K,m.q+(size_t)n*kp,m.s+(size_t)n*nb); }
    free(col); return m;
}
/* Q, K and V are three separate [K][*] matrices that all consume the SAME normed
 * activation row, so they are concatenated output-major into one QMat: one
 * q8_row of x instead of three, one pool dispatch instead of three, and one
 * contiguous weight stream instead of three. */
static QMat qmat_KN3(const float *Wq, const float *Wk, const float *Wv,
                     int K, int Nq, int Nk, int Nv){
    int nb=(K+31)/32, kp=rowbytes(nb), N=Nq+Nk+Nv; QMat m;
    m.q=malloc((size_t)N*kp); m.s=malloc(sizeof(float)*(size_t)N*nb);
    float *col=malloc(sizeof(float)*K);
    const float *src[3]={Wq,Wk,Wv}; int cnt[3]={Nq,Nk,Nv}; int o=0;
    for(int t=0;t<3;t++)
        for(int n=0;n<cnt[t];n++,o++){
            for(int k=0;k<K;k++) col[k]=src[t][(size_t)k*cnt[t]+n];
            quant_row(col,K,m.q+(size_t)o*kp,m.s+(size_t)o*nb);
        }
    free(col); return m;
}
/* W1 and W3 interleaved: row 2n is W1's column n, row 2n+1 is W3's. Both feed
 * the same SwiGLU output element, so this turns two streams 362 KB apart (for
 * the 220M's experts) into one contiguous walk. */
static QMat qmat_KN_ilv(const float *Wa, const float *Wb, int K, int N){
    int nb=(K+31)/32, kp=rowbytes(nb); QMat m;
    m.q=malloc((size_t)2*N*kp); m.s=malloc(sizeof(float)*(size_t)2*N*nb);
    float *col=malloc(sizeof(float)*K);
    for(int n=0;n<N;n++){
        for(int k=0;k<K;k++) col[k]=Wa[(size_t)k*N+n];
        quant_row(col,K,m.q+(size_t)(2*n)*kp,   m.s+(size_t)(2*n)*nb);
        for(int k=0;k<K;k++) col[k]=Wb[(size_t)k*N+n];
        quant_row(col,K,m.q+(size_t)(2*n+1)*kp, m.s+(size_t)(2*n+1)*nb);
    }
    free(col); return m;
}
static QMat qmat_NK(const float *W, int N, int K){   /* W[N][K] (emb) -> quant as-is */
    int nb=(K+31)/32, kp=rowbytes(nb); QMat m;
    m.q=malloc((size_t)N*kp); m.s=malloc(sizeof(float)*(size_t)N*nb);
    for(int n=0;n<N;n++) quant_row(W+(size_t)n*K,K,m.q+(size_t)n*kp,m.s+(size_t)n*nb);
    return m;
}
static int8_t *gq_xq=NULL; static float *gq_xs=NULL; static int gq_cap=0;
/* Fanned out over the decode pool, never OpenMP: at m=1 a whole GEMV is a few
 * hundred KB, which is below libgomp's fork/join cost (see the pool comment). */
typedef struct { const QMat *m; const int8_t *xq; const float *xs; float *Y; int nb,kp; } GemvJob;
static void gemv_body(void *p,int lo,int hi,int tid){
    GemvJob *j=(GemvJob*)p; (void)tid;
    for(int n=lo;n<hi;n++)
        j->Y[n]=qdot_row(j->m->q+(size_t)n*j->kp, j->m->s+(size_t)n*j->nb, j->xq, j->xs, j->nb);
}
/* Y[N] = X[K] . W  (W pre-quantized, output-major); quantizes X on the fly */
static void gemv_q8(const QMat *m, const float *X, float *Y, int N, int K){
    int nb=(K+31)/32, wkp=rowbytes(nb), xkp=nb*32;   /* activations always int8 */
    if(xkp>gq_cap){ gq_xq=realloc(gq_xq,xkp); gq_xs=realloc(gq_xs,sizeof(float)*(size_t)nb); gq_cap=xkp; }
    q8_row(X,K,gq_xq,gq_xs);
    GemvJob j={m,gq_xq,gq_xs,Y,nb,wkp};              /* kp = weight row stride */
    /* Dispatch is ~1 us; below ~32 KB of weights the split does not pay it back. */
    if((long)N*K < (32L<<10)) gemv_body(&j,0,N,0);
    else                      tl_for(N, gemv_body, &j);
}
/* Y[m][N] = X[m][K] . W, W pre-quantized output-major. The m>1 companion to
 * gemv_q8.
 *
 * Loop order matters more than the arithmetic here. Calling gemv_q8 once per
 * row would stream the entire weight matrix m times; for one of the 220M's
 * expert layers that is 200 MB per pass. Instead each weight row is loaded once
 * and reused across all m activation rows, so the weights are read exactly
 * once and the m activations stay in L1.
 *
 * Activations are quantized ONCE up front rather than per row per column. */
typedef struct { const QMat *mq; const int8_t *xq; const float *xs; float *Y; int m,N,nb,wkp,xkp; } GemmJob;
static void gemm_body(void *p,int lo,int hi,int tid){
    GemmJob *j=(GemmJob*)p; (void)tid;
    for(int n=lo;n<hi;n++){
        const int8_t *wr=j->mq->q+(size_t)n*j->wkp; const float *ws=j->mq->s+(size_t)n*j->nb;
        for(int i=0;i<j->m;i++)
            j->Y[(size_t)i*j->N+n]=qdot_row(wr, ws, j->xq+(size_t)i*j->xkp, j->xs+(size_t)i*j->nb, j->nb);
    }
}
static void gemm_q8(const QMat *mq,const float *X,float *Y,int m,int N,int K){
    int nb=(K+31)/32, wkp=rowbytes(nb), xkp=nb*32;
    int8_t *xq=malloc((size_t)m*xkp);
    float  *xs=malloc(sizeof(float)*(size_t)m*nb);
    for(int i=0;i<m;i++) q8_row(X+(size_t)i*K,K,xq+(size_t)i*xkp,xs+(size_t)i*nb);
    GemmJob j={mq,xq,xs,Y,m,N,nb,wkp,xkp};
    tl_for(N, gemm_body, &j);
    free(xq); free(xs);
}
/* decode matmul: int8 whenever a quantized copy exists. Previously this was
 * m==1 only, so prefill, scoring and speculative chunks all silently ran fp32
 * -- i.e. the int8 path never engaged for anything except single-token decode. */
static void mm_dec(float *Y,const float *X,const float *Wf,const QMat *qm,int m,int K,int N){
    if(qm && qm->q){ if(m==1) gemv_q8(qm,X,Y,N,K); else gemm_q8(qm,X,Y,m,N,K); }
    else mm(Y,X,Wf,m,K,N);
}
static void mmbt_dec(float *Y,const float *X,const float *Wf,const QMat *qm,int m,int V,int D){
    if(qm && qm->q){ if(m==1) gemv_q8(qm,X,Y,V,D); else gemm_q8(qm,X,Y,m,V,D); }
    else { memset(Y,0,sizeof(float)*(size_t)m*V); mm_bt(Y,X,Wf,m,V,D); }
}

/* ----------------------------- KV-cache decode -------------------------- */
/* Chunked sliding-KV forward. Buffers hold up to CH=ctx rows so we can process
 * a prompt prefill or a speculative-draft chunk in one batched (parallel) pass. */
#define KV_SPLIT_MAX 16           /* cap on split-K chunks per (row, kv-head) */
typedef struct {
    void  **Kc, **Vc;                 /* per-layer key/value cache [T*KD], dtype kvdt */
    float **Ksc, **Vsc;               /* per-layer q8 scales [T*KV] (NULL otherwise) */
    int     kvdt;
    float  *accs;                     /* per-thread head accumulators [nt*H*hd] */
    float  *part;                     /* split-K partials [KV_SPLIT_MAX*H*(hd+2)] */
    float *x,*xn,*q,*k,*v,*ctx,*tmp,*g,*u;   /* chunk activations (CH rows) */
    float *fn,*logits,*rinv,*probs;
    float *hlog,*hh,*ht;              /* single-row scratch (main/MTP heads) */
    int   *posbuf;
    int   *moe_as; float *moe_gt,*moe_rp;    /* MoE routing scratch, allocated once:
                                                forward_chunk used to malloc/free
                                                three of these per layer per token */
    int    q8;                        /* int8 decode active */
    float *qkv;                       /* fused Q|K|V output rows, [CH][D+2*KD] */
    /* Pre-quantized decode weights. qwqkv is Q, K and V concatenated
     * output-major, and qw13 interleaves W1 and W3 row-by-row -- see
     * qmat_KN3 / qmat_KN_ilv. */
    QMat   qemb, *qwqkv,*qwo,*qw13,*qw2,*qmtp;
} Gen;

/* TINYLM_PROF=1 attributes decode time to attention / matmul / MoE / lm_head.
 * Guarded by a flag rather than compiled out: the timer calls are a handful of
 * ns against ms-scale buckets, and being able to ask a shipped binary where the
 * time went beats rebuilding to find out. */
static int max_threads(void){
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static int this_thread(void){
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
/* hd-length dot / axpy. Kept as plain loops so -O3 -march=native -ffast-math
 * vectorises them; the win here is not hand-written intrinsics but that the
 * CALLER now hands them contiguous runs instead of modulo-strided addresses. */
static inline float dot_f(const float *restrict a, const float *restrict b, int n){
    float s=0.0f; for(int i=0;i<n;i++) s+=a[i]*b[i]; return s;
}
static inline void axpy_f(float *restrict y, const float *restrict x, float a, int n){
    for(int i=0;i<n;i++) y[i]+=a*x[i];
}
/* These stay simple row-at-a-time loops on purpose. A 4-row-blocked variant
 * (four accumulator chains, one horizontal reduction per four rows, accumulator
 * stored once per four rows) was written and measured: attention 1.239s ->
 * 1.213s, ~2% and inside run-to-run noise. -O3 -march=native already vectorises
 * them, and the limit is streaming K/V out of L2/L3, not reduction latency. The
 * blocking DID reassociate the sums, and because q8_row's quantisation is a step
 * function that fp noise amplified into a 1.2e-2 shift in summed logprobs and a
 * different continuation on one of five prompts. Not worth 2% inside noise. */
/* ------------------------- KV cache element type -------------------------
 * Attention is the largest single cost at real context lengths and it is
 * limited by streaming K/V, not by arithmetic -- so shrinking the cache buys
 * speed as well as memory. Three widths, chosen with TINYLM_KV:
 *   f32 (default)  4 B/elem   bit-exact with every historical number
 *   f16            2 B/elem   ~1e-3 relative on the stored value
 *   q8             1 B/elem   per-(position,kv-head) scale, ~1e-2
 * The query side stays fp32 in all three: only the CACHE is narrowed, so the
 * dot products still accumulate in fp32 and the error is purely the stored
 * K/V rounding rather than a quantised matmul. */
/* Split-K decode: fusing 8 heads into KV=2 groups leaves only 2 jobs, so the
 * position range is split into chunks and the partial softmaxes merged, keeping
 * every core busy. It works, and it is OFF by default, because measured on this
 * box it bought nothing (22-23 tok/s at ~700 ctx either way) while its merge
 * reassociates the softmax and so gives up bit-exact decode. Enable with
 * TINYLM_KVSPLIT=1 on a machine with more cores than this one, where 2 jobs
 * would actually leave cores idle. */
/* Splitting the key range is what gives the m==1 attention more than KV=2 jobs
 * to hand out. It used to default OFF because under OpenMP the extra regions
 * cost more than the split saved; with the decode pool it is the difference
 * between attention running on one core and on all of them, so it now defaults
 * ON whenever the pool is live. TINYLM_KVSPLIT=0 forces the unsplit path, which
 * is the bit-exact one (the online merge reassociates the softmax). */
static int kv_split_on(void){
    static int v=-1;
    if(v<0){ const char *e=getenv("TINYLM_KVSPLIT"); v = e ? (*e!='0') : (g_pool_nt>1); }
    return v;
}
#define KVDT_F32 0
#define KVDT_F16 1
#define KVDT_Q8  2
static int kv_elem_sz(int dt){ return dt==KVDT_F32?4 : dt==KVDT_F16?2 : 1; }
static const char *kv_name(int dt){ return dt==KVDT_F32?"f32" : dt==KVDT_F16?"f16" : "q8"; }
/* Default f16. Measured on 60 HellaSwag sequences (1,076 continuation tokens)
 * against f32: max 1.15e-02 relative on summed logprobs and 4 greedy-token
 * disagreements, i.e. 0.4% -- for half the cache. It does mean decode is no
 * longer bit-exact with pre-f16 numbers; TINYLM_KV=f32 restores that exactly,
 * and is what to use when reproducing an older benchmark figure. */
static int kv_dtype_env(void){
    const char *e=getenv("TINYLM_KV");
    if(!e||!*e) return KVDT_F16;
    if(!strcmp(e,"f16")||!strcmp(e,"fp16")) return KVDT_F16;
    if(!strcmp(e,"q8") ||!strcmp(e,"int8")) return KVDT_Q8;
    if(!strcmp(e,"f32")||!strcmp(e,"fp32")) return KVDT_F32;
    fprintf(stderr,"[tinylm] unknown TINYLM_KV=%s (want f32|f16|q8); using f32\n",e);
    return KVDT_F32;
}
static inline float h2f(unsigned short h){
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    unsigned int s=(h>>15)&1u, e=(h>>10)&0x1fu, m=h&0x3ffu, b;
    if(e==0){ if(!m){ b=s<<31; } else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; b=(s<<31)|(e<<23)|(m<<13); } }
    else if(e==31) b=(s<<31)|0x7f800000u|(m<<13);
    else b=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&b,4); return f;
#endif
}
static inline unsigned short f2h(float f){
#if defined(__F16C__)
    return _cvtss_sh(f,0);
#else
    unsigned int b; memcpy(&b,&f,4);
    unsigned int s=(b>>31)&1u; int e=(int)((b>>23)&0xff)-127+15; unsigned int m=b&0x7fffffu;
    if(e<=0) return (unsigned short)(s<<15);
    if(e>=31) return (unsigned short)((s<<15)|0x7c00u);
    return (unsigned short)((s<<15)|((unsigned)e<<10)|(m>>13));
#endif
}
/* exp for the attention softmax only.
 *
 * MinGW's expf measured 553 ns/call inside this loop -- 75% of all attention
 * time, and 8.7x its own cost in an isolated microbenchmark. It is not
 * denormals (flush-to-zero is set and changed nothing); the library call simply
 * does not inline or vectorise here. This is the standard 2^k * poly(f) form:
 * range-reduce by log2(e), split into integer and fractional parts, build 2^k by
 * writing the exponent field directly, and evaluate 2^f with a degree-4
 * polynomial on [0,1). Branch-free so the loop still vectorises.
 *
 * Accuracy is ~1e-7 relative, far below the fp32 noise the softmax already
 * carries; TINYLM_EXACT_EXP=1 restores libm expf for comparison. */
static inline float fast_expf(float x){
    x = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    float t = x * 1.44269504088896f;          /* log2(e) */
    float ff = floorf(t);
    float f = t - ff;
    float p = 1.0f + f*(0.6931471805f + f*(0.2402265069f
              + f*(0.0555041087f + f*(0.0096181291f + f*0.0013333558f))));
    union { float f; unsigned int u; } bits;
    bits.u = (unsigned int)(((int)ff + 127) << 23);
    return p * bits.f;
}
static int exact_exp_on(void){
    static int v=-1;
    if(v<0){ const char *e=getenv("TINYLM_EXACT_EXP"); v = e && *e!='0'; }
    return v;
}
/* dot(q[n], row) with the row stored as dt; rs is the row's q8 scale (else 1) */
static inline float kv_dot(const float *restrict q, const void *row, int n, int dt, float rs){
    if(dt==KVDT_F32) return dot_f(q,(const float*)row,n);
    if(dt==KVDT_F16){ const unsigned short *h=(const unsigned short*)row;
        float s=0.0f;
#if defined(__F16C__)
        int i=0; __m256 acc=_mm256_setzero_ps();
        for(; i+8<=n; i+=8)
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(q+i),
                                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(h+i))), acc);
        __m128 lo=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1));
        lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo); s=_mm_cvtss_f32(lo);
        for(; i<n; i++) s+=q[i]*h2f(h[i]);
#else
        for(int i=0;i<n;i++) s+=q[i]*h2f(h[i]);
#endif
        return s; }
    { const signed char *b=(const signed char*)row; float s=0.0f;
      for(int i=0;i<n;i++) s+=q[i]*(float)b[i];
      return s*rs; }
}
/* y[n] += a * row */
static inline void kv_axpy(float *restrict y, const void *row, float a, int n, int dt, float rs){
    if(dt==KVDT_F32){ axpy_f(y,(const float*)row,a,n); return; }
    if(dt==KVDT_F16){ const unsigned short *h=(const unsigned short*)row;
#if defined(__F16C__)
        int i=0; __m256 va=_mm256_set1_ps(a);
        for(; i+8<=n; i+=8)
            _mm256_storeu_ps(y+i,_mm256_fmadd_ps(va,
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(h+i))),_mm256_loadu_ps(y+i)));
        for(; i<n; i++) y[i]+=a*h2f(h[i]);
#else
        for(int i=0;i<n;i++) y[i]+=a*h2f(h[i]);
#endif
        return; }
    { const signed char *b=(const signed char*)row; float s=a*rs;
      for(int i=0;i<n;i++) y[i]+=s*(float)b[i]; }
}
/* store one position's KD-wide K or V row, quantising per kv-head when dt==q8 */
static inline void kv_store(void *base, float *scales, int slot, const float *src,
                            int KD, int KV, int hd, int dt){
    if(dt==KVDT_F32){ memcpy((float*)base+(size_t)slot*KD, src, sizeof(float)*KD); return; }
    if(dt==KVDT_F16){ unsigned short *d=(unsigned short*)base+(size_t)slot*KD;
        for(int i=0;i<KD;i++) d[i]=f2h(src[i]); return; }
    { signed char *d=(signed char*)base+(size_t)slot*KD;
      for(int g=0; g<KV; g++){
        const float *s=src+(size_t)g*hd; float amax=0.0f;
        for(int i=0;i<hd;i++){ float v=fabsf(s[i]); if(v>amax)amax=v; }
        float sc=amax>0?amax/127.0f:1.0f, inv=amax>0?127.0f/amax:0.0f;
        scales[(size_t)slot*KV+g]=sc;
        for(int i=0;i<hd;i++){ int v=(int)lrintf(s[i]*inv);
            if(v>127)v=127; if(v<-127)v=-127; d[(size_t)g*hd+i]=(signed char)v; }
      } }
}
static double g_t_attn=0,g_t_qkv=0,g_t_moe=0,g_t_head=0,g_t_o=0; static int g_prof=-1;
/* Attention sub-phases. These are CPU-time sums across threads, so they add up
 * to more than the wall-clock attention bucket when threads>1 -- compare them
 * to each other, not to the wall figure. */
static double g_a_dot=0,g_a_exp=0,g_a_acc=0;
static int prof_on(void){ if(g_prof<0){ const char*e=getenv("TINYLM_PROF"); g_prof = e&&*e!='0'; } return g_prof; }
#define PT(bucket, ...) do{ if(prof_on()){ double _t0=wtime(); __VA_ARGS__; bucket+=wtime()-_t0; } else { __VA_ARGS__; } }while(0)

static Gen *gen_new(const Cfg *c, const Weights *w, int use_q8){
    int CH=c->T;
    tlp_init();                       /* must precede the per-thread scratch sizing */
    Gen *gn=calloc(1,sizeof(Gen));
    gn->kvdt=kv_dtype_env();
    size_t kvbytes=(size_t)c->T*c->KD*kv_elem_sz(gn->kvdt);
    gn->Kc=malloc(sizeof(void*)*c->L); gn->Vc=malloc(sizeof(void*)*c->L);
    gn->Ksc=gn->Vsc=NULL;
    if(gn->kvdt==KVDT_Q8){ gn->Ksc=malloc(sizeof(float*)*c->L); gn->Vsc=malloc(sizeof(float*)*c->L); }
    for(int l=0;l<c->L;l++){
        gn->Kc[l]=calloc(kvbytes,1); gn->Vc[l]=calloc(kvbytes,1);
        if(!gn->Kc[l]||!gn->Vc[l]){ fprintf(stderr,"OOM kv cache\n"); exit(1); }
        if(gn->kvdt==KVDT_Q8){ gn->Ksc[l]=fz((size_t)c->T*c->KV); gn->Vsc[l]=fz((size_t)c->T*c->KV); }
    }
    if(getenv("TINYLM_KV")||getenv("TINYLM_PROF"))
        fprintf(stderr,"[tinylm] KV cache %s: %.1f MB for %d layers x %d ctx\n",
                kv_name(gn->kvdt), 2.0*c->L*kvbytes/1e6, c->L, c->T);
    gn->x=fz((size_t)CH*c->D); gn->xn=fz((size_t)CH*c->D); gn->q=fz((size_t)CH*c->D);
    gn->k=fz((size_t)CH*c->KD); gn->v=fz((size_t)CH*c->KD); gn->ctx=fz((size_t)CH*c->D);
    gn->qkv=fz((size_t)CH*(c->D+2*c->KD));
    gn->tmp=fz((size_t)CH*c->D); gn->g=fz((size_t)CH*c->F); gn->u=fz((size_t)CH*c->F);
    gn->fn=fz((size_t)CH*c->D); gn->logits=fz((size_t)CH*c->V); gn->rinv=fz(CH);
    /* pb holds grp interleaved score rows per thread, so it is H*T not T.
     * Sized by the POOL's thread count -- attention no longer runs under OpenMP,
     * so omp_get_max_threads() would be both wrong and (when it is larger) waste. */
    gn->probs=fz((size_t)c->T*c->H*g_pool_nt);
    gn->accs =fz((size_t)c->H*c->hd*g_pool_nt);
    gn->part =fz((size_t)KV_SPLIT_MAX*c->H*(c->hd+2));
    gn->hlog=fz(c->V); gn->hh=fz(c->D); gn->ht=fz(c->D); gn->posbuf=malloc(sizeof(int)*CH);
    gn->moe_as=malloc(sizeof(int)*CH);
    gn->moe_gt=fz(CH); gn->moe_rp=fz((size_t)CH*(c->n_exp>0?c->n_exp:1));
    /* Pre-quantize the decode weights to int8. QKV/O + emb, the dense FFN, MTP
     * heads -- and, since this change, the MoE EXPERTS.
     *
     * The experts were the omission that mattered: in the 220M they are
     * 208.5M of 220.7M parameters (94.5%), so skipping them meant "q8 decode"
     * covered 5.5% of the model. Everything heavy ran fp32. */
    if(use_q8 && w){
        int L=c->L,D=c->D,KD=c->KD,F=c->F,V=c->V;
        int E=c->n_exp>0?c->n_exp:1;
        gn->qemb=qmat_NK(w->emb, V, D);
        gn->qwqkv=malloc(sizeof(QMat)*L); gn->qwo=malloc(sizeof(QMat)*L);
        /* one QMat per (layer, expert); dense models have E==1 so the indexing
           l*E+e collapses back to l */
        gn->qw13=malloc(sizeof(QMat)*(size_t)L*E);
        gn->qw2 =malloc(sizeof(QMat)*(size_t)L*E);
        for(int l=0;l<L;l++){
            gn->qwqkv[l]=qmat_KN3(w->wq[l],w->wk[l],w->wv[l], D, D,KD,KD);
            gn->qwo[l]  =qmat_KN(w->wo[l],D,D);
            for(int e=0;e<E;e++){
                size_t i=(size_t)l*E+e;
                gn->qw13[i]=qmat_KN_ilv(w->w1[l]+(size_t)e*D*F,
                                        w->w3[l]+(size_t)e*D*F, D, F);
                gn->qw2[i] =qmat_KN(w->w2[l]+(size_t)e*F*D, F, D);
            }
        }
        if(c->n_mtp>0){ gn->qmtp=malloc(sizeof(QMat)*c->n_mtp);
            for(int k=0;k<c->n_mtp;k++) gn->qmtp[k]=qmat_KN(w->mtp[k],D,D); }
        gn->q8=1;
    }
    return gn;
}

/* Decode-path MoE with int8 experts. Mirrors moe_forward exactly -- same
 * top-K routing, same gather/scatter, same maths -- but the three expert GEMMs
 * go through mm_dec, so they run int8 when a quantized copy exists.
 * The router stays fp32: it is D x E (512 x 16 here), i.e. 0.004% of the layer,
 * and it decides which expert every token gets, so it is the last thing worth
 * approximating. */
/* Prefill (N>1) helpers. These are elementwise/memcpy work, but they run while
 * the decode pool is live, so they must not open an OpenMP region -- two
 * thread pools awake at once oversubscribe the machine. */
typedef struct { int *assign; float *gate,*rprobs; int E; } MoeRtJob;
static void moe_rt_body(void *p,int lo,int hi,int tid){
    MoeRtJob *j=(MoeRtJob*)p; int E=j->E; (void)tid;
    for(int n=lo;n<hi;n++){
        float *r=me_rt+(size_t)n*E, *pr=j->rprobs+(size_t)n*E;
        float mx=-1e30f; for(int e=0;e<E;e++) if(r[e]>mx)mx=r[e];
        float s=0; for(int e=0;e<E;e++){ pr[e]=expf(r[e]-mx); s+=pr[e]; }
        float inv=1.0f/s; int best=0; for(int e=0;e<E;e++){ pr[e]*=inv; if(pr[e]>pr[best])best=e; }
        j->assign[n]=best; j->gate[n]=pr[best];
    }
}
typedef struct { const float *fnorm; float *out; const float *gate; int D; } MoeGsJob;
static void moe_gather_body(void *p,int lo,int hi,int tid){
    MoeGsJob *j=(MoeGsJob*)p; int D=j->D; (void)tid;
    for(int r=lo;r<hi;r++) memcpy(me_X+(size_t)r*D, j->fnorm+(size_t)me_idx[r]*D, sizeof(float)*D);
}
static void moe_scatter_body(void *p,int lo,int hi,int tid){
    MoeGsJob *j=(MoeGsJob*)p; int D=j->D; (void)tid;
    for(int r=lo;r<hi;r++){
        int n=me_idx[r]; float gt=j->gate[n];
        float *o=j->out+(size_t)n*D, *y=me_y+(size_t)r*D;
        for(int d=0;d<D;d++) o[d]+=gt*y[d];
    }
}
static void moe_silu_body(void *p,int lo,int hi,int tid){
    int F=*(const int*)p; (void)tid;
    for(int r=lo;r<hi;r++){
        size_t o=(size_t)r*F;
        for(int i=0;i<F;i++) me_a[o+i]=siluf(me_g[o+i])*me_u[o+i];
    }
}
/* out[rows][F] = silu(X @ W1) * (X @ W3), straight off the interleaved [2F][D]
 * int8 matrix. One walk of the weights, and the SwiGLU is folded into the same
 * pass -- the separate g/u buffers and the elementwise sweep over them are gone.
 * The activation is quantized once, not once per output column. */
typedef struct { const QMat *q13; const int8_t *xq; const float *xs; float *out;
                 int rows,F,nb,wkp,xkp; } SwigluJob;
static void swiglu_body(void *p,int lo,int hi,int tid){
    SwigluJob *j=(SwigluJob*)p; (void)tid;
    int nb=j->nb, wkp=j->wkp, xkp=j->xkp, rows=j->rows, F=j->F;
    for(int n=lo;n<hi;n++){
        const int8_t *r1=j->q13->q+(size_t)(2*n)*wkp, *r3=r1+wkp;
        const float  *s1=j->q13->s+(size_t)(2*n)*nb, *s3=s1+nb;
        for(int i=0;i<rows;i++){
            const int8_t *xq=j->xq+(size_t)i*xkp; const float *xs=j->xs+(size_t)i*nb;
            float a1=qdot_row(r1,s1,xq,xs,nb), a3=qdot_row(r3,s3,xq,xs,nb);
            j->out[(size_t)i*F+n]=siluf(a1)*a3;
        }
    }
}
static void swiglu_q8(const QMat *q13, const float *X, float *out, int rows, int D, int F){
    int nb=(D+31)/32, wkp=rowbytes(nb), xkp=nb*32;
    int8_t *xq, *lq=NULL; float *xs, *ls=NULL;
    if(rows==1){ me_qensure(D,F); q8_row(X,D,me_xq,me_xs); xq=me_xq; xs=me_xs; }
    else { lq=malloc((size_t)rows*xkp); ls=malloc(sizeof(float)*(size_t)rows*nb);
           for(int i=0;i<rows;i++) q8_row(X+(size_t)i*D,D,lq+(size_t)i*xkp,ls+(size_t)i*nb);
           xq=lq; xs=ls; }
    SwigluJob j={q13,xq,xs,out,rows,F,nb,wkp,xkp};
    tl_for(F, swiglu_body, &j);
    free(lq); free(ls);
}
typedef struct { const QMat *q2; const float *fin; float *out; float gt; int nbF,kpF; } MoeDnJob;
static void moe_dn_body(void *p,int lo,int hi,int tid){
    MoeDnJob *j=(MoeDnJob*)p; (void)tid;
    for(int n=lo;n<hi;n++)
        j->out[n]=j->fin[n]+j->gt*qdot_row(j->q2->q+(size_t)n*j->kpF, j->q2->s+(size_t)n*j->nbF,
                                           me_aq, me_as, j->nbF);
}
/* Router utilization histogram, opt-in via TINYLM_ROUTER. g_rhist[l*E+e] counts
 * how many token-slots layer l routed to expert e (all K slots). NULL = off. */
long *g_rhist=NULL;
static void moe_forward_q8(const Cfg *c, const Weights *w, const Gen *gn, int l,
                           const float *fnorm, const float *fin, float *out,
                           int N, int *assign, float *gate, float *rprobs){
    int D=c->D,F=c->F,E=c->n_exp;
    /* Single-token decode is the hot path, and it was paying for ~7 OpenMP
     * regions per layer (router, gather, w1, w3, silu, w2, scatter) to do a few
     * hundred KB of work each. On this 4-core laptop that fork/join cost MORE
     * than the arithmetic it parallelised: 8 threads measured 59 tok/s against
     * 113 tok/s single-threaded. Here the whole expert is TWO regions --
     * {w1,w3,silu} fused into one sweep over F, then w2 -- and everything
     * trivially small (router, top-1, residual add) stays serial. */
    int K=moe_topk(E);
    if(N==1 && gn->q8){
        me_ensure(1,D,F,E); me_qensure(D,F);
        mm(me_rt, fnorm, w->wr[l], 1, D, E);
        float mx=-1e30f; for(int e=0;e<E;e++) if(me_rt[e]>mx)mx=me_rt[e];
        float s=0; for(int e=0;e<E;e++){ rprobs[e]=expf(me_rt[e]-mx); s+=rprobs[e]; }
        float inv=1.0f/s; for(int e=0;e<E;e++) rprobs[e]*=inv;
        memcpy(out, fin, sizeof(float)*D);           /* residual; slots accumulate onto it */
        int nbF=(F+31)/32, kpF=rowbytes(nbF);        /* weight row stride (int8 me_aq activation) */
        for(int k=0;k<K;k++){
            int best=moe_kth(rprobs,E,k);
            assign[0]=best; gate[0]=rprobs[best];
            if(g_rhist) g_rhist[(size_t)l*E+best]++;
            size_t qi=(size_t)l*E+best;
            swiglu_q8(&gn->qw13[qi], fnorm, me_a, 1, D, F);
            q8_row(me_a,F,me_aq,me_as);
            MoeDnJob dj={&gn->qw2[qi],out,out,rprobs[best],nbF,kpF};  /* fin=out: += onto residual */
            tl_for(D, moe_dn_body, &dj);
        }
        return;
    }
    me_ensure(N,D,F,E);
    mm(me_rt, fnorm, w->wr[l], N, D, E);
    { MoeRtJob rj={assign,gate,rprobs,E};  tl_for(N, moe_rt_body, &rj); }   /* normalizes rprobs */
    memcpy(out, fin, sizeof(float)*(size_t)N*D);
    for(int k=0;k<K;k++){
        if(k) for(int n=0;n<N;n++){ int sel=moe_kth(rprobs+(size_t)n*E,E,k);
            assign[n]=sel; gate[n]=rprobs[(size_t)n*E+sel]; }
        if(g_rhist) for(int n=0;n<N;n++) g_rhist[(size_t)l*E+assign[n]]++;
        for(int e=0;e<E;e++){
            int ne=0; for(int n=0;n<N;n++) if(assign[n]==e) me_idx[ne++]=n;
            if(!ne) continue;
            { MoeGsJob gj={fnorm,NULL,gate,D};  tl_for(ne, moe_gather_body, &gj); }
            size_t qi=(size_t)l*E+e;
            /* gen_drop_fp32 nulls w->w1/w3/w2 once the int8 copies exist, so the
             * fp32 pointers may only be formed on the !q8 branch. */
            const QMat *q2 = gn->q8?&gn->qw2[qi]:NULL;
            const float *w2 = gn->q8?NULL:w->w2[l]+(size_t)e*F*D;
            if(gn->q8) swiglu_q8(&gn->qw13[qi], me_X, me_a, ne, D, F);
            else { mm(me_g, me_X, w->w1[l]+(size_t)e*D*F, ne, D, F);
                   mm(me_u, me_X, w->w3[l]+(size_t)e*D*F, ne, D, F);
                   tl_for(ne, moe_silu_body, &F); }
            mm_dec(me_y, me_a, w2, q2, ne, F, D);
            { MoeGsJob sj={NULL,out,gate,D};   tl_for(ne, moe_scatter_body, &sj); }
        }
    }
}

/* Release the fp32 parameter block once every heavy tensor has an int8 copy.
 *
 * Without this, quantizing the experts would ADD ~237 MB on top of the 883 MB
 * fp32 block rather than replacing it -- q8 would cost more memory than fp32,
 * which is the opposite of the point. Only the tensors with no int8 copy are
 * kept: the RMSNorm gains and the MoE router. For the 220M that residue is
 * 111K floats (444 KB) against 883 MB.
 *
 * emb is kept in fp32 despite having an int8 copy, because it has TWO roles:
 * the lm_head (which uses qemb) and the input embedding ROW LOOKUP, which
 * forward_chunk memcpy's straight out of w->emb. Quantizing the lookup would
 * add error to every token's starting representation for no memory benefit
 * worth having -- V*D is 16.8 MB against the 883 MB this frees.
 *
 * Returns the small buffer the caller must free instead of the old block. */
static float *gen_drop_fp32(const Cfg *c, Weights *w, float **P){
    if(!P || !*P) return NULL;
    int L=c->L, D=c->D, E=c->n_exp, V=c->V;
    size_t need=(size_t)V*D + D
              + (size_t)L*(2*D + (E>0?(size_t)D*E:0) + (c->qknorm?2*(size_t)c->hd:0));
    float *keep=malloc(sizeof(float)*need), *p=keep;
    if(!keep) return NULL;                       /* keep the fp32 block instead */
    memcpy(p,w->emb,sizeof(float)*(size_t)V*D); w->emb=p; p+=(size_t)V*D;
    for(int l=0;l<L;l++){
        memcpy(p,w->an1[l],sizeof(float)*D); w->an1[l]=p; p+=D;
        memcpy(p,w->an2[l],sizeof(float)*D); w->an2[l]=p; p+=D;
        if(c->qknorm){
            memcpy(p,w->qn[l],sizeof(float)*c->hd); w->qn[l]=p; p+=c->hd;
            memcpy(p,w->kn[l],sizeof(float)*c->hd); w->kn[l]=p; p+=c->hd;
        }
        if(E>0){ memcpy(p,w->wr[l],sizeof(float)*(size_t)D*E); w->wr[l]=p; p+=(size_t)D*E; }
    }
    memcpy(p,w->nf,sizeof(float)*D); w->nf=p;
    /* the rest now exists only as int8; null the stale pointers so a missed
       code path crashes loudly instead of reading freed memory */
    for(int l=0;l<L;l++){ w->wq[l]=w->wk[l]=w->wv[l]=w->wo[l]=NULL;
                          w->w1[l]=w->w3[l]=w->w2[l]=NULL; }
    free(*P); *P=NULL;
    return keep;
}

/* --- Phase 2A: low-peak TLQ3 loader ---------------------------------------
 * The default TLQ3 path used to decode the whole selective-q4 file into a full
 * fp32 param block (537 MB for the 137M model), build int8 QMats from it, then
 * free it -- a transient 1.2 GB peak just to end up at ~250 MB. This loader
 * decodes ONE tensor at a time straight into a scratch buffer, builds its int8
 * QMat, and moves on, so the fp32 block never exists. Output is byte-identical
 * to the old path: the same bytes decode to the same fp32 values and the same
 * qmat_* run on them. TINYLM_HIGHPEAK falls back to the old loader. */
static void f16_gap(Rd *r, float *dst, size_t n){
    for(size_t k=0;k<n;k++){ uint16_t h; if(!rd(r,&h,2)){fprintf(stderr,"q4 gap\n");exit(1);} dst[k]=f16_to_f32(h); }
}
/* One selective-q4 span: fp16 scale + 4-bit nibbles per `group`, reset at j=0
 * (mirrors q4_selective_decode's inner loop exactly). */
static void q4_decode_span(Rd *r, float *dst, size_t sn, int group){
    for(size_t j=0;j<sn;j+=group){
        int g=(j+(size_t)group<=sn)?group:(int)(sn-j);
        uint16_t sh; if(!rd(r,&sh,2)){fprintf(stderr,"q4 span scale\n");exit(1);} float scale=f16_to_f32(sh);
        for(int t=0;t<g;t+=2){
            uint8_t b; if(!rd(r,&b,1)){fprintf(stderr,"q4 span byte\n");exit(1);}
            int lo=b&0xF; if(lo>=8)lo-=16; dst[j+t]=(float)lo*scale;
            if(t+1<g){ int hi=(b>>4)&0xF; if(hi>=8)hi-=16; dst[j+t+1]=(float)hi*scale; }
        }
    }
}
static Gen *load_q4_lowpeak(const char *path, Cfg *c, Weights *w, Tok **tk_out, float **keep_out){
    FILE *fp=fopen(path,"rb");
    if(!fp){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    size_t rcap=1<<18; uint8_t *rbuf=malloc(rcap);       /* refill window, not the whole file */
    if(!rbuf){fprintf(stderr,"OOM rbuf\n");exit(1);}
    Rd r={rbuf,rbuf,fp,rbuf,rcap}; char m[4]; rd(&r,m,4);
    if(memcmp(m,"TLQ3",4)){fprintf(stderr,"load_q4_lowpeak: not TLQ3\n");exit(1);}
    int hdr[HDR_MAX]; rd(&r,hdr,4*15); cfg_from_hdr(c,hdr,6);
    int qtype=0,group=128; rd(&r,&qtype,4); rd(&r,&group,4);
    check_supported(c,path);
    *tk_out=NULL;

    int L=c->L,D=c->D,KD=c->KD,F=c->F,V=c->V,hd=c->hd,ne=c->n_exp;
    int E=n_slots(c);                            /* == n_exp for supported models */

    /* keep block: exactly gen_drop_fp32's layout (emb, per-layer small tensors, nf) */
    size_t need=(size_t)V*D + D
              + (size_t)L*(2*(size_t)D + (ne>0?(size_t)D*ne:0) + (c->qknorm?2*(size_t)hd:0));
    float *keep=malloc(sizeof(float)*need); if(!keep){fprintf(stderr,"OOM keep\n");exit(1);}
    memset(w,0,sizeof(*w));
    w->an1=malloc(sizeof(float*)*L); w->an2=malloc(sizeof(float*)*L);
    w->qn =malloc(sizeof(float*)*L); w->kn =malloc(sizeof(float*)*L);
    w->wr =malloc(sizeof(float*)*L);
    w->wq =malloc(sizeof(float*)*L); w->wk =malloc(sizeof(float*)*L);
    w->wv =malloc(sizeof(float*)*L); w->wo =malloc(sizeof(float*)*L);
    w->w1 =malloc(sizeof(float*)*L); w->w3 =malloc(sizeof(float*)*L);
    w->w2 =malloc(sizeof(float*)*L); w->mtp=NULL;
    float *p=keep;
    w->emb=p; p+=(size_t)V*D;
    for(int l=0;l<L;l++){
        w->an1[l]=p; p+=D; w->an2[l]=p; p+=D;
        if(c->qknorm){ w->qn[l]=p; p+=hd; w->kn[l]=p; p+=hd; } else { w->qn[l]=NULL; w->kn[l]=NULL; }
        if(ne>0){ w->wr[l]=p; p+=(size_t)D*ne; } else w->wr[l]=NULL;
        w->wq[l]=w->wk[l]=w->wv[l]=w->wo[l]=NULL;   /* heavy: int8-only, never dereffed at q8 */
        w->w1[l]=w->w3[l]=w->w2[l]=NULL;
    }
    w->nf=p; p+=D;

    /* two scratch buffers, reused across every matrix. sA holds a whole w1 span
     * (its experts must persist while w3 streams in to pair with them); sB only
     * ever holds ONE expert (w3_e / w2_e) or a small attn matrix, since those
     * spans are consumed expert-by-expert (D*F and F*D are group-aligned). */
    size_t qkv_n=(size_t)D*D+2*(size_t)D*KD, bigA=(size_t)E*D*F;
    if(qkv_n>bigA) bigA=qkv_n;
    size_t bigB=(size_t)D*F; if((size_t)F*D>bigB) bigB=(size_t)F*D;
    if((size_t)D*D>bigB) bigB=(size_t)D*D;
    float *sA=malloc(sizeof(float)*bigA), *sB=malloc(sizeof(float)*bigB);
    if(!sA||!sB){fprintf(stderr,"OOM q4 scratch\n");exit(1);}

    Gen *gn=gen_new(c,w,0);                       /* buffers only, no QMats */
    gn->qwqkv=malloc(sizeof(QMat)*L); gn->qwo=malloc(sizeof(QMat)*L);
    gn->qw13=malloc(sizeof(QMat)*(size_t)L*E); gn->qw2=malloc(sizeof(QMat)*(size_t)L*E);
    if(c->n_mtp>0) gn->qmtp=malloc(sizeof(QMat)*c->n_mtp);

    /* walk the file in byte order (see q4_spans): gap(emb,an1[0]),
     * per layer [wq wk wv wo | gap(qn,kn,an2,wr) | w1 w3 w2 | gap(an1[l+1])],
     * trailing gap(nf, mtp). */
    f16_gap(&r, w->emb, (size_t)V*D);
    gn->qemb=qmat_NK(w->emb, V, D);
    f16_gap(&r, w->an1[0], D);
    for(int l=0;l<L;l++){
        q4_decode_span(&r, sA,                              (size_t)D*D,  group); /* wq */
        q4_decode_span(&r, sA+(size_t)D*D,                 (size_t)D*KD, group); /* wk */
        q4_decode_span(&r, sA+(size_t)D*D+(size_t)D*KD,    (size_t)D*KD, group); /* wv */
        q4_decode_span(&r, sB,                              (size_t)D*D,  group); /* wo */
        gn->qwqkv[l]=qmat_KN3(sA, sA+(size_t)D*D, sA+(size_t)D*D+(size_t)D*KD, D, D, KD, KD);
        gn->qwo[l]  =qmat_KN(sB, D, D);
        if(c->qknorm){ f16_gap(&r, w->qn[l], hd); f16_gap(&r, w->kn[l], hd); }
        f16_gap(&r, w->an2[l], D);
        if(ne>0) f16_gap(&r, w->wr[l], (size_t)D*ne);
        q4_decode_span(&r, sA, (size_t)E*D*F, group);      /* w1: all experts (kept in sA) */
        for(int e=0;e<E;e++){                              /* w3: one expert at a time into sB */
            q4_decode_span(&r, sB, (size_t)D*F, group);    /* D*F is group-aligned -> same bytes */
            gn->qw13[(size_t)l*E+e]=qmat_KN_ilv(sA+(size_t)e*D*F, sB, D, F);
        }
        for(int e=0;e<E;e++){                              /* w2: one expert at a time into sB */
            q4_decode_span(&r, sB, (size_t)F*D, group);
            gn->qw2[(size_t)l*E+e]=qmat_KN(sB, F, D);
        }
        if(l<L-1) f16_gap(&r, w->an1[l+1], D);
    }
    f16_gap(&r, w->nf, D);
    if(c->n_mtp>0) for(int k=0;k<c->n_mtp;k++){ f16_gap(&r, sA, (size_t)D*D); gn->qmtp[k]=qmat_KN(sA, D, D); }
    gn->q8=1;

    free(sA); free(sB);
    int tl=0;
    if(rd(&r,&tl,4) && tl>0){
        uint8_t *tb=malloc(tl);                          /* tok blob: drain window, fread the rest */
        size_t rem=(size_t)(r.end-r.p); if(rem>(size_t)tl) rem=tl;
        memcpy(tb,r.p,rem); r.p+=rem;
        if(rem<(size_t)tl) fread(tb+rem,1,(size_t)tl-rem,fp);
        *tk_out=tok_parse(tb,tl,1); free(tb);
    }
    fclose(fp); free(rbuf);
    *keep_out=keep;
    return gn;
}

/* One (row, kv-head, chunk) slice of the attention loop. Extracted from
 * forward_chunk so the decode pool can run it: at m=1 this is the only place
 * left with real per-token parallelism, and a libgomp region here cost more
 * than the dots it was splitting. `tid` now comes from the pool, not
 * omp_get_thread_num, and indexes the same per-thread probs/accs scratch. */
typedef struct {
    const Cfg *c; Gen *gn;
    int l, start_pos, wlim, W, NC, KV, grp, hd, D, KD, dt, esz;
    float scale;
} AttnJob;
static void attn_body(void *p, int jlo, int jhi, int tid){
    AttnJob *aj=(AttnJob*)p;
    const Cfg *c=aj->c; Gen *gn=aj->gn;
    int l=aj->l, start_pos=aj->start_pos, wlim=aj->wlim, W=aj->W, NC=aj->NC;
    int KV=aj->KV, grp=aj->grp, hd=aj->hd, D=aj->D, KD=aj->KD, dt=aj->dt, esz=aj->esz;
    int H=c->H; float scale=aj->scale;
    for(int job=jlo;job<jhi;job++){
        int ck=job%NC, kvh=(job/NC)%KV, j=job/(NC*KV);
        int pos=start_pos+j;
        int lo = pos>=wlim ? pos-wlim+1 : 0, cnt = pos-lo+1;
        int a=(int)(((long)cnt*ck)/NC), b=(int)(((long)cnt*(ck+1))/NC), len=b-a;
        /* The ring range is at most two CONTIGUOUS spans; walking them
         * directly keeps a hardware modulo out of the innermost loop and
         * lets the prefetcher work. */
        int s0=lo%W, r1=W-s0; if(r1>cnt) r1=cnt;
        float *pb =gn->probs+(size_t)tid*H*W;          /* [grp][len] */
        float *acc=gn->accs +(size_t)tid*H*hd;         /* [grp][hd]  */
        float mxh[64], lsum[64];
        for(int g=0;g<grp;g++){ mxh[g]=-1e30f; lsum[g]=0.0f;
            for(int d=0;d<hd;d++) acc[(size_t)g*hd+d]=0.0f; }
        if(len>0){
            const char *Kb=(const char*)gn->Kc[l]+(size_t)kvh*hd*esz;
            const char *Vb=(const char*)gn->Vc[l]+(size_t)kvh*hd*esz;
            const float *Ks=gn->Ksc?gn->Ksc[l]:NULL, *Vs=gn->Vsc?gn->Vsc[l]:NULL;
            double _p0=prof_on()?wtime():0.0;
            /* pass 1: one K row -> grp scores */
            for(int t=a;t<b;t++){
                int slot = t<r1 ? s0+t : t-r1;
                const void *kr=Kb+(size_t)slot*KD*esz;
                float rs=Ks?Ks[(size_t)slot*KV+kvh]:1.0f;
                for(int g=0;g<grp;g++){
                    const float *qh=gn->q+(size_t)j*D+(size_t)(kvh*grp+g)*hd;
                    float d0=kv_dot(qh,kr,hd,dt,rs)*scale;
                    pb[(size_t)g*len+(t-a)]=d0; if(d0>mxh[g])mxh[g]=d0;
                }
            }
            /* With no split, normalise BEFORE accumulating. That reproduces
             * the pre-fusion multiply order exactly, so f32 stays bit-exact
             * with every historical number; the split path cannot, because
             * the merge is what reassociates. */
            double _p1=prof_on()?wtime():0.0;
            int xex=exact_exp_on();
            for(int g=0;g<grp;g++){ float s=0.0f; float *r=pb+(size_t)g*len; float mg=mxh[g];
                if(xex){ for(int t=0;t<len;t++){ r[t]=expf(r[t]-mg); s+=r[t]; } }
                else   { for(int t=0;t<len;t++){ r[t]=fast_expf(r[t]-mg); s+=r[t]; } }
                lsum[g]=s;
                if(NC==1){ float inv=s>0?1.0f/s:0.0f; for(int t=0;t<len;t++) r[t]*=inv; } }
            double _p2=prof_on()?wtime():0.0;
            /* pass 2: one V row -> grp accumulators */
            for(int t=a;t<b;t++){
                int slot = t<r1 ? s0+t : t-r1;
                const void *vr=Vb+(size_t)slot*KD*esz;
                float rs=Vs?Vs[(size_t)slot*KV+kvh]:1.0f;
                for(int g=0;g<grp;g++)
                    kv_axpy(acc+(size_t)g*hd, vr, pb[(size_t)g*len+(t-a)], hd, dt, rs);
            }
            if(prof_on()){ double _p3=wtime();
                #pragma omp atomic
                g_a_dot += _p1-_p0;
                #pragma omp atomic
                g_a_exp += _p2-_p1;
                #pragma omp atomic
                g_a_acc += _p3-_p2; }
        }
        if(NC==1){                       /* already normalised above */
            for(int g=0;g<grp;g++)
                memcpy(gn->ctx+(size_t)j*D+(size_t)(kvh*grp+g)*hd,
                       acc+(size_t)g*hd, sizeof(float)*hd);
        } else {
            for(int g=0;g<grp;g++){
                float *p=gn->part+((size_t)(kvh*grp+g)*NC+ck)*(hd+2);
                p[0]=mxh[g]; p[1]=lsum[g];
                memcpy(p+2, acc+(size_t)g*hd, sizeof(float)*hd);
            }
        }
    }
}

/* Process m tokens at positions start_pos..start_pos+m-1 through a sliding ring
 * cache. Fills gn->logits (m*V, main head) and gn->fn (m*D, final hidden). The
 * cache-write+attention step is sequential over the chunk to keep the ring valid
 * across a wrap; the heavy matmuls stay batched (parallel over the m rows). */
static void forward_chunk(const Cfg *c, const Weights *w, Gen *gn,
                          const int *tokens, int m, int start_pos){
    int D=c->D,F=c->F,KD=c->KD,V=c->V,H=c->H,KV=c->KV,hd=c->hd,half=c->half,W=c->T;
    float scale=1.0f/sqrtf((float)hd); int grp=H/KV;
    for(int j=0;j<m;j++){ memcpy(gn->x+(size_t)j*D, w->emb+(size_t)tokens[j]*D, sizeof(float)*D);
                          gn->posbuf[j]=start_pos+j; }
    if(w->ng.R>0) ngram_forward(c, &w->ng, tokens, m, start_pos, gn->x);
    for(int l=0;l<c->L;l++){
        rmsnorm_fwd(gn->x, w->an1[l], gn->xn, gn->rinv, m, D);
        /* One fused Q|K|V GEMV, then split the rows back out. The split copies
         * m*(D+2*KD) floats -- 3 KB at m=1, against the 655 KB of weights the
         * fusion streams once instead of three times. */
        PT(g_t_qkv,
        if(gn->q8){
            int QN=D+2*KD;
            mm_dec(gn->qkv, gn->xn, NULL, &gn->qwqkv[l], m, D, QN);
            for(int j=0;j<m;j++){
                const float *r=gn->qkv+(size_t)j*QN;
                memcpy(gn->q+(size_t)j*D,  r,      sizeof(float)*D);
                memcpy(gn->k+(size_t)j*KD, r+D,    sizeof(float)*KD);
                memcpy(gn->v+(size_t)j*KD, r+D+KD, sizeof(float)*KD);
            }
        } else {
            mm(gn->q, gn->xn, w->wq[l], m, D, D);
            mm(gn->k, gn->xn, w->wk[l], m, D, KD);
            mm(gn->v, gn->xn, w->wv[l], m, D, KD);
        });
        if(c->qknorm){
            qknorm_apply(gn->q, w->qn[l], m, H,  hd);
            qknorm_apply(gn->k, w->kn[l], m, KV, hd);
        }
        rope_apply(gn->q, gn->posbuf, m, H,  hd, half, 0);
        rope_apply(gn->k, gn->posbuf, m, KV, hd, half, 0);
        /* The KV ring holds W=c->T entries, but a TLM5 model was TRAINED with a
         * narrower window. Attending over the full ring would show it 4x more
         * context than it ever saw (at win=1024, T=4096) -- a silent
         * train/inference mismatch. Clamp to the trained window. */
        int eff=cfg_lwin(c,l), wlim=(eff>0 && eff<W)?eff:W;
        /* KV writes are hoisted out of the attention loop so the attention below
         * can run in ONE parallel region over (row, head) instead of being
         * serialised per row. Safe for any m<=W: two chunk positions p,p' map to
         * the same ring slot only if p==p' (mod W), and |p-p'| < m <= W forces
         * p==p'. Each position's window is at most W wide, so nothing it reads
         * has been overwritten by a later position in the same chunk. */
        int dt=gn->kvdt, esz=kv_elem_sz(dt);
        for(int j=0;j<m;j++){
            int slot=(start_pos+j)%W;
            kv_store(gn->Kc[l], gn->Ksc?gn->Ksc[l]:NULL, slot, gn->k+(size_t)j*KD, KD,KV,hd,dt);
            kv_store(gn->Vc[l], gn->Vsc?gn->Vsc[l]:NULL, slot, gn->v+(size_t)j*KD, KD,KV,hd,dt);
        }
        double _ta = prof_on()?wtime():0.0;
        /* Jobs are (row, kv-head, chunk), NOT (row, head).
         *
         * There are only KV=2 distinct cached rows per position but H=8 heads,
         * so the previous head-per-job loop read every K and V row `grp`=4 times
         * over. Iterating a kv-head GROUP instead loads each row once and dots
         * it against all grp queries while it sits in L1 -- 4x less traffic
         * through the level that actually limits this loop.
         *
         * Folding 8 jobs into 2 would have starved a 4-thread machine, so the
         * position range is split into NC chunks and the partial softmaxes are
         * merged afterwards (max/sum rescale, the standard online form). That
         * keeps >= 2 jobs per thread while still getting the reuse. Splitting is
         * only done for m==1; prefill already has m*KV jobs to hand out. */
        int nt=g_pool_nt, NC=1;
        if(m==1 && KV<2*nt && kv_split_on()){
            NC=(2*nt+KV-1)/KV; if(NC>KV_SPLIT_MAX)NC=KV_SPLIT_MAX; if(NC<1)NC=1; }
        int njob=m*KV*NC;
        { AttnJob aj={c,gn,l,start_pos,wlim,W,NC,KV,grp,hd,D,KD,dt,esz,scale};
          tl_for(njob, attn_body, &aj); }
        if(NC>1){                                       /* online softmax merge */
            for(int h=0;h<H;h++){
                float M=-1e30f;
                for(int ck=0;ck<NC;ck++){ float v=gn->part[((size_t)h*NC+ck)*(hd+2)]; if(v>M)M=v; }
                float L=0.0f; float *ch=gn->ctx+(size_t)h*hd;
                for(int d=0;d<hd;d++) ch[d]=0.0f;
                for(int ck=0;ck<NC;ck++){
                    float *p=gn->part+((size_t)h*NC+ck)*(hd+2);
                    if(p[1]<=0.0f) continue;
                    float w=expf(p[0]-M); L+=p[1]*w;
                    for(int d=0;d<hd;d++) ch[d]+=p[2+d]*w;
                }
                float inv=L>0?1.0f/L:0.0f;
                for(int d=0;d<hd;d++) ch[d]*=inv;
            }
        }
        if(prof_on()) g_t_attn += wtime()-_ta;
        PT(g_t_o, mm_dec(gn->tmp, gn->ctx, w->wo[l], gn->q8?&gn->qwo[l]:NULL, m, D, D));
        for(size_t i=0;i<(size_t)m*D;i++) gn->x[i]+=gn->tmp[i];
        rmsnorm_fwd(gn->x, w->an2[l], gn->xn, gn->rinv, m, D);
        if(c->n_exp>0){
            PT(g_t_moe, moe_forward_q8(c,w,gn,l, gn->xn, gn->x, gn->tmp, m,
                                       gn->moe_as, gn->moe_gt, gn->moe_rp));  /* out=tmp=x+ffn */
            memcpy(gn->x, gn->tmp, sizeof(float)*(size_t)m*D);
        } else {
            if(gn->q8) swiglu_q8(&gn->qw13[l], gn->xn, gn->u, m, D, F);
            else { mm(gn->g, gn->xn, w->w1[l], m, D, F);
                   mm(gn->u, gn->xn, w->w3[l], m, D, F);
                   for(size_t i=0;i<(size_t)m*F;i++) gn->u[i]=siluf(gn->g[i])*gn->u[i]; }
            mm_dec(gn->tmp, gn->u, w->w2[l], gn->q8?&gn->qw2[l]:NULL, m, F, D);
            for(size_t i=0;i<(size_t)m*D;i++) gn->x[i]+=gn->tmp[i];
        }
    }
    rmsnorm_fwd(gn->x, w->nf, gn->fn, gn->rinv, m, D);
    PT(g_t_head, mmbt_dec(gn->logits, gn->fn, w->emb, gn->q8?&gn->qemb:NULL, m, V, D));
}
static void prof_report(double wall){
    if(!prof_on()) return;
    double o=wall-(g_t_attn+g_t_qkv+g_t_o+g_t_moe+g_t_head);
    fprintf(stderr,"[prof] attention %.3fs %.1f%% | qkv %.3fs %.1f%% | o_proj %.3fs %.1f%%"
                   " | moe %.3fs %.1f%% | lm_head %.3fs %.1f%% | other %.3fs %.1f%%\n",
        g_t_attn,100*g_t_attn/wall, g_t_qkv,100*g_t_qkv/wall, g_t_o,100*g_t_o/wall,
        g_t_moe,100*g_t_moe/wall, g_t_head,100*g_t_head/wall, o,100*o/wall);
    double as=g_a_dot+g_a_exp+g_a_acc; if(as<=0) return;
    fprintf(stderr,"[prof] attention cpu-time split: qk-dot %.3fs %.1f%% | softmax/expf %.3fs %.1f%%"
                   " | av-accum %.3fs %.1f%%  (sum %.3fs across threads)\n",
        g_a_dot,100*g_a_dot/as, g_a_exp,100*g_a_exp/as, g_a_acc,100*g_a_acc/as, as);
}

static int argmax_v(const float *v, int n){ int mi=0; for(int i=1;i<n;i++) if(v[i]>v[mi])mi=i; return mi; }

/* Top-k admission threshold = the k-th largest of v[0..n).
 *
 * The old sampler ran a k-pass selection sort over the whole vocabulary: at
 * topk=40, V=8192 that is ~327k compares per token, plus a malloc and a memcpy
 * of the logits, and it measured ~0.8 ms/token -- 15% of decode. A k-element
 * min-heap does one compare against the current k-th best for the ~99.5% of
 * entries that lose immediately, so it is a single pass and no allocation.
 * `h` is k floats of caller scratch. Ties are admitted, exactly as before. */
static void heap_sift(float *h, int k, int i){
    for(;;){
        int l=2*i+1, r=l+1, s=i;
        if(l<k && h[l]<h[s]) s=l;
        if(r<k && h[r]<h[s]) s=r;
        if(s==i) return;
        float t=h[i]; h[i]=h[s]; h[s]=t; i=s;
    }
}
static float topk_thresh(const float *v, int n, int k, float *h){
    if(k>=n) return -3.0e38f;
    for(int i=0;i<k;i++) h[i]=v[i];
    for(int i=k/2-1;i>=0;i--) heap_sift(h,k,i);
    for(int i=k;i<n;i++) if(v[i]>h[0]){ h[0]=v[i]; heap_sift(h,k,0); }
    return h[0];
}
/* Sample one token from raw logits. `work` is V floats of caller scratch (it
 * doubles as the heap, which is finished with before the probabilities land in
 * it). Arithmetically the same as the old five-pass form: exp(l/T - max(l)/T)
 * == exp((l-max(l))/T), and the mask threshold is unchanged. */
/* CTRL-style repetition penalty (Keskar et al. 2019): each token seen in the last
 * `rep_win` positions has its logit divided by `rep_pen` if positive, multiplied if
 * negative -- pushing its probability down either way. Applied once per unique token
 * (duplicates in the window don't compound, matching HF's behaviour). This is what
 * breaks the degenerate loops a low-capacity model falls into; it runs on a penalized
 * COPY (`work`) so the caller's logits row stays intact. `heap` is topk scratch. */
static int sample_logits(const Cfg *c, const float *lg, float *work, float *heap,
                         float temp, int topk, const int *hist, int hn,
                         float rep_pen, int rep_win){
    int V=c->V;
    for(int i=0;i<V;i++) work[i]=lg[i];
    if(rep_pen>1.0f && hn>0){
        int lo = hn>rep_win ? hn-rep_win : 0;
        for(int j=lo;j<hn;j++){
            int t=hist[j]; if(t<0||t>=V) continue;
            int dup=0; for(int u=lo;u<j;u++) if(hist[u]==t){dup=1;break;}
            if(dup) continue;
            work[t] = work[t]>0 ? work[t]/rep_pen : work[t]*rep_pen;
        }
    }
    if(topk<=1 || temp<=1e-4f) return argmax_v(work,V);
    float it = temp>1e-6f?temp:1e-6f;
    float thr = (topk>0 && topk<V) ? topk_thresh(work,V,topk,heap) : -3.0e38f;
    float mx=work[0]; for(int i=1;i<V;i++) if(work[i]>mx)mx=work[i];
    float sum=0.0f;
    for(int i=0;i<V;i++){ float e = work[i]>=thr ? expf((work[i]-mx)/it) : 0.0f; work[i]=e; sum+=e; }
    float r=rnd_uniform()*sum, ac=0.0f;
    for(int i=0;i<V;i++){ ac+=work[i]; if(ac>=r) return i; }
    return V-1;
}
/* MTP head k argmax: hk = fn1 + silu(fn1@Wk); argmax(hk @ emb^T) */
static int mtp_argmax(const Cfg *c, const Weights *w, Gen *gn, const float *fn1, int k){
    int D=c->D;
    mm_dec(gn->ht, fn1, w->mtp[k], gn->q8?&gn->qmtp[k]:NULL, 1, D, D);
    for(int i=0;i<D;i++) gn->hh[i]=fn1[i]+siluf(gn->ht[i]);
    mmbt_dec(gn->hlog, gn->hh, w->emb, gn->q8?&gn->qemb:NULL, 1, c->V, c->D);
    return argmax_v(gn->hlog, c->V);
}

/* ----------------------------- generation core -------------------------- */
/* Prefill `prompt` (chat template applied if the model is alpaca-typed) and
 * decode `nnew` tokens, printing as it goes. Greedy + MTP heads -> self-
 * speculative (exact). echo_prompt prints the formatted prompt; show_stats
 * prints the trailing timing/acceptance line. Caller owns c/w/gn/tk. */
static void run_generation(const Cfg *c, const Weights *w, Gen *gn, const Tok *tk,
                           const char *prompt, int nnew, float temp, int topk, int stop_eot,
                           int echo_prompt, int show_stats){
    char fmt[16384];
    /* tmpl 2 = chatml, added by the CUDA build. Without this case a chatml model
     * falls through to RAW here and never sees the <|user|>/<|assistant|> markers
     * it was trained on -- it still generates, just noticeably worse, which is
     * the kind of gap that looks like a quality regression rather than a bug.
     * Markers are plain text, NOT special tokens: tok_encode has no
     * special-token path, so they must tokenize exactly as they did in training. */
    if(c->tmpl==1)      snprintf(fmt,sizeof(fmt),"### Instruction:\n%s\n\n### Response:\n", prompt);
    else if(c->tmpl==2) snprintf(fmt,sizeof(fmt),"<|user|>\n%s\n<|assistant|>\n", prompt);
    else                snprintf(fmt,sizeof(fmt),"%s", prompt);
    int pn; int *pids=tok_encode(tk, fmt, &pn);
    rope_init(c, pn + nnew + 8);
    /* repetition penalty (env-tunable, like TINYLM_Q8/NOSPEC). Default on and mild;
     * TINYLM_REP=1.0 disables. When active, the greedy self-speculative path is off
     * because its MTP drafting bypasses the sampler where the penalty lives. */
    const char *rp_env=getenv("TINYLM_REP"); float rep_pen = rp_env?atof(rp_env):1.3f;
    const char *rw_env=getenv("TINYLM_REPWIN"); int rep_win = rw_env?atoi(rw_env):128;
    if(rep_pen<1.0f) rep_pen=1.0f;
    if(getenv("TINYLM_ROUTER") && c->n_exp>0){
        free(g_rhist); g_rhist=calloc((size_t)c->L*c->n_exp,sizeof(long)); }
    int spec = (c->n_mtp>0) && (topk<=1 || temp<=1e-4f) && !getenv("TINYLM_NOSPEC") && rep_pen<=1.0f;
    if(echo_prompt){ fputs(fmt,stdout); fflush(stdout); }
    int *hist=malloc(sizeof(int)*(size_t)(pn+nnew+8)); int hn=0;
    for(int i=0;i<pn;i++) hist[hn++]=pids[i];

    double w0=wtime();
    /* lg_cur tracks the logits row that goes with fn_cur. forward_chunk already
     * ran the (int8) lm_head over the whole chunk; the sampler below used to
     * throw that away and recompute the V x D head in fp32, streaming all
     * 16.8 MB of w->emb every token. Measured at 1.09 ms/token -- 21% of decode
     * -- for a result forward_chunk had already produced. */
    int pos=0; float *fn_cur=NULL, *lg_cur=NULL;
    for(int i=0;i<pn;){ int m=pn-i; if(m>c->T)m=c->T; forward_chunk(c,w,gn,pids+i,m,pos);
                        fn_cur=gn->fn+(size_t)(m-1)*c->D;
                        lg_cur=gn->logits+(size_t)(m-1)*c->V; pos+=m; i+=m; }
    if(pn==0){ int z=tk->eot; forward_chunk(c,w,gn,&z,1,pos); fn_cur=gn->fn; lg_cur=gn->logits; pos=1; }
    free(pids);

    int generated=0; long acc_sum=0, fwd_chunks=0;
    if(spec){
        int K=c->n_mtp; int *draft=malloc(sizeof(int)*(K+1)); int done=0;
        while(generated<nnew && !done){
            int x1=argmax_v(lg_cur, c->V); draft[0]=x1;
            for(int k=0;k<K;k++) draft[k+1]=mtp_argmax(c,w,gn,fn_cur,k);
            forward_chunk(c,w,gn,draft,K+1,pos); fwd_chunks++;
            if(stop_eot && x1==tk->eot) done=1; else { tok_print(tk,x1); generated++; }
            int n_acc=0;
            for(int r=0;r<K && !done && generated<nnew;r++){
                int m_r=argmax_v(gn->logits+(size_t)r*c->V, c->V);
                if(draft[r+1]==m_r){ if(stop_eot && m_r==tk->eot) done=1;
                                     else { tok_print(tk,m_r); generated++; n_acc++; } }
                else break;
            }
            acc_sum+=n_acc; pos+=n_acc+1; fn_cur=gn->fn+(size_t)n_acc*c->D;
            lg_cur=gn->logits+(size_t)n_acc*c->V; fflush(stdout);
        }
        free(draft);
    } else {
        float *work=malloc(sizeof(float)*c->V);
        float *heap=malloc(sizeof(float)*(size_t)(topk>0?topk:1));
        while(generated<nnew){
            int nxt=sample_logits(c, lg_cur, work, heap, temp, topk, hist, hn, rep_pen, rep_win);
            if(stop_eot && nxt==tk->eot) break;
            tok_print(tk,nxt); fflush(stdout); generated++;
            hist[hn++]=nxt;
            forward_chunk(c,w,gn,&nxt,1,pos); fwd_chunks++; pos++;
            fn_cur=gn->fn; lg_cur=gn->logits;
        }
        free(work); free(heap);
    }
    free(hist);
    if(g_rhist){
        int E=c->n_exp, L=c->L;
        long tot=0; for(int i=0;i<L*E;i++) tot+=g_rhist[i];
        long *agg=calloc(E,sizeof(long));
        for(int l=0;l<L;l++) for(int e=0;e<E;e++) agg[e]+=g_rhist[(size_t)l*E+e];
        fprintf(stderr,"\n[router] %d experts x %d layers, %ld routing decisions total\n",E,L,tot);
        double ideal=100.0/E;
        fprintf(stderr,"[router] aggregate expert usage (ideal %.2f%% each):\n",ideal);
        int dead=0; double maxp=0; long mx=0;
        for(int e=0;e<E;e++){ double p=tot?100.0*agg[e]/tot:0; if(p>maxp)maxp=p; if(agg[e]>mx)mx=agg[e];
            if(agg[e]==0)dead++; }
        for(int e=0;e<E;e++){ double p=tot?100.0*agg[e]/tot:0;
            int bar=(int)(p/ideal*20+0.5); if(bar>60)bar=60;
            char b[64]; int k=0; for(;k<bar&&k<63;k++)b[k]='#'; b[k]=0;
            fprintf(stderr,"  e%02d %6.2f%% %s\n",e,p,b); }
        /* Gini + how many experts hold 90% of the mass = collapse indicators */
        long *srt=malloc(E*sizeof(long)); memcpy(srt,agg,E*sizeof(long));
        for(int i=0;i<E;i++)for(int j=i+1;j<E;j++)if(srt[j]>srt[i]){long t=srt[i];srt[i]=srt[j];srt[j]=t;}
        long cum=0; int n90=0; for(;n90<E;){ cum+=srt[n90]; n90++; if(tot&&cum>=tot*9/10)break; }
        fprintf(stderr,"[router] dead experts: %d/%d | busiest %.2f%% | top %d experts carry 90%% of tokens\n",
                dead,E,maxp,n90);
        free(agg); free(srt); free(g_rhist); g_rhist=NULL;
    }
    double dt=wtime()-w0; if(dt<1e-9)dt=1e-9;
    prof_report(dt);
    if(show_stats){
        printf("\n------------------------------------------------------------\n");
        if(spec) printf("[gen] %d tok in %.2fs = %.0f tok/s | self-speculative: %.0f%% accept, %.2f tok/forward\n",
                        generated,dt,generated/dt, fwd_chunks?100.0*acc_sum/((double)fwd_chunks*c->n_mtp):0.0,
                        fwd_chunks?(double)generated/fwd_chunks:0.0);
        else printf("[gen] %d tok in %.2fs = %.0f tok/s (sliding KV, cache=%d, attn window=%s)\n",
                    generated,dt,generated/dt,c->T,
                    c->win>0 ? "trained win" : "full");
    } else printf("\n");
}

/* ----------------------------- gen (low-level) -------------------------- */
static int cmd_gen(int argc, char **argv){
    if(argc<4){ fprintf(stderr,
        "usage: tinylm gen <model.bin> \"<prompt>\" [n] [temp] [topk] [stop_eot]\n"
        "  (legacy: tinylm gen <model.bin> <tok.bin> \"<prompt>\" ... ; or use 'tinylm run')\n"); return 1; }
    char mg[4]={0}; FILE *mf=fopen(argv[2],"rb");
    if(!mf){ fprintf(stderr,"cannot open %s\n",argv[2]); return 1; }
    if(fread(mg,1,4,mf)!=4){ fclose(mf); fprintf(stderr,"bad model %s\n",argv[2]); return 1; }
    fclose(mf);
    int bundled = !memcmp(mg,"TLM2",4) || !memcmp(mg,"TLM3",4)
               || !memcmp(mg,"TLM4",4) || !memcmp(mg,"TLM5",4)
               || !memcmp(mg,"TLQ1",4) || !memcmp(mg,"TLQ2",4);
    Cfg c; float *P; Tok *tk; const char *prompt; int ai;
    if(bundled){
        P=model_load_full(argv[2],&c,&tk);
        if(!tk){ fprintf(stderr,"bundle has no tokenizer; re-pack it\n"); return 1; }
        prompt=argv[3]; ai=4;
    } else {
        if(argc<5){ fprintf(stderr,"legacy TLM1 model needs a tokenizer:\n"
            "  tinylm gen %s <tok.bin> \"<prompt>\"   (or: tinylm pack ...)\n",argv[2]); return 1; }
        P=model_load(argv[2],&c); tk=tok_load(argv[3],1); prompt=argv[4]; ai=5;
    }
    int nnew = argc>ai?atoi(argv[ai]):200;
    float temp= argc>ai+1?atof(argv[ai+1]):0.8f;
    int topk  = argc>ai+2?atoi(argv[ai+2]):40;
    int stop_eot = argc>ai+3?atoi(argv[ai+3]):1;
    int use_q8 = !memcmp(mg,"TLQ1",4) || !memcmp(mg,"TLQ2",4) || getenv("TINYLM_Q8")!=NULL;
    Weights w; map_weights(&c,P,&w); Gen *gn=gen_new(&c,&w,use_q8);
    /* every heavy tensor now has an int8 copy -- drop the fp32 block */
    float *kept = gn->q8 ? gen_drop_fp32(&c,&w,&P) : NULL; (void)kept;
    int nthreads=g_pool_nt;
    #ifdef _OPENMP
    nthreads=g_pool_nt>1?g_pool_nt:omp_get_max_threads();
    #endif
    printf("[c-tinylm gen] params=%zu threads=%d temp=%.2f topk=%d template=%s n_mtp=%d decode=%s\n",
           param_count(&c), nthreads, temp, topk, c.tmpl?"alpaca":"raw", c.n_mtp, use_q8?"int8":"fp32");
    printf("------------------------------------------------------------\n");
    run_generation(&c,&w,gn,tk, prompt, nnew, temp, topk, stop_eot, 1, 1);
    return 0;
}

/* ----------------------------- run (named) ------------------------------ */
/* ----------------------------- likelihood scoring ------------------------
 * `score` reports log P(continuation | context) -- the primitive every
 * multiple-choice benchmark is built on. HellaSwag, ARC, MMLU, PIQA and
 * friends are all "which of these endings does the model find least
 * surprising", which needs summed token log-probs, not generation.
 *
 * The request file is NUL-separated rather than JSON so there is no escaping
 * to get wrong and no JSON parser in here: ctx \0 cont \0 ctx \0 cont \0 ...
 * A NUL cannot occur inside UTF-8 text, so this is unambiguous.
 *
 * Output, one line per request:  <sum_logp> <n_cont_tokens> <n_greedy_match>
 * The caller decides between raw and length-normalised comparison (lm-eval
 * reports both as acc and acc_norm; they disagree often at small scale).
 * n_greedy_match counts continuation positions where the model's argmax IS the
 * actual token -- what LAMBADA-style exact-match needs. Comparing choices is
 * meaningless for a task with only one continuation.
 *
 * Context handling matches the standard harness: the pair is tokenised
 * TOGETHER and the continuation is the suffix after tokenising the context
 * alone. BPE is not prefix-consistent, so tokenising them separately and
 * concatenating would produce token sequences the model never saw. */
static int cmd_score(int argc, char **argv){
    if(argc<4){ fprintf(stderr,
        "usage: tinylm score <name|model.bin> <requests.bin> [out.txt]\n"
        "  requests.bin: NUL-separated pairs  ctx \\0 cont \\0 ctx \\0 cont \\0 ...\n"
        "  prints one line per pair: <sum log P(cont|ctx)> <n_cont_tokens>\n"); return 1; }
    const char *arg=argv[2];
    char path[1024]; model_path(arg, path, sizeof(path));
    FILE *e=fopen(path,"rb"); if(e) fclose(e); else snprintf(path,sizeof(path),"%s",arg);
    Cfg c; Tok *tk=NULL; float *P=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"'%s' has no bundled tokenizer\n",path); return 1; }
    int use_q8=getenv("TINYLM_Q8")!=NULL;
    { FILE *qf=fopen(path,"rb"); if(qf){ char mg[4]={0}; if(fread(mg,1,4,qf)==4 &&
        (!memcmp(mg,"TLQ1",4)||!memcmp(mg,"TLQ2",4)||!memcmp(mg,"TLQ3",4))) use_q8=1; fclose(qf); } }
    Weights w; map_weights(&c,P,&w); memset(&w.ng,0,sizeof(w.ng)); ngram_load(path,&c,&w.ng);
    Gen *gn=gen_new(&c,&w,use_q8);
    /* every heavy tensor now has an int8 copy -- drop the fp32 block */
    float *kept = gn->q8 ? gen_drop_fp32(&c,&w,&P) : NULL; (void)kept;
    rope_init(&c, c.T+8);

    size_t rlen; uint8_t *rbuf=read_file(argv[3],&rlen);
    if(!rbuf){ fprintf(stderr,"cannot open %s\n",argv[3]); return 1; }
    FILE *out = (argc>4) ? fopen(argv[4],"w") : stdout;
    if(!out){ fprintf(stderr,"cannot write %s\n",argv[4]); return 1; }

    int V=c.V; size_t off=0; long done=0; double t0=wtime();
    while(off<rlen){
        const char *ctx=(const char*)rbuf+off;
        size_t cl=strnlen(ctx, rlen-off); if(off+cl>=rlen) break;
        off += cl+1;
        const char *cont=(const char*)rbuf+off;
        size_t ol=strnlen(cont, rlen-off); if(off+ol>rlen) break;
        off += ol+1;

        char *both=malloc(cl+ol+1);
        memcpy(both,ctx,cl); memcpy(both+cl,cont,ol); both[cl+ol]='\0';
        int nc=0,nf=0;
        int *ic=tok_encode(tk,ctx,&nc);
        int *fi=tok_encode(tk,both,&nf);
        free(both); free(ic);

        /* left-truncate to the trained context, keeping the continuation */
        int start=0;
        if(nf>c.T){ start=nf-c.T; }
        int nctx=nc-start; if(nctx<1) nctx=1;      /* need one token to condition on */
        int m=nf-start;
        double lp=0.0; int cnt=0, nmatch=0;
        if(m>1 && nctx<m){
            forward_chunk(&c,&w,gn,fi+start,m,0);
            for(int i=nctx;i<m;i++){
                const float *lg=gn->logits+(size_t)(i-1)*V;
                float mx=-1e30f; int am=0;
                for(int v=0;v<V;v++) if(lg[v]>mx){ mx=lg[v]; am=v; }
                double s=0; for(int v=0;v<V;v++) s+=exp((double)lg[v]-mx);
                lp += ((double)lg[fi[start+i]]-mx) - log(s);
                if(am==fi[start+i]) nmatch++;
                cnt++;
            }
        }
        free(fi);
        fprintf(out,"%.6f %d %d\n",lp,cnt,nmatch);
        if(++done%200==0){ fprintf(stderr,"\r[score] %ld reqs, %.1f/s   ",
                                   done, done/(wtime()-t0+1e-9)); fflush(stderr); }
    }
    fprintf(stderr,"\r[score] %ld requests in %.1fs (%.1f/s)          \n",
            done, wtime()-t0, done/(wtime()-t0+1e-9));
    if(out!=stdout) fclose(out);
    free(rbuf);
    return 0;
}

static int cmd_run(int argc, char **argv){
    if(argc<3){ fprintf(stderr,
        "usage: tinylm run <name|model.bin> [\"prompt\"] [n] [temp] [topk]\n"
        "       tinylm run <name> -i        (interactive: type prompts, blank line quits)\n"
        "  <name> resolves to models/<name>.bin (else treated as a file path).\n"
        "  With no prompt (or -i), starts an interactive session.\n"); return 1; }
    const char *arg=argv[2];
    char path[1024]; model_path(arg, path, sizeof(path));
    FILE *e=fopen(path,"rb"); if(e) fclose(e); else snprintf(path,sizeof(path),"%s",arg);
    char mg[4]={0}; { FILE *qf=fopen(path,"rb"); if(qf){ if(fread(mg,1,4,qf)!=4) mg[0]=0; fclose(qf); } }
    int is_tlq3 = !memcmp(mg,"TLQ3",4);
    /* Phase 2B: TINYLM_Q4 makes the low-peak loader build true 4-bit QMats
     * (half the weight RAM) instead of int8. int8 remains the default. Only the
     * low-peak TLQ3 path honours it. */
    g_q4 = is_tlq3 && !getenv("TINYLM_HIGHPEAK") && getenv("TINYLM_Q4")!=NULL;
    Cfg c; Tok *tk=NULL; Weights w; Gen *gn; float *P=NULL, *kept=NULL;
    if(is_tlq3 && !getenv("TINYLM_HIGHPEAK")){
        /* low-peak: decode tensor-by-tensor straight into QMats (Phase 2A/2B) */
        gn=load_q4_lowpeak(path,&c,&w,&tk,&kept);
        memset(&w.ng,0,sizeof(w.ng)); ngram_load(path,&c,&w.ng);
    } else {
        P=model_load_full(path,&c,&tk);
        int use_q8=getenv("TINYLM_Q8")!=NULL;
        if(!memcmp(mg,"TLQ1",4)||!memcmp(mg,"TLQ2",4)||!memcmp(mg,"TLQ3",4)) use_q8=1;
        map_weights(&c,P,&w); memset(&w.ng,0,sizeof(w.ng)); ngram_load(path,&c,&w.ng);
        gn=gen_new(&c,&w,use_q8);
        /* every heavy tensor now has an int8 copy -- drop the fp32 block */
        kept = gn->q8 ? gen_drop_fp32(&c,&w,&P) : NULL;
    }
    (void)kept;
    if(!tk){ fprintf(stderr,"'%s' has no bundled tokenizer (untrained?).\n"
                            "  train it:  tinylm train %s <data_dir> <epochs>\n", path, arg); return 1; }

    int interactive=0; const char *prompt=NULL; int ai=3;
    if(argc>3 && (!strcmp(argv[3],"-i")||!strcmp(argv[3],"--interactive")||!strcmp(argv[3],"+interactive"))){
        interactive=1; ai=4;
    } else if(argc>3){ prompt=argv[3]; ai=4; }
    else interactive=1;
    int n   = argc>ai?atoi(argv[ai]):256;
    float temp= argc>ai+1?atof(argv[ai+1]):0.8f;
    int topk = argc>ai+2?atoi(argv[ai+2]):40;

    if(interactive){
        printf("[tinylm] %s | %s | %zu params | n_mtp=%d | temp=%.2f topk=%d\n",
               arg, c.tmpl?"alpaca/chat":"raw/continuation", param_count(&c), c.n_mtp, temp, topk);
        printf("Type an %s and press Enter. Blank line (or Ctrl-Z, Enter) to quit.\n",
               c.tmpl?"instruction":"prompt to continue");
        char line[16384];
        for(;;){
            printf("\n> "); fflush(stdout);
            if(!fgets(line,sizeof(line),stdin)) break;
            size_t L=strlen(line); while(L && (line[L-1]=='\n'||line[L-1]=='\r')) line[--L]=0;
            if(L==0) break;
            run_generation(&c,&w,gn,tk, line, n, temp, topk, 1, /*echo*/0, /*stats*/0);
        }
        printf("[bye]\n");
    } else {
        printf("[tinylm] %s | temp=%.2f topk=%d | decode=%s\n", arg, temp, topk,
               g_q4?"int4":(gn->q8?"int8":"fp32"));
        printf("------------------------------------------------------------\n");
        run_generation(&c,&w,gn,tk, prompt, n, temp, topk, 1, /*echo*/1, /*stats*/1);
    }
    return 0;
}

/* ----------------------------- chat ------------------------------------- */
/* Multi-turn chat with a PERSISTENT KV cache.
 *
 * `run -i` already loops on input, but it calls run_generation fresh every time,
 * so each turn starts from an empty context -- a prompt loop, not a
 * conversation. Here `pos` and the KV ring carry across turns, so the model
 * actually sees what was said before, and each turn costs only its own tokens
 * instead of re-prefilling the whole transcript.
 *
 * STOPPING is the part that matters most for this model. The chat markers are
 * plain text, not special tokens, so nothing prevents the model from emitting
 * `<|user|>` and then answering itself -- it does this constantly (visible all
 * through the benchmarks). So generation stops on the decoded TEXT, not on a
 * token id, and a short holdback keeps a partially-emitted marker from reaching
 * stdout before we know what it is. */
#define CHAT_HOLD  16          /* >= longest stop marker */
#define CHAT_RESP  65536

/* Scan only the freshly-written tail: called once per token, so a full rescan
 * of a long reply would be quadratic. */
static int chat_stop_at(const char *s, int len, int from){
    static const char *marks[] = {"<|user|>", "<|system|>", "<|assistant|>", NULL};
    if(from < 0) from = 0;
    for(int m=0; marks[m]; m++){
        int ml=(int)strlen(marks[m]);
        for(int i=from; i+ml<=len; i++)
            if(!memcmp(s+i, marks[m], ml)) return i;
    }
    return -1;
}

static int cmd_chat(int argc, char **argv){
    if(argc<3){ fprintf(stderr,
        "usage: tinylm chat <name|model.bin> [max_tokens] [temp] [topk]\n"
        "  Multi-turn chat; the conversation stays in context between turns.\n"
        "  Commands:  /reset  clear the conversation   /quit  exit\n"
        "  Defaults: max_tokens=256 temp=0.8 topk=40 (use temp 0 for greedy)\n"); return 1; }
    char path[1024]; model_path(argv[2], path, sizeof(path));
    { FILE *e=fopen(path,"rb"); if(e) fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]); }
    Cfg c; Tok *tk=NULL; float *P=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"'%s' has no bundled tokenizer.\n",path); return 1; }
    int use_q8=getenv("TINYLM_Q8")!=NULL;
    { FILE *qf=fopen(path,"rb"); if(qf){ char mg[4]={0}; if(fread(mg,1,4,qf)==4 &&
        (!memcmp(mg,"TLQ1",4)||!memcmp(mg,"TLQ2",4)||!memcmp(mg,"TLQ3",4))) use_q8=1; fclose(qf); } }
    Weights w; map_weights(&c,P,&w); memset(&w.ng,0,sizeof(w.ng)); ngram_load(path,&c,&w.ng);
    Gen *gn=gen_new(&c,&w,use_q8);
    /* every heavy tensor now has an int8 copy -- drop the fp32 block */
    float *kept = gn->q8 ? gen_drop_fp32(&c,&w,&P) : NULL; (void)kept;

    int nmax  = argc>3?atoi(argv[3]):256;
    float temp= argc>4?atof(argv[4]):0.8f;
    int topk  = argc>5?atoi(argv[5]):40;
    const char *rp_env=getenv("TINYLM_REP"); float rep_pen = rp_env?atof(rp_env):1.3f;
    const char *rw_env=getenv("TINYLM_REPWIN"); int rep_win = rw_env?atoi(rw_env):128;
    if(rep_pen<1.0f) rep_pen=1.0f;

    /* Absolute positions keep climbing across turns, so the RoPE tables must
     * cover more than one context. Past this budget the session is reset. */
    int ropemax = c.T*4;
    rope_init(&c, ropemax);

    int nthreads=g_pool_nt;
    #ifdef _OPENMP
    nthreads=g_pool_nt>1?g_pool_nt:omp_get_max_threads();
    #endif
    printf("[tinylm chat] %s | %zu params | %s | %s | threads=%d | temp=%.2f topk=%d\n",
           path, param_count(&c),
           c.tmpl==2?"chatml":(c.tmpl==1?"alpaca":"raw"),
           use_q8?"int8 decode":"fp32 decode", nthreads, temp, topk);
    if(c.win>0) printf("[tinylm chat] attention window %d over a %d cache\n", c.win, c.T);
    if(c.tmpl==0) printf("[tinylm chat] WARNING: this model has no chat template; "
                         "it will continue text rather than reply.\n");
    printf("Type a message. /reset clears context, /quit exits.\n");

    float *work=(float*)malloc(sizeof(float)*c.V);
    float *heap=(float*)malloc(sizeof(float)*(size_t)(topk>0?topk:1));
    int   *hist=(int*)malloc(sizeof(int)*(size_t)ropemax);
    int    hn=0;
    char  *resp=(char*)malloc(CHAT_RESP);
    char   line[16384], fmt[16640];
    int pos=0;

    for(;;){
        printf("\nyou> "); fflush(stdout);
        if(!fgets(line,sizeof(line),stdin)) break;
        size_t L=strlen(line); while(L && (line[L-1]=='\n'||line[L-1]=='\r')) line[--L]=0;
        if(L==0) continue;
        if(!strcmp(line,"/quit")||!strcmp(line,"/exit")) break;
        if(!strcmp(line,"/reset")){ pos=0; hn=0; printf("[context cleared]\n"); continue; }

        /* Position budget: a rough token estimate is enough, the reset is cheap. */
        if(pos + (int)(L/2) + nmax + 32 > ropemax){
            pos=0; hn=0; printf("[context full - starting fresh]\n");
        }
        if(c.tmpl==2)      snprintf(fmt,sizeof(fmt),"<|user|>\n%s\n<|assistant|>\n",line);
        else if(c.tmpl==1) snprintf(fmt,sizeof(fmt),"### Instruction:\n%s\n\n### Response:\n",line);
        else               snprintf(fmt,sizeof(fmt),"%s",line);

        int pn; int *pids=tok_encode(tk,fmt,&pn);
        float *lg_cur=NULL;
        for(int i=0;i<pn;){ int m=pn-i; if(m>c.T)m=c.T;
            forward_chunk(&c,&w,gn,pids+i,m,pos);
            lg_cur=gn->logits+(size_t)(m-1)*c.V; pos+=m; i+=m; }
        for(int i=0;i<pn && hn<ropemax;i++) hist[hn++]=pids[i];
        free(pids);
        if(!lg_cur) continue;

        printf("bot> "); fflush(stdout);
        int rlen=0, printed=0, gen=0, cut=-1;
        double t0=wtime();
        while(gen<nmax){
            int nxt=sample_logits(&c, lg_cur, work, heap, temp, topk, hist, hn, rep_pen, rep_win);
            if(nxt==tk->eot) break;
            if(hn<ropemax) hist[hn++]=nxt;
            int dl=tk->declen[nxt];
            if(rlen+dl >= CHAT_RESP) break;
            memcpy(resp+rlen, tk->dec[nxt], dl); rlen+=dl; gen++;
            cut = chat_stop_at(resp, rlen, rlen-dl-CHAT_HOLD);
            if(cut>=0) break;                       /* model started a new turn */
            int safe = rlen-CHAT_HOLD;
            if(safe>printed){ fwrite(resp+printed,1,safe-printed,stdout); printed=safe; fflush(stdout); }
            forward_chunk(&c,&w,gn,&nxt,1,pos); pos++; lg_cur=gn->logits;
        }
        int endp = (cut>=0)?cut:rlen;               /* drop the marker and anything after */
        if(endp>printed) fwrite(resp+printed,1,endp-printed,stdout);
        double dt=wtime()-t0; if(dt<1e-9)dt=1e-9;
        printf("\n      [%d tok, %.1f tok/s, ctx %d/%d]\n", gen, gen/dt, pos, ropemax);
        fflush(stdout);
    }
    printf("\n[bye]\n");
    free(work); free(heap); free(hist); free(resp);
    return 0;
}

/* ----------------------------- quantize --------------------------------- */
/* Shrink a checkpoint on disk. Weights are stored fp16 (~2x) or q8 block
 * (~3.8x); loading dequantizes back to fp32, so run-time compute is unchanged
 * (this is checkpoint compression, not lower-precision compute). Keeps the
 * bundled tokenizer/template so the quantized file still runs standalone. */
static int cmd_quantize(int argc, char **argv){
    if(argc<4){ fprintf(stderr,
        "usage: tinylm quantize <in_model.bin> <out_model.bin> [fp16|q8|q4]\n"
        "  fp16  half precision, ~2x smaller, near-lossless (default)\n"
        "  q8    8-bit blocks (group 64), ~3.8x smaller, slightly lossy\n"
        "  q4    4-bit blocks (group 128, symmetric [-7,7]), ~7.5x smaller (matches QAT)\n"
        "  Loading dequantizes to fp32 -> compute/quality at run time is that of the\n"
        "  stored precision; the file just takes less disk. Tokenizer is preserved.\n"); return 1; }
    const char *in=argv[2], *out=argv[3];
    int qtype=0, group=64;
    if(argc>4){
        if(!strcmp(argv[4],"q8")||!strcmp(argv[4],"int8")) qtype=1;
        else if(!strcmp(argv[4],"q4")||!strcmp(argv[4],"int4")){ qtype=2; group=128; }
    }
    size_t flen; uint8_t *buf=read_file(in,&flen);
    if(!buf){ fprintf(stderr,"cannot open %s\n",in); return 1; }
    Rd r={buf,buf+flen}; char m[4]; rd(&r,m,4);
    if(!memcmp(m,"TLQ1",4)||!memcmp(m,"TLQ2",4)||!memcmp(m,"TLQ3",4)){ fprintf(stderr,"%s is already quantized\n",in); free(buf); return 1; }
    int ver, nh=hdr_ints(m,&ver);
    if(!nh){ fprintf(stderr,"%s: not a tinylm model\n",in); free(buf); return 1; }
    int hdr[HDR_MAX]; if(!rd(&r,hdr,4*nh)){ fprintf(stderr,"bad header\n"); free(buf); return 1; }
    Cfg c; cfg_from_hdr(&c,hdr,ver);
    size_t n=param_count(&c); float *P=fz(n);
    if(!rd(&r,P,sizeof(float)*n)){ fprintf(stderr,"bad params\n"); free(buf); return 1; }
    int tl=0; uint8_t *blob=NULL;
    if(ver>=2 && rd(&r,&tl,4) && tl>0) blob=r.p;      /* points into buf (still alive) */

    /* q4 uses selective spans (stored qtype=3): only linear projections get 4-bit,
       sensitive gaps stay fp16, matching the QAT layout. */
    int wtype=qtype; size_t qlen; uint8_t *q;
    if(qtype==2){ q=q4_selective_encode(P,n,&c,group,&qlen); wtype=3; }
    else q=quantize_params(P,n,qtype,group,&qlen);
    /* measure the quantization error (dequantize back and compare) */
    float *P2=fz(n); Rd rq={q,q+qlen};
    if(wtype==3) q4_selective_decode(&rq,P2,n,&c,group);
    else dequantize_params(&rq,P2,n,qtype,group);
    double maxe=0,sume=0,sumx=0;
    for(size_t i=0;i<n;i++){ double e=fabs((double)P[i]-(double)P2[i]);
        if(e>maxe)maxe=e; sume+=e; sumx+=fabs((double)P[i]); }

    FILE *f=fopen(out,"wb"); if(!f){ fprintf(stderr,"cannot write %s\n",out); return 1; }
    /* Selective q4 (wtype==3) is stamped TLQ3 (qver6, 15 ints) so the magic
     * matches the content -- the low-peak/int4 run paths gate on the magic.
     * Otherwise the TLM4/TLM5 split: TLQ2 when there is a window, else TLQ1. */
    if(wtype==3){
        int oh[15]={c.V,c.D,c.L,c.H,c.KV,c.F,c.T,c.eot,c.tmpl,c.n_mtp,c.n_exp,
                    c.win,c.full_every,c.n_shared,c.qknorm};
        fwrite("TLQ3",1,4,f); fwrite(oh,4,15,f);
    } else if(c.win>0){
        int oh[13]={c.V,c.D,c.L,c.H,c.KV,c.F,c.T,c.eot,c.tmpl,c.n_mtp,c.n_exp,c.win,c.full_every};
        fwrite("TLQ2",1,4,f); fwrite(oh,4,13,f);
    } else {
        int oh[11]={c.V,c.D,c.L,c.H,c.KV,c.F,c.T,c.eot,c.tmpl,c.n_mtp,c.n_exp};
        fwrite("TLQ1",1,4,f); fwrite(oh,4,11,f);
    }
    fwrite(&wtype,4,1,f); fwrite(&group,4,1,f);
    fwrite(q,1,qlen,f);
    fwrite(&tl,4,1,f); if(tl>0&&blob) fwrite(blob,1,tl,f);
    fclose(f);

    int hbytes=(wtype==3?15:c.win>0?13:11)*4;
    double insz=(double)flen, outsz=4+hbytes+8+(double)qlen+4+tl;
    printf("[quantize] %s -> %s  (%s, %zu params)\n", in, out, qtype==2?"q4":qtype==1?"q8":"fp16", n);
    printf("[quantize] %.2f MB -> %.2f MB  (%.2fx smaller) | max|err| %.2e, mean-rel %.2e | tok %s\n",
           insz/1e6, outsz/1e6, insz/outsz, maxe, sumx>0?sume/sumx:0.0, tl>0?"kept":"none");
    printf("[quantize] run it:  tinylm run %s   (loads = dequantized fp32)\n", out);
    free(P); free(P2); free(q); free(buf);
    return 0;
}

/* ----------------------------- pack ------------------------------------- */
/* bundle an existing checkpoint + tokenizer + template into one self-contained file */
/* Dump token ids for a text file, using the tokenizer bundled in the model.
 * Exists so other tools can work at the id level without reimplementing the
 * BPE: the JAX/TPU port needs the exact same ids tinylm produces, and an
 * independent tokenizer reimplementation is a silent-divergence risk. */
static int cmd_tokenize(int argc, char **argv){
    if(argc<4){ fprintf(stderr,
        "usage: tinylm tokenize <name|model.bin> <in.txt> [out.bin]\n"
        "  writes uint16 token ids; with no out.bin, prints them as text\n"); return 1; }
    char path[1024]; model_path(argv[2], path, sizeof(path));
    FILE *e=fopen(path,"rb"); if(e) fclose(e); else snprintf(path,sizeof(path),"%s",argv[2]);
    Cfg c; Tok *tk=NULL; float *P=model_load_full(path,&c,&tk);
    if(!tk){ fprintf(stderr,"'%s' has no bundled tokenizer\n",path); return 1; }
    free(P);
    size_t tl; uint8_t *txt=read_file(argv[3],&tl);
    if(!txt){ fprintf(stderr,"cannot read %s\n",argv[3]); return 1; }
    char *s=malloc(tl+1); memcpy(s,txt,tl); s[tl]='\0';
    int n=0; int *ids=tok_encode(tk,s,&n);
    if(argc>4){
        FILE *f=fopen(argv[4],"wb");
        if(!f){ fprintf(stderr,"cannot write %s\n",argv[4]); return 1; }
        for(int i=0;i<n;i++){ uint16_t u=(uint16_t)ids[i]; fwrite(&u,2,1,f); }
        fclose(f);
        fprintf(stderr,"[tokenize] %d tokens -> %s\n",n,argv[4]);
    } else {
        for(int i=0;i<n;i++) printf("%d%s", ids[i], i+1<n?" ":"\n");
    }
    free(ids); free(s); free(txt);
    return 0;
}

static int cmd_pack(int argc, char **argv){
    if(argc<6){ fprintf(stderr,"usage: tinylm pack <in_model.bin> <tok.bin> <raw|alpaca> <out.bin>\n"); return 1; }
    Cfg c; Tok *ignore=NULL; float *P=model_load_full(argv[2],&c,&ignore);
    c.tmpl = !strcmp(argv[4],"alpaca") ? 1 : 0;
    size_t tl; uint8_t *blob=read_file(argv[3],&tl);
    if(!blob){ fprintf(stderr,"cannot read %s\n",argv[3]); return 1; }
    model_save_bundle(argv[5], &c, P, param_count(&c), blob, (int)tl);
    printf("[pack] wrote %s  (weights + tokenizer + template=%s)\n", argv[5], c.tmpl?"alpaca":"raw");
    printf("[pack] now just: tinylm gen %s \"<prompt>\"\n", argv[5]);
    return 0;
}

/* ----------------------------- help ------------------------------------- */
static void print_help(void){
    printf(
"tinylm - a 0.1M-1M parameter language model in multithreaded C\n"
"================================================================================\n"
"A dependency-free port of the PyTorch tinylm. Decoder-only mini-Llama:\n"
"RMSNorm + RoPE + grouped-query attention + SwiGLU + tied embeddings. The forward\n"
"pass, the full backward pass, and AdamW are hand-written; the hot loops use\n"
"OpenMP. Runtime needs only libc + libgomp.\n"
"\n"
"USAGE\n"
"  tinylm <command> [args...]\n"
"  tinylm help | -h | --help            show this page\n"
"\n"
"COMMANDS  (the three you need)\n"
"--------------------------------------------------------------------------------\n"
"  new   <name> <preset>\n"
"      Create a model named <name> from a preset (see PRESETS). Writes\n"
"      models/<name>.bin (untrained). No MTP heads by default (TINYLM_NMTP=2 to add).\n"
"\n"
"  train <name> <data_dir> [epochs]\n"
"      Train models/<name>.bin on a dataset for [epochs] passes (default 1;\n"
"      fractional ok). The model adopts the data's tokenizer + chat template on\n"
"      first train. Auto-resumes (epochs is the TOTAL target) and checkpoints as\n"
"      it goes - Ctrl-C is safe; re-run to continue. If <name> is itself a preset\n"
"      and no such model exists, it is created automatically.\n"
"        data_dir  folder with train.bin, val.bin, tokenizer.bin (a prepare_* run)\n"
"\n"
"  run   <name> [\"prompt\"]            one-shot generation\n"
"  run   <name> -i                    interactive session (type prompts; blank quits)\n"
"      Loads models/<name>.bin (or a file path). Alpaca models wrap your input in\n"
"      the instruction template automatically; raw models continue your text.\n"
"      Optional trailing [n] [temp] [topk]. Greedy + MTP -> self-speculative.\n"
"\n"
"  low-level: gen <model.bin> \"<prompt>\" [n] [temp] [topk] [stop_eot]   (file paths;\n"
"             also legacy 'gen <model> <tok.bin> ...');  pack <in> <tok.bin> <raw|alpaca> <out>\n"
"  chat <name|model.bin> [ntok] [temp] [topk]  multi-turn chat; context persists\n"
"             between turns (unlike `run -i`, which restarts each time).\n"
"             /reset clears the conversation, /quit exits.\n"
"  quantize <in.bin> <out.bin> [fp16|q8]   shrink a checkpoint on disk (fp16 ~2x,\n"
"             q8 ~3.8x); loading dequantizes to fp32 so run-time quality = stored\n"
"             precision. Keeps the tokenizer; the output still runs standalone.\n"
"\n"
"PRESETS (V is taken from the data's tokenizer; params shown at vocab 2048)\n"
"--------------------------------------------------------------------------------\n"
"  name   params   d_model  layers  heads  kv  d_ff   ctx     (CPU train speed)\n"
"  nano   ~0.2M      64       4      4     2   176    256     fastest\n"
"  micro  ~0.6M      96       4      6     2   256    256\n"
"  m1     ~0.9M     128       4      4     1   336    256     recommended small\n"
"  m3     ~2.8M     192       6      6     2   512    256     slow on CPU\n"
"  m6     ~6.1M     256       8      8     2   688    256     very slow on CPU\n"
"  m8     ~7.6M     288       8      8     2   768    256     very slow on CPU\n"
"  m16    ~15M      512       5      8     2  1408    256     hours/epoch (use BLAS)\n"
"  m32    ~31M      512      10      8     2  1536    256     ~half a day/epoch\n"
"  m70    ~69M      768      11     12     2  2048    256     days/epoch (research only)\n"
"  (m3+ take hours-to-days/epoch on this laptop. Use 'tinylm bench <preset>' to\n"
"   measure throughput first; the OpenBLAS build helps most on m16+.)\n"
"\n"
"ENVIRONMENT\n"
"  OMP_NUM_THREADS   TRAINING threads (default = all logical cores). On a 4-core\n"
"                    /8-thread CPU, 6-8 are all within a few %%; dense GEMM slightly\n"
"                    prefers 6 (less hyperthread contention), MoE slightly prefers\n"
"                    8. Set OPENBLAS_NUM_THREADS to match for the BLAS build.\n"
"  TINYLM_THREADS    DECODE threads. Generation uses its own spin-wait pool, not\n"
"                    OpenMP: at one token per forward the regions are too small\n"
"                    for libgomp's fork/join, which measured a NET LOSS (1 thread\n"
"                    beat 8). Default is one fewer than the logical cores; falls\n"
"                    back to OMP_NUM_THREADS if that is set. 1 disables the pool.\n"
"  TINYLM_KVSPLIT    split the key range so single-token attention has more than\n"
"                    KV jobs to hand out (default on with the pool, off without).\n"
"                    0 forces the unsplit path, which is the bit-exact one.\n"
"  TINYLM_NMTP       MTP/Medusa heads on a FRESH model (default 0 = off). They\n"
"                    cost ~30%% of training at V=2048 (each adds 3 vocab-sized\n"
"                    GEMMs in backward) and only help self-speculative decoding.\n"
"                    Set 2 to re-enable. Read by 'new'/'train' (fresh only).\n"
"  TINYLM_BATCH      override the auto-selected training batch size.\n"
"  TINYLM_PRINT_SEC  seconds between cheap progress lines (loss/tok-s/ETA; def 15).\n"
"  TINYLM_VAL_SEC    seconds between val evals (4 batches, def 600).\n"
"  TINYLM_SAVE_SEC   seconds between checkpoints (def 300). All time-based, so the\n"
"                    cadence is steady even when one step takes minutes.\n"
"\n"
"DIAGNOSTICS\n"
"  tinylm selftest   verify the register-blocked GEMM kernels vs a naive reference\n"
"\n"
"FILE FORMATS\n"
"  train.bin/val.bin  raw uint16 token stream (little-endian).\n"
"  tokenizer.bin      'TLTK' decode table + byte map + merge ranks.\n"
"  template.txt       'alpaca' or 'raw' (read at train time, baked into ckpt).\n"
"  models/<name>.bin  'TLM3' header (V,D,L,H,KV,F,T,eot,template,n_mtp) + fp32\n"
"                     params + embedded tokenizer. Self-contained. (Reads older\n"
"                     'TLM2'=no n_mtp and 'TLM1'=no tokenizer too.)\n"
"  models/<name>.bin.opt  'TOPT' iter + AdamW m,v (only used to resume training).\n"
"\n"
"EXAMPLES\n"
"--------------------------------------------------------------------------------\n"
"  set OMP_NUM_THREADS=8\n"
"  tinylm new  mychat m3                       # create a ~3M model\n"
"  tinylm train mychat ../data/alpaca 2        # 2 epochs (Ctrl-C safe; re-run to resume)\n"
"  tinylm run  mychat -i                       # chat with it interactively\n"
"  tinylm run  mychat \"Give three tips for staying healthy.\"   # one-shot\n"
"\n"
"  # shortcut: training a name that IS a preset auto-creates it\n"
"  tinylm train m1 ../data/alpaca 1\n"
"\n"
"NOTES\n"
"  - dropout is off (deterministic). Generation uses a sliding KV cache and, for\n"
"    MTP models decoded greedily, self-speculative decoding (exact, faster only\n"
"    when the heads draft well - i.e. on a well-trained model).\n"
"  - LR schedule: linear warmup (200 iters) then cosine decay to 0.1*lr over the\n"
"    epoch budget. m3/m6/m8 are slow on CPU - expect hours per epoch.\n"
"================================================================================\n");
}

/* ----------------------------- bench ------------------------------------ */
/* Peak working set (RAM) via kernel32 — no extra link libs needed.
 * Struct matches PROCESS_MEMORY_COUNTERS under MinGW x64 (LLP64: long=4, size_t=8). */
typedef struct { unsigned long cb, PageFaultCount;
    size_t Peak, Cur, q0,q1,q2,q3,pf0,pf1; } ProcMem_;
__declspec(dllimport) void* __stdcall GetCurrentProcess(void);
__declspec(dllimport) int   __stdcall K32GetProcessMemoryInfo(void*, void*, unsigned long);
static double peak_ram_mb(void){
    ProcMem_ pm; pm.cb=sizeof(pm);
    if(K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof(pm)))
        return (double)pm.Peak/(1024.0*1024.0);
    return 0.0;
}

/* Time real forward+backward+AdamW steps on random tokens — measures training
 * throughput for any preset without a dataset or waiting for the iter-50 report. */
static int cmd_bench(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: tinylm bench <preset> [iters] [batch] [experts]\n"
        "  times forward+backward+AdamW on random tokens; reports tok/s, s/step,\n"
        "  est. hours per 13M-token epoch, and peak RAM. experts>=2 -> top-1 MoE.\n"); return 1; }
    Cfg c; if(!preset(argv[2],&c)){ fprintf(stderr,"unknown preset '%s'\n",argv[2]); return 1; }
    int iters = argc>3?atoi(argv[3]):8; if(iters<3)iters=3;
    int B     = argc>4?atoi(argv[4]):32; if(B<1)B=1;
    c.n_mtp=default_nmtp(); c.n_exp = argc>5?atoi(argv[5]):0; if(c.n_exp==1)c.n_exp=0;
    cfg_derive(&c);
    rope_init(&c, c.T);
    size_t np=param_count(&c);
    float *P=fz(np),*G=fz(np),*M=fz(np),*Vv=fz(np);
    Weights w,gr; map_weights(&c,P,&w); map_weights(&c,G,&gr); memset(&w.ng,0,sizeof(w.ng)); memset(&gr.ng,0,sizeof(gr.ng));
    init_params(&c,&w);
    Acts *a=acts_new(&c,B);
    int N=B*c.T; int *tok=malloc(sizeof(int)*N),*tgt=malloc(sizeof(int)*N);
    for(int i=0;i<N;i++){ tok[i]=(int)(xorshift()%c.V); tgt[i]=(int)(xorshift()%c.V); }
    int **tgt_mtp = c.n_mtp>0 ? malloc(sizeof(int*)*c.n_mtp) : NULL;
    for(int k=0;k<c.n_mtp;k++){ tgt_mtp[k]=malloc(sizeof(int)*N);
        for(int i=0;i<N;i++) tgt_mtp[k][i]=(int)(xorshift()%c.V); }
    int nthreads=g_pool_nt;
    #ifdef _OPENMP
    nthreads=g_pool_nt>1?g_pool_nt:omp_get_max_threads();
    #endif
    printf("[bench] %s params=%.2fM D=%d L=%d H=%d KV=%d F=%d T=%d B=%d threads=%d n_mtp=%d n_exp=%d\n",
        argv[2], np/1e6, c.D,c.L,c.H,c.KV,c.F,c.T,B,nthreads,c.n_mtp,c.n_exp);
    fflush(stdout);
    int warm = iters>=5 ? 2 : 1;
    double t0=0;
    for(int it=0; it<iters; it++){
        if(it==warm) t0=wtime();
        forward(&c,&w,a,tok,tgt);
        memset(G,0,sizeof(float)*np);
        backward(&c,&w,&gr,a,tok,tgt,tgt_mtp);
        float gnm=global_gradnorm(G,np); float gs=gnm>1.0f?1.0f/gnm:1.0f;
        adamw(P,G,M,Vv,np,3e-4f,0.9f,0.95f,0.1f,it+1,gs);
        printf("  step %d/%d%s   \r", it+1, iters, it<warm?" (warmup)":""); fflush(stdout);
    }
    double dt=wtime()-t0; int timed=iters-warm; if(dt<1e-9)dt=1e-9;
    double tps=(double)N*timed/dt;
    printf("\n[bench] %d timed steps | %.2fs | %.0f tok/s | %.2f s/step | ~%.1f h / 13M-tok epoch | peak RAM %.0f MB\n",
        timed, dt, tps, dt/timed, 13.0e6/tps/3600.0, peak_ram_mb());
    return 0;
}

/* ----------------------------- selftest --------------------------------- */
/* Verify the three GEMM primitives against naive triple-loop references on
 * random matrices with non-multiple-of-4 dims (exercises remainder paths). */
static int cmd_selftest(int argc, char **argv){
    (void)argc;(void)argv;
    int M=201, K=131, N=259;          /* deliberately not multiples of 4 */
    float *X=fz((size_t)M*K), *W=fz((size_t)K*N), *dY=fz((size_t)M*N);
    for(size_t i=0;i<(size_t)M*K;i++) X[i]=rnd_normal();
    for(size_t i=0;i<(size_t)K*N;i++) W[i]=rnd_normal();
    for(size_t i=0;i<(size_t)M*N;i++) dY[i]=rnd_normal();
    float *Y=fz((size_t)M*N), *Yr=fz((size_t)M*N);
    float *dX=fz((size_t)M*K), *dXr=fz((size_t)M*K);
    float *dW=fz((size_t)K*N), *dWr=fz((size_t)K*N);
    /* references */
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)X[(size_t)m*K+k]*W[(size_t)k*N+n]; Yr[(size_t)m*N+n]=(float)s; }
    for(int m=0;m<M;m++) for(int k=0;k<K;k++){ double s=0; for(int n=0;n<N;n++) s+=(double)dY[(size_t)m*N+n]*W[(size_t)k*N+n]; dXr[(size_t)m*K+k]=(float)s; }
    for(int k=0;k<K;k++) for(int n=0;n<N;n++){ double s=0; for(int m=0;m<M;m++) s+=(double)X[(size_t)m*K+k]*dY[(size_t)m*N+n]; dWr[(size_t)k*N+n]=(float)s; }
    /* kernels under test (mm_bt/mm_atb accumulate, so start from zero) */
    mm(Y, X, W, M, K, N);
    mm_bt(dX, dY, W, M, K, N);
    mm_atb(dW, X, dY, M, K, N);
    double e1=0,e2=0,e3=0, s1=0,s2=0,s3=0;
    for(size_t i=0;i<(size_t)M*N;i++){ double d=fabs(Y[i]-Yr[i]); if(d>e1)e1=d; s1+=fabs(Yr[i]); }
    for(size_t i=0;i<(size_t)M*K;i++){ double d=fabs(dX[i]-dXr[i]); if(d>e2)e2=d; s2+=fabs(dXr[i]); }
    for(size_t i=0;i<(size_t)K*N;i++){ double d=fabs(dW[i]-dWr[i]); if(d>e3)e3=d; s3+=fabs(dWr[i]); }
    printf("selftest GEMM (M=%d K=%d N=%d)  build=%s\n", M,K,N,
#ifdef USE_BLAS
        "BLAS"
#else
        "portable-rblk"
#endif
    );
    printf("  mm     max|err|=%.2e  rel=%.2e\n", e1, e1/(s1/((size_t)M*N)));
    printf("  mm_bt  max|err|=%.2e  rel=%.2e\n", e2, e2/(s2/((size_t)M*K)));
    printf("  mm_atb max|err|=%.2e  rel=%.2e\n", e3, e3/(s3/((size_t)K*N)));
    int ok = (e1<1e-2)&&(e2<1e-2)&&(e3<1e-2);
    printf("  %s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}

/* Flush denormals to zero, on EVERY thread.
 *
 * The softmax computes exp(s - max); for positions far from the peak that
 * underflows into denormal floats, and denormal arithmetic traps to microcode at
 * hundreds of cycles per operation. Measured here before this was set: expf cost
 * 535 ns/call inside attention against 61.5 ns/call for the identical call count
 * in isolation -- 8.7x, and 75% of all attention time. -ffast-math is supposed
 * to arrange this via crtfastmath.o, but MinGW does not reliably link it, and
 * MXCSR is per-thread so the OpenMP workers would each need it anyway. */
static void set_ftz(void){
#if defined(__SSE2__) || defined(__x86_64__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}
int main(int argc, char **argv){
    { const char *sd=getenv("TINYLM_SEED");   /* pin the batch order so two runs
           see identical data -- required for any A/B to mean anything */
      if(sd) g_rng = (uint64_t)strtoull(sd,NULL,10) | 1ULL;
      else   g_rng ^= (uint64_t)time(NULL); }
    set_ftz();
#ifdef _OPENMP
    /* Default to PHYSICAL cores, not logical. Decode is FMA- and L1-bound, so
     * the two SMT siblings on a core contend for the same ports and the same L1
     * rather than adding throughput. Measured on this 4c/8t i5-1135G7 at short
     * context: 139 tok/s on 4 threads against 119 on 8. An explicit
     * OMP_NUM_THREADS still wins -- this only changes the unset default. */
    if(!getenv("OMP_NUM_THREADS")){
        int lg=omp_get_max_threads();
        if(lg>1) omp_set_num_threads(lg/2);
    }
    #pragma omp parallel
    { set_ftz(); }                 /* MXCSR is per-thread; the workers need it too */
#endif
    if(argc<2 || !strcmp(argv[1],"help") || !strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")){
        print_help(); return argc<2?1:0;
    }
    if(!strcmp(argv[1],"selftest")) return cmd_selftest(argc,argv);
    if(!strcmp(argv[1],"bench"))    return cmd_bench(argc,argv);
    if(!strcmp(argv[1],"new"))   return cmd_new(argc,argv);
    if(!strcmp(argv[1],"train")) return cmd_train(argc,argv);
    if(!strcmp(argv[1],"run"))   return cmd_run(argc,argv);
    if(!strcmp(argv[1],"gen"))   return cmd_gen(argc,argv);
    if(!strcmp(argv[1],"pack"))  return cmd_pack(argc,argv);
    if(!strcmp(argv[1],"chat"))     return cmd_chat(argc,argv);
    if(!strcmp(argv[1],"quantize")) return cmd_quantize(argc,argv);
    if(!strcmp(argv[1],"score"))    return cmd_score(argc,argv);
    if(!strcmp(argv[1],"tokenize")) return cmd_tokenize(argc,argv);
    fprintf(stderr,"unknown command '%s'\n\n", argv[1]);
    print_help(); return 1;
}
