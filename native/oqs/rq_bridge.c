#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include "poly.h"
#define API __attribute__((visibility("default")))
typedef struct { poly h; } nr_rq;
API int nr_rq_abi(void) { return 1; }
API int nr_rq_n(void) { return NTRU_N; }
API int nr_rq_q(void) { return NTRU_Q; }
API size_t nr_rq_prepared_bytes(void) { return sizeof(nr_rq); }
API const char *nr_rq_implementation(void) { return NR_IMPLEMENTATION; }
static void *alloc_aligned(size_t bytes) {
    void *p=NULL;
    if (posix_memalign(&p,32,bytes)) return NULL;
    memset(p,0,bytes);
    return p;
}
API void *nr_rq_prepare(const uint16_t *h, size_t n) {
    if (!h || n!=NTRU_N) return NULL;
    nr_rq *ctx=alloc_aligned(sizeof(nr_rq));
    if (!ctx) return NULL;
    for (size_t i=0;i<n;i++) {
        if (h[i]>=NTRU_Q) { free(ctx); return NULL; }
        ctx->h.coeffs[i]=h[i];
    }
    return ctx;
}
API void nr_rq_free(void *ctx) { free(ctx); }
API int nr_rq_batch(void *handle, const uint16_t *inputs, size_t count, uint16_t *outputs, size_t batch) {
    nr_rq *ctx=handle;
    if (!ctx || !inputs || !outputs || !batch || batch>SIZE_MAX/NTRU_N || count!=batch*NTRU_N) return -1;
    poly r={0},out={0};
    for (size_t b=0;b<batch;b++) {
        for (size_t i=0;i<NTRU_N;i++) {
            if (inputs[b*NTRU_N+i]>=NTRU_Q) return -1;
            r.coeffs[i]=inputs[b*NTRU_N+i];
        }
        NR_RQ_MUL(&out,&ctx->h,&r);
        for (size_t i=0;i<NTRU_N;i++) outputs[b*NTRU_N+i]=out.coeffs[i]&(NTRU_Q-1);
    }
    return 0;
}
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW,&t)) return 0;
    return (uint64_t)t.tv_sec*1000000000ULL+(uint64_t)t.tv_nsec;
}
/* Construction/copy of input polys is outside the native timed interval. */
API int nr_rq_window(void *handle, const uint16_t *inputs, size_t count, uint16_t *outputs, size_t batch, size_t iterations, uint64_t *elapsed, uint64_t *checksum) {
    nr_rq *ctx=handle;
    if (!ctx || !inputs || !outputs || !batch || !iterations || !elapsed || !checksum || batch>SIZE_MAX/sizeof(poly) || batch>SIZE_MAX/NTRU_N || count!=batch*NTRU_N) return -1;
    poly *r=alloc_aligned(batch*sizeof(poly));
    if (!r) return -1;
    for (size_t b=0;b<batch;b++) for (size_t i=0;i<NTRU_N;i++) {
        if (inputs[b*NTRU_N+i]>=NTRU_Q) { free(r); return -1; }
        r[b].coeffs[i]=inputs[b*NTRU_N+i];
    }
    poly out={0};
    volatile uint64_t sum=0;
    uint64_t start=now_ns();
    if (!start) { free(r); return -1; }
    for (size_t j=0;j<iterations;j++) for (size_t b=0;b<batch;b++) {
        NR_RQ_MUL(&out,&ctx->h,&r[b]);
        for (size_t i=0;i<NTRU_N;i++) outputs[b*NTRU_N+i]=out.coeffs[i]&(NTRU_Q-1);
        sum+=outputs[b*NTRU_N+(j%NTRU_N)];
    }
    *elapsed=now_ns()-start; *checksum=sum;
    free(r);
    return 0;
}
