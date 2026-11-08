#define _GNU_SOURCE
#include <oqs/oqs.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#define API __attribute__((visibility("default")))
API int nr_abi(void) { return 1; }
API const char *nr_oqs_version(void) { return OQS_version(); }
API const char *nr_oqs_path(void) {
    Dl_info info;
    return dladdr((void *)OQS_version, &info) ? info.dli_fname : "";
}
API const char *nr_implementation(void) { return NR_IMPLEMENTATION; }
API void *nr_kem_new(void) {
    OQS_init();
    if (!OQS_KEM_alg_is_enabled("NTRU-HRSS-701")) return NULL;
    return OQS_KEM_new("NTRU-HRSS-701");
}
API void nr_kem_free(void *handle) { OQS_KEM_free(handle); }
API int nr_kem_sizes(void *handle, size_t *out) {
    OQS_KEM *k = handle;
    if (!k || !out) return -1;
    out[0]=k->length_public_key; out[1]=k->length_secret_key;
    out[2]=k->length_ciphertext; out[3]=k->length_shared_secret;
    return 0;
}
API void nr_cleanse(void *p, size_t n) { if (p) OQS_MEM_cleanse(p,n); }
API int nr_keypair(void *handle, uint8_t *pk, size_t np, uint8_t *sk, size_t ns) {
    OQS_KEM *k=handle;
    if (!k || !pk || !sk || np!=k->length_public_key || ns!=k->length_secret_key) return -1;
    return OQS_KEM_keypair(k,pk,sk);
}
API int nr_encaps(void *handle, uint8_t *ct, size_t nc, uint8_t *ss, size_t nz, const uint8_t *pk, size_t np) {
    OQS_KEM *k=handle;
    if (!k || !ct || !ss || !pk || nc!=k->length_ciphertext || nz!=k->length_shared_secret || np!=k->length_public_key) return -1;
    return OQS_KEM_encaps(k,ct,ss,pk);
}
API int nr_decaps(void *handle, uint8_t *ss, size_t nz, const uint8_t *ct, size_t nc, const uint8_t *sk, size_t ns) {
    OQS_KEM *k=handle;
    if (!k || !ct || !ss || !sk || nc!=k->length_ciphertext || nz!=k->length_shared_secret || ns!=k->length_secret_key) return -1;
    return OQS_KEM_decaps(k,ss,ct,sk);
}
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW,&t)) return 0;
    return (uint64_t)t.tv_sec*1000000000ULL+(uint64_t)t.tv_nsec;
}
/* Buffers are supplied/allocated before timing. RNG remains the upstream system RNG. */
API int nr_kem_window(void *handle, int op, size_t iterations, uint8_t *pk, uint8_t *sk, uint8_t *ct, uint8_t *ss, uint64_t *elapsed, uint64_t *checksum) {
    OQS_KEM *k=handle;
    if (!k || !iterations || !pk || !sk || !ct || !ss || !elapsed || !checksum || op<0 || op>2) return -1;
    volatile uint64_t sum=0;
    uint64_t start=now_ns();
    if (!start) return -1;
    for (size_t i=0;i<iterations;i++) {
        OQS_STATUS rc;
        if (op==0) { rc=OQS_KEM_keypair(k,pk,sk); sum+=pk[i%k->length_public_key]; }
        else if (op==1) { rc=OQS_KEM_encaps(k,ct,ss,pk); sum+=ct[i%k->length_ciphertext]; }
        else { rc=OQS_KEM_decaps(k,ss,ct,sk); sum+=ss[i%k->length_shared_secret]; }
        if (rc!=OQS_SUCCESS) return -1;
    }
    *elapsed=now_ns()-start; *checksum=sum;
    return 0;
}
