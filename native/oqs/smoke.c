#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern void *nr_kem_new(void);
extern void nr_kem_free(void *);
extern int nr_kem_sizes(void *,size_t *);
extern int nr_keypair(void *,uint8_t *,size_t,uint8_t *,size_t);
extern int nr_encaps(void *,uint8_t *,size_t,uint8_t *,size_t,const uint8_t *,size_t);
extern int nr_decaps(void *,uint8_t *,size_t,const uint8_t *,size_t,const uint8_t *,size_t);
extern void *nr_rq_prepare(const uint16_t *,size_t);
extern void nr_rq_free(void *);
extern int nr_rq_batch(void *,const uint16_t *,size_t,uint16_t *,size_t);
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"smoke failure line %d\n",__LINE__); exit(2); } } while(0)
int main(void) {
 void *k=nr_kem_new(); size_t n[4]; CHECK(k); CHECK(!nr_kem_sizes(k,n));
 uint8_t *pk=calloc(1,n[0]),*sk=calloc(1,n[1]),*ct=calloc(1,n[2]),*ss=calloc(1,n[3]),*got=calloc(1,n[3]);
 CHECK(pk && sk && ct && ss && got);
 for (size_t i=0;i<1000;i++) {
  CHECK(!nr_keypair(k,pk,n[0],sk,n[1]));
  CHECK(nr_keypair(k,pk,n[0]-1,sk,n[1])!=0);
  CHECK(!nr_encaps(k,ct,n[2],ss,n[3],pk,n[0]));
  CHECK(!nr_decaps(k,got,n[3],ct,n[2],sk,n[1])); CHECK(!memcmp(ss,got,n[3]));
 }
 nr_kem_free(k); free(pk);free(sk);free(ct);free(ss);free(got);
 uint16_t h[701]={1},r[701]={0},out[701]={0};r[700]=8191;
 void *a=nr_rq_prepare(h,701); CHECK(a);
 CHECK(!nr_rq_batch(a,r,701,out,1));CHECK(out[700]==8191);
 CHECK(nr_rq_batch(a,r,700,out,1)!=0);nr_rq_free(a);
 puts("bridge smoke: 1000 KEM round-trips and Rq checks passed");return 0;
}
