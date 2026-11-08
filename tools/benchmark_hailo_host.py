"""CPU-only diagnostic microbenchmarks. These never establish NPU execution."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from ntru_pytorch import RandomSource
from ntru_pytorch.codec import encode
from ntru_pytorch.formats import load
from ntru_pytorch.hailo.engine import SoftwareHailoEncryptor
from ntru_pytorch.hailo.host import encode_array
from ntru_pytorch.hailo.proof import (
    affine_lookup,
    compact_affine_lookup,
    decode_affine_compact,
    decode_affine_lookup,
)
from ntru_pytorch.hailo.scheme import BinaryScheme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise ValueError('Preserve previous measurements')
    root = Path(args.artifact)
    key = load(root/'public.json')
    scheme = BinaryScheme(key)
    source = RandomSource(42)
    polynomials = tuple(source.ternary(key.params.N,key.params.d,key.params.d) for _ in range(128))
    batch = scheme.pack_batch(polynomials,'active16-v1')
    # Real software partial sums, prepared outside each phase timer.
    partials = scheme.software(batch.frames)
    spec = json.loads((root/'compiler-report.json').read_text())['affine']
    lut = affine_lookup(spec)
    compact = compact_affine_lookup(lut)
    valid = np.flatnonzero((lut>=0)&(lut<=16)).astype(np.uint8)
    codes = np.random.default_rng(42).choice(valid,size=partials.shape)
    message = bytes(i%256 for i in range(128*key.params.N//8-8))
    samples = []

    def measure(stage,mode,fn):
        result = None
        for iteration in range(23):
            start = perf_counter_ns()
            result = fn()
            elapsed = perf_counter_ns()-start
            if iteration>=3:
                samples.append(dict(stage=stage,mode=mode,iteration=iteration-3,ns=elapsed))
        return result

    for mode in ('reference','optimized'):
        measure('packing',mode,lambda mode=mode:scheme.pack_batch(polynomials,'active16-v1',host_processing=mode))
        product = measure('reconstruction',mode,lambda mode=mode:scheme.reconstruct_batch(
            partials,batch,host_processing=mode))
        if mode=='reference':
            reference = product
        else:
            assert np.array_equal(reference,product)
        measure('affine_decode',mode,lambda mode=mode:decode_affine_lookup(codes,lut) if mode=='reference'
                else decode_affine_compact(codes,compact))
        measure('message_encode',mode,lambda mode=mode:encode(message,key.params.N) if mode=='reference'
                else encode_array(message,key.params.N))
        engine = SoftwareHailoEncryptor(key,packing='active16-v1',host_processing=mode)
        engine._infer = lambda frames:partials
        ciphertext = measure('api_with_precomputed_partials',mode,lambda engine=engine:engine.encrypt_bytes(
            key,message,r_polys=polynomials))
        if mode=='reference':
            expected = ciphertext
        else:
            assert ciphertext==expected
    summary = [{ 'stage':stage,'mode':mode,
                 'median_ns':median(s['ns'] for s in samples if s['stage']==stage and s['mode']==mode)}
               for stage in dict.fromkeys(s['stage'] for s in samples)
               for mode in ('reference','optimized')]
    target.write_text(json.dumps({'hardware_tested':False,'blocks':128,'seed':42,
                                  'samples':samples,'summary':summary},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
