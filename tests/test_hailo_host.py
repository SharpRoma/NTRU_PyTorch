"""Exact host-kernel comparisons; never hardware confirmation."""
from dataclasses import replace

import numpy as np
import pytest

from ntru_pytorch import PRESETS, Parameters, PublicKey, RandomSource
from ntru_pytorch.hailo.scheme import BinaryScheme


@pytest.fixture(params=[17, 107, 701])
def scheme(request):
    p = Parameters(17, 3, 256, 3, 2, 1) if request.param == 17 else (
        PRESETS['moderate'] if request.param == 107 else PRESETS['dead'])
    return BinaryScheme(PublicKey(p, tuple((19*i+7) % p.q for i in range(p.N))))


def test_grouped_reconstruction(scheme):
    s = scheme
    source = RandomSource(42)
    probes = [(0,)*s.n, (1,)*s.n, (-1,)*s.n,
              tuple((-1)**i for i in range(s.n)), source.ternary(s.n, 1, 1)]
    for count in (15, 16, 17):
        for sign in (1, -1):
            probes.append(tuple(sign if i in set(np.linspace(0, s.n-1, count, dtype=int))
                                else 0 for i in range(s.n)))
    for mode in ('active16-v1', 'position16-v1'):
        # Bound memory for the dense N=701 case.
        for r in probes:
            batch = s.pack_batch([r], mode)
            out = s.software(batch.frames)
            ref = s.reconstruct_batch(out, batch)
            opt = s.reconstruct_batch(out, batch, host_processing='optimized')
            assert np.array_equal(ref, opt)
            linear = np.convolve(np.asarray(r, dtype=np.int64),
                                 np.asarray(s.key.h, dtype=np.int64) % s.key.params.q)
            expected = linear[:s.n].copy()
            expected[:s.n-1] += linear[s.n:]
            assert np.array_equal(opt[0], expected)
    batch = s.pack_batch(probes, 'active16-v1')
    out = s.software(batch.frames)
    for permutation in (np.arange(len(out)), source.rng.sample(range(len(out)), len(out))):
        mixed = replace(batch, frames=batch.frames[permutation], owners=batch.owners[permutation],
                        signs=batch.signs[permutation])
        assert np.array_equal(s.reconstruct_batch(out[permutation], mixed),
                              s.reconstruct_batch(out[permutation], mixed,
                                                  host_processing='optimized'))
    for bad in (replace(batch, signs=np.zeros_like(batch.signs)),
                replace(batch, owners=np.full_like(batch.owners, batch.count))):
        with pytest.raises(ValueError):
            s.reconstruct_batch(out, bad, host_processing='optimized')
    for bad in (out[:-1], out.astype(float), np.full_like(out, 17)):
        with pytest.raises(ValueError):
            s.reconstruct_batch(bad, batch, host_processing='optimized')


def test_all_compact_codes():
    from ntru_pytorch.hailo import ArtifactError
    from ntru_pytorch.hailo.proof import (
        affine_lookup,
        compact_affine_lookup,
        decode_affine_compact,
        decode_affine_lookup,
    )
    for scale, zero in [('1/8', '0'), ('8421505/67108864', '0'), ('1/3', '127/2')]:
        exact = affine_lookup({'output': {'scale': scale, 'zero_point': zero}})
        compact = compact_affine_lookup(exact)
        assert compact.dtype == np.uint8 and not compact.flags.writeable
        for code in range(256):
            raw = np.array([code], dtype=np.uint8)
            if 0 <= exact[code] <= 16:
                assert np.array_equal(decode_affine_lookup(raw, exact),
                                      decode_affine_compact(raw, compact))
            else:
                with pytest.raises(ArtifactError, match='outside'):
                    decode_affine_compact(raw, compact)
        with pytest.raises(ArtifactError, match='UINT8'):
            decode_affine_compact(np.array([1.0]), compact)


def test_optimized_packing(scheme):
    s = scheme
    rng = RandomSource(42)
    probes = [(0,)*s.n, (1,)*s.n, (-1,)*s.n,
              tuple((-1)**i for i in range(s.n))]
    for count in (1,15,16,17):
        for sign in (1,-1):
            probes.append(tuple(sign if i < count else 0 for i in range(s.n)))
    probes += [rng.ternary(s.n,s.key.params.d,s.key.params.d) for _ in range(20)]
    for mode in ('position16-v1','active16-v1'):
        ref = s.pack_batch(probes,mode)
        opt = s.pack_batch(probes,mode,host_processing='optimized')
        for name in ('frames','owners','signs'):
            assert np.array_equal(getattr(ref,name),getattr(opt,name))
            assert not getattr(opt,name).flags.writeable
        s.validate_frames(opt.frames)
    for value in ([[True]], [[1.0]], [[2]], [[2**80]], [[np.bool_(True)]], [],
                  np.array([[True]*s.n]), np.full((1,s.n),2**63,dtype=np.uint64)):
        with pytest.raises(ValueError):
            s.pack_batch(value,'active16-v1',host_processing='optimized')
    for value in ([[1,-1]], [[np.int64(1),np.int64(-1)]], (iter([1,-1]),)):
        ref = s.pack_batch([[1,-1]],'active16-v1')
        opt = s.pack_batch(value,'active16-v1',host_processing='optimized')
        assert np.array_equal(ref.frames,opt.frames)


def test_encoder_and_api(scheme):
    from ntru_pytorch.codec import encode
    from ntru_pytorch.hailo.engine import SoftwareHailoEncryptor
    from ntru_pytorch.hailo.host import encode_array

    s = scheme
    sizes = {0, 1, max(0,s.n//8-8), max(0,s.n//8-7), s.n//8+1}
    messages = [b'', 'Привет 😀\0'.encode(), bytes(range(256))]
    messages += [bytes(i%256 for i in range(size)) for size in sorted(sizes)]
    for message in messages:
        assert tuple(map(tuple,encode_array(message,s.n))) == encode(message,s.n)
    ref = SoftwareHailoEncryptor(s.key,packing='active16-v1')
    opt = SoftwareHailoEncryptor(s.key,packing='active16-v1',host_processing='optimized')
    for message in messages:
        rs = RandomSource(42)
        a,b = [],[]
        expected = ref.encrypt_bytes(s.key,message,random_source=RandomSource(42),
                                     trace=lambda *event, sink=a:sink.append(event))
        actual = opt.encrypt_bytes(s.key,message,random_source=rs,
                                    trace=lambda *event, sink=b:sink.append(event))
        assert actual == expected and a == b
        assert opt.encrypt_bytes(s.key,message,random_source=RandomSource(42)) == expected
        timings = opt.last_timings
        assert timings['unaccounted_ns'] >= 0
        assert timings['encrypt_total_ns'] == sum(timings[k] for k in (
            'message_encode_ns','key_normalization_ns','prepare_ns','inference_with_io_ns',
            'reconstruct_ns','ciphertext_construction_ns','unaccounted_ns'))
    for engine in (ref,opt):
        for bad in ([True],[1.0],[2],[2**80],[1]*(s.n+1)):
            with pytest.raises(ValueError):
                engine.encrypt_poly(s.key,bad,r=[0])
            with pytest.raises(ValueError):
                engine.encrypt_poly(s.key,[0],r=bad)
        with pytest.raises(ValueError):
            engine.encrypt_polys(s.key,[[0]],r_polys=[])
        with pytest.raises(TypeError):
            engine.encrypt_bytes(s.key,'text')
    with pytest.raises(ValueError):
        SoftwareHailoEncryptor(s.key,host_processing='unknown')


def test_maps_with_missing_owners_and_repeated_sign_groups(scheme):
    from ntru_pytorch.hailo.scheme import PackedBatch
    s = scheme
    out = np.zeros((5,1,1,s.cout),dtype=np.uint8)
    out[:,:,:,:s.n*s.bits] = np.arange(5,dtype=np.uint8)[:,None,None,None]
    batch = PackedBatch(np.zeros((5,1,1,s.cin),dtype=np.uint8),
                        np.array([0,0,0,2,2]),np.array([1,-1,1,-1,1]),4,'active16-v1')
    assert np.array_equal(s.reconstruct_batch(out,batch),
                          s.reconstruct_batch(out,batch,host_processing='optimized'))
    if s.cout > s.n*s.bits:
        out[0,0,0,-1] = 1
        with pytest.raises(ValueError,match='padded'):
            s.reconstruct_batch(out,batch,host_processing='optimized')


def test_hardware_failure_diagnostics(tmp_path):
    from types import SimpleNamespace

    from ntru_pytorch.hailo import ArtifactError
    from ntru_pytorch.hailo.validation import _hardware_mismatch
    with pytest.raises(ArtifactError,match='seed=42, input=7'):
        _hardware_mismatch(SimpleNamespace(diagnostic_directory=tmp_path),'partial_sums',42,7,
                           [[1]],np.array([1]),np.array([2]))
    with np.load(tmp_path/'mismatch-partial_sums-7.npz') as data:
        assert data['expected'].tolist()==[1] and data['actual'].tolist()==[2]


@pytest.mark.parametrize('fault', ['ciphertext', 'plaintext', 'decrypt_exception'])
def test_qualification_failure_capture(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace

    from ntru_pytorch import Ciphertext, PrivateKey, SympyNTRU
    from ntru_pytorch.hailo import engine, validation
    from ntru_pytorch.hailo.engine import SoftwareHailoEncryptor

    p = Parameters(17, 3, 256, 3, 2, 1)
    f = (1,)+(0,)*16
    private = PrivateKey(p, f, f, f, (1,-1)+(0,)*15)
    key = PublicKey(p,(3,-3)+(0,)*15)

    class FakeDevice(SoftwareHailoEncryptor):
        def __init__(self, directory, **kwargs):
            super().__init__(key,packing=kwargs['packing'],host_processing=kwargs['host_processing'])
            self.session = SimpleNamespace(identity={'device_id':'TEST-ONLY-NOT-HARDWARE'})
            self.manifest = {'compiled':{'files':{'model.hef':'TEST-ONLY'}}}

        def encrypt_bytes(self, *args, **kwargs):
            result = super().encrypt_bytes(*args, **kwargs)
            if fault=='ciphertext':
                blocks = [list(row) for row in result.blocks]
                blocks[0][0] += 1
                return Ciphertext(p,blocks)
            return result

    monkeypatch.setattr(engine,'HailoEncryptor',FakeDevice)
    monkeypatch.setattr(validation,'validate_hardware',lambda *a,**kw:{'stage':'TEST-ONLY'})
    if fault=='plaintext':
        monkeypatch.setattr(SympyNTRU,'decrypt_bytes',lambda *a,**kw:b'invalid')
    if fault=='decrypt_exception':
        def fail(*args, **kwargs):
            raise ValueError('TEST-ONLY decryption failure')
        monkeypatch.setattr(SympyNTRU,'decrypt_bytes',fail)
    with pytest.raises(ValueError):
        validation.qualify_hardware(tmp_path,private,host_processing='optimized',
                                    diagnostic_directory=tmp_path)
    files = list(tmp_path.glob('*.npz'))
    assert len(files)==1
    with np.load(files[0]) as data:
        assert data['seed']==42 and data['index']==0
        assert 'expected' in data
        assert ('actual' in data) if fault!='decrypt_exception' else ('ciphertext' in data)
