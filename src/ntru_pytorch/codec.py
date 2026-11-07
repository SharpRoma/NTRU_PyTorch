def encode(data, n):
    if not isinstance(data, bytes):
        raise TypeError("Expected bytes")
    payload = len(data).to_bytes(8, "big") + data
    bits = tuple((b >> i) & 1 for b in payload for i in range(7, -1, -1))
    bits += (0,) * ((-len(bits)) % n)
    return tuple(bits[i : i + n] for i in range(0, len(bits), n))


def decode(blocks):
    bits = tuple(v for b in blocks for v in b)
    if any(v not in (0, 1) for v in bits):
        raise ValueError("Decryption produced nonbinary coefficients")
    if len(bits) < 64:
        raise ValueError("Missing length header")
    raw = bytes(sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, len(bits) - 7, 8))
    size = int.from_bytes(raw[:8], "big")
    end = 64 + size * 8
    if end > len(bits) or any(bits[end:]):
        raise ValueError("Invalid length or padding")
    return raw[8 : 8 + size]


class MessageAPI:
    def encrypt_bytes(self, key, data, *, random_source=None, r_polys=None, trace=None):
        from .models import Ciphertext
        from .randomness import RandomSource

        blocks = encode(data, key.params.N)
        rs = RandomSource() if random_source is None else random_source
        supplied = None if r_polys is None else tuple(r_polys)
        if supplied is not None and len(supplied) != len(blocks):
            raise ValueError("One r per block required")
        return Ciphertext(
            key.params,
            tuple(
                self.encrypt_poly(
                    key,
                    b,
                    r=supplied[i]
                    if supplied is not None
                    else rs.ternary(key.params.N, key.params.d, key.params.d),
                    trace=trace,
                )
                for i, b in enumerate(blocks)
            ),
        )

    def decrypt_bytes(self, key, cipher, *, trace=None):
        if (key.params.N, key.params.p, key.params.q) != (
            cipher.params.N,
            cipher.params.p,
            cipher.params.q,
        ):
            raise ValueError("Parameter mismatch")
        return decode(tuple(self.decrypt_poly(key, b, trace=trace) for b in cipher.blocks))

    def encrypt_text(self, key, text, **kwargs):
        return self.encrypt_bytes(key, text.encode("utf-8"), **kwargs)

    def decrypt_text(self, key, cipher, **kwargs):
        return self.decrypt_bytes(key, cipher, **kwargs).decode("utf-8")
