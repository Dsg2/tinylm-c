"""Write the C-side tokenizer.bin from a trained HF tokenizer.json.

Shared by prepare_data.py / prepare_alpaca.py (so data prep is one step) and by
c/export_tokenizer.py (to upgrade an existing data dir). See c/tinylm.c for the
'TLTK' format consumed by the C trainer/generator.
"""
import json
import os
import struct

EOT = "<|endoftext|>"


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def write_tokenizer_bin(vdir):
    """Read vdir/{tokenizer.json,meta.json}, write vdir/tokenizer.bin. Returns its path."""
    spec = json.load(open(os.path.join(vdir, "tokenizer.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(vdir, "meta.json")))
    vocab = spec["model"]["vocab"]
    merges_raw = spec["model"]["merges"]
    V = len(vocab)
    eot_id = meta["eot_id"]

    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    id2str = [None] * V
    for s, i in vocab.items():
        id2str[i] = s

    def to_raw_bytes(s):
        if s == EOT:
            return b""
        out = bytearray()
        for ch in s:
            out.append(u2b[ch]) if ch in u2b else out.extend(ch.encode("utf-8"))
        return bytes(out)

    base = [vocab.get(b2u[b], -1) for b in range(256)]

    merges = []
    for m in merges_raw:
        a, b = (m.split(" ", 1) if isinstance(m, str) else (m[0], m[1]))
        ia, ib, inew = vocab.get(a), vocab.get(b), vocab.get(a + b)
        if None not in (ia, ib, inew):
            merges.append((ia, ib, inew))

    out_path = os.path.join(vdir, "tokenizer.bin")
    with open(out_path, "wb") as f:
        f.write(b"TLTK")
        f.write(struct.pack("<iii", V, eot_id, len(merges)))
        for i in range(V):
            raw = to_raw_bytes(id2str[i])
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)
        f.write(struct.pack("<256i", *base))
        for a, b, n in merges:
            f.write(struct.pack("<iii", a, b, n))
    print(f"[tokbin] {out_path}  V={V} eot={eot_id} merges={len(merges)}")
    return out_path
