"""Preflight: Konverter-XF-Zweige byte-exakt gg. official-GGUF-Tensoren."""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
sys.path.insert(0, "ququint_quant")
import convert_gf3_to_gguf as cv  # noqa: E402
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(cv.OFFICIAL)
gf3h = cv.st_header(cv.GF3)
gf3h.pop("__metadata__", None)


def tbytes(name):
    t = next(t for t in r.tensors if t.name == name)
    return (np.frombuffer(t.data, dtype=np.uint8).copy(),
            [int(s) for s in t.shape], int(t.tensor_type))


def tern_test(name):
    bo, ne, _ = tbytes(name)
    basis, _ = cv.resolve(name)
    src = cv.gdn_row_src(ne[1])
    gq3v = gf3h[basis + ".gq3"]
    gs3v = gf3h[basis + ".gs3"]
    gq3_mm = cv.st_memmap(cv.GF3, gq3v, "<i4")
    gs3_mm = cv.st_memmap(cv.GF3, gs3v, "<f2")
    m = ne[1]
    parts = []
    for lo in range(0, m, 128):
        rows = src[lo:lo + 128]
        gq3 = np.ascontiguousarray(gq3_mm[:, rows])
        gs3 = np.ascontiguousarray(gs3_mm[:, rows])
        blk = cv.encode_blocks(cv.gf3_digits(gq3), gs3)
        parts.append(blk.reshape(-1).tobytes())
    got = b"".join(parts)
    ok = got == bo.tobytes()
    print(f"{name:30s} tern_perm      identisch: {ok}")
    return ok


def small_test(name):
    bo, ne, _ = tbytes(name)
    basis, kind = cv.resolve(name)
    val = gf3h[basis]
    buf = cv.st_bytes(cv.GF3, val)
    xf = cv.SUFFIX_TRANSFORMS.get(cv.suffix_of(name))
    if xf == "rows_48":
        u32 = np.frombuffer(buf, dtype="<u4")
        out = (u32 >> np.uint32(16)).astype("<u2") \
            .reshape(-1, ne[0])[cv.TBL_GDN]
        got = np.ascontiguousarray(out).tobytes()
    elif xf == "conv_perm":
        m2 = np.frombuffer(buf, dtype="<f4").reshape(ne[1], ne[0])
        got = np.ascontiguousarray(m2[cv.gdn_row_src(ne[1])]).tobytes()
    elif xf == "elem_48":
        got = np.frombuffer(buf, dtype="<f4")[cv.TBL_GDN].tobytes()
    elif xf == "ssm_a":
        v = np.frombuffer(buf, dtype="<f4")
        got = (-np.exp(v[cv.TBL_GDN])).tobytes()
    elif xf == "plus1":
        v = np.frombuffer(buf, dtype="<f4")
        got = (v + np.float32(1.0)).tobytes()
    else:
        raise AssertionError(xf)
    ok = got == bo.tobytes()
    print(f"{name:30s} {xf:12s} identisch: {ok}")
    return ok


def main():
    oks = []
    for L in (0, 32, 62):
        oks.append(tern_test(f"blk.{L}.attn_gate.weight"))
        oks.append(tern_test(f"blk.{L}.attn_qkv.weight"))
        oks.append(small_test(f"blk.{L}.ssm_conv1d.weight"))
        oks.append(small_test(f"blk.{L}.ssm_alpha.weight"))
        oks.append(small_test(f"blk.{L}.ssm_beta.weight"))
        oks.append(small_test(f"blk.{L}.ssm_dt.bias"))
        oks.append(small_test(f"blk.{L}.ssm_a"))
        oks.append(small_test(f"blk.{L}.attn_norm.weight"))
        oks.append(small_test(f"blk.{L}.post_attention_norm.weight"))
    oks.append(small_test("output_norm.weight"))
    print(f"\nGESAMT: {sum(oks)}/{len(oks)} byte-exakt")


if __name__ == "__main__":
    main()