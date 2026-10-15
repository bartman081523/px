import sys
from collections import Counter

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")

from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType, GGUFValueType  # noqa: E402

path = sys.argv[1]
out = open(sys.argv[2], "w", encoding="utf-8")

r = GGUFReader(path)
fields = r.fields

out.write(f"TENSOR_COUNT={len(r.tensors)} KV_COUNT={len(fields)}\n")

v2s = GGUFValueType._value2member_map_
t2s = GGMLQuantizationType._value2member_map_

# --- full KV dump ---
out.write("\n=== KV (full) ===\n")
for name, field in fields.items():
    t = field.types[0] if field.types else "?"
    tn = v2s.get(t, "?")
    if t in (GGUFValueType.ARRAY,):
        elem = field.types[1] if len(field.types) > 1 else "?"
        en = v2s.get(elem, "?")
        n = len(field.data)
        out.write(f"{name} = ARRAY<{en}> len={n}\n")
    else:
        c = field.contents()
        out.write(f"{name} = {tn} {c!r}\n")

# --- tensor table ---
out.write("\n=== TENSORS (full) ===\n")
ttype_hist = Counter()
for t in r.tensors:
    tt = int(t.tensor_type)
    ttype_hist[tt] += 1
    off = getattr(t, "data_offset", getattr(t, "offset", -1))
    out.write(f"{t.name} type={tt} {t2s[tt].name if tt in t2s else 'UNK'} shape={list(t.shape)} offset={off}\n")

out.write("\n=== TYPE HISTOGRAM ===\n")
for tt, n in sorted(ttype_hist.items()):
    nm = t2s[tt].name if tt in t2s else "UNK"
    out.write(f"type {tt} ({nm}): {n}\n")

out.close()

# --- stdout summary ---
print(f"tensor_count={len(r.tensors)} kv_count={len(fields)}")
print("--- KV summary ---")
for name, field in fields.items():
    t = field.types[0] if field.types else "?"
    tn = v2s.get(t, "?")
    if t == GGUFValueType.ARRAY:
        elem = field.types[1] if len(field.types) > 1 else "?"
        print(f"{name} = ARRAY<{v2s.get(elem,'?')}> len={len(field.data)}")
    else:
        print(f"{name} = {tn} {field.contents()!r}")
print("--- type histogram ---")
for tt, n in sorted(ttype_hist.items()):
    nm = t2s[tt].name if tt in t2s else "UNK"
    print(f"  {nm}: {n}")
print("--- notable tensors ---")
for t in r.tensors:
    nm = t.name
    if ("token_embd" in nm or "output" in nm or "position" in nm or "conv" in nm
            or nm.startswith("blk.0.") or nm.startswith("blk.1.") or "gdn" in nm or "lin" in nm.split(".")[-2:]):
        print(f"  {nm} type={int(t.tensor_type)} {t2s.get(int(t.tensor_type),'?').name} shape={list(t.shape)}")