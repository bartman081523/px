#!/usr/bin/env python
"""Komprimiert einen px-metrics-Snapshot (JSON-Datei vom Metrics-Endpoint) auf eine Zeile."""
import json
import sys

d = json.load(open(sys.argv[1]))
trace = d.get("telemetry_trace") or []
sig = d.get("cognitive_signature") or {}
zw = d.get("zone_weights") or {}
out = {
    "phi": d.get("phi"),
    "phi_decode_mean": d.get("phi_decode_mean"),
    "steps": d.get("steps"),
    "path_len": len(d.get("path") or []),
    "path_head": (d.get("path") or [])[:3],
    "zone": d.get("zone"),
    "active": d.get("active"),
    "decode_tokens": d.get("decode_tokens"),
    "trace_len": len(trace),
    "trace_first": trace[0] if trace else None,
    "trace_last": trace[-1] if trace else None,
    "sig": {k: sig.get(k) for k in ("kurtosis", "phi", "zone", "loops_run", "focus_index", "gamma")},
    "zw": {k: round(v, 3) for k, v in zw.items()} if all(isinstance(v, (int, float)) for v in zw.values()) else zw,
    "aks": (d.get("aks_profile") or {}).get("correction_strength"),
    "em": (d.get("subjective_metrics") or {}).get("emancipation"),
    "ent": d.get("entropy"),
}
print(json.dumps(out, ensure_ascii=False))