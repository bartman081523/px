"""G5 Phase-2 (Plan hf-space-v4-publish): ZeroGPU-Chat-Test via gradio_client.

T1: gemma3-270m BASELINE — Parität/Contract: app + chat_fn-Lease + Generation.
T2: ternary-bonsai-27b ACTIVE_MANIFOLD_RELAY, relay_layer=34 — der Kern:
    privates Repo (6 GB GF3, Space-Secret HF_TOKEN) + GF3-Build +
    triton-JIT + Manifold-Load (models---Fallback, C1) + Relay (dwidth-
    Fallback, L34) + Generierung — alles innerhalb EINER Lease.
    Lease-Realität 2026-10-08 (A-Grad, KORRIGIERT): Request = duration×1.5.
    **Owner-Quota FALSIFIZIERT**: der Token (Free-Account) hat ~262 s
    ZeroGPU pro rollierendem 24-h-Fenster; BILLING = volle Request-Dauer
    (Reservierung), nicht nur tatsächliche GPU-Sekunden. Messreihe:
    T1 request 60 ✓ (261→202?), T2 request 165 → "GPU task aborted"
    (Lease-Tod im Load), Retry → "165s requested vs. 55s left. Try again
    in 21:50:32" → T2 erst nach Quota-Reset (Plan: set_lease_secret.py
    174 → Request 261 ≤ frisches 262, oder T1 40 + T2 130 → 60+195=255).
    Startup-Prefetch hat den vollständigen Snapshot lease-frei gezogen
    (Snap ac4233470a41c6866a2b34c45f5ea28a1786bf24) — Load-Kette im
    Lease bewiesen komplett (GF3-Build 26.896B/5.92 GiB, Manifold via
    models---Fallback, patched successfully); G5-Befund: relay inactive
    (DWIDTH-Lade-Site ohne Portable-Fallback) → Fix 8d61e11d, unit-geprüft.

Format-Hinweis (v2-Fix): job.result() von /bot_response = FINAL-GEYIELDER
History (Liste von gradio-6-Messages: dict mit content als str ODER
text-block-Liste, teils auch plain str) — NICHT (history, session)-Tupel
mit res[0]-Unwrap. Parser ist jetzt format-tolerant (v1 crashte an
Message-Dicts, deren Keys iteriert wurden).
Auswahl (v3-Fix): argv[1] in {t1, t2, both} — T1/T2 einzeln leasbar,
um die Free-Quota in Fenster zu splitten.

Erwartete Fehlertexte bei Quota-Problemen (spaces/zero/client.py):
  "too many ZeroGPU credits allocated", "Space app has reached its GPU
  limit", "exceeded your ZeroGPU runs limit".

Run: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
     scratches/hfspace/g5_client_test.py
"""
import sys
import traceback

URL = "https://neuralworm-px-explorer-v4.hf.space"

GEN_KW = dict(
    px_preset="BASELINE", temp=0.7, tp=0.95, mt=64, rp=1.15, gamma=0.08,
    thinking=False, thinking_budget=0, thinking_effort="xhigh",
    relay_sign=1, relay_alpha=0.3, relay_layer=14,
    system_profile="neutral", system_prompt_text=None, session_id=None,
)


def user_turn(text):
    """gradio-6-Messages-Format (History-Input der /bot_response-Signatur)."""
    return dict(role="user", metadata=None, options=None,
                content=[{"text": text, "type": "text"}])


def as_history(res):
    """job.result() -> Liste von Messages (tolerant gegenüber Wrap-Formen)."""
    if isinstance(res, tuple) and res and not isinstance(res[0], (list, tuple)):
        # (history, ...) → erstes nicht-list-Element ist selbst die Message
        return [res[0]] if isinstance(res[0], (dict, str)) else list(res)
    if isinstance(res, tuple) and res:
        return list(res[0]) if isinstance(res[0], (list, tuple)) else list(res)
    if isinstance(res, (list, list)):
        return res
    if isinstance(res, (dict, str)):
        return [res]
    return [str(res)]


def extract(history):
    """[(role, text)] tolerant: content als str / Block-Liste / plain str."""
    msgs = []
    for m in history or []:
        if isinstance(m, dict):
            role = m.get("role", "")
            blocks = m.get("content")
            if isinstance(blocks, str):
                txt = blocks
            elif isinstance(blocks, (list, tuple)):
                txt = "".join(b.get("text", "") for b in blocks
                              if isinstance(b, dict))
            else:
                txt = ""
        elif isinstance(m, str):
            role, txt = "", m
        else:
            role, txt = "", str(m)
        msgs.append((role, txt))
    return msgs


def fmt(msgs):
    return "\n".join(f"[{r or '?'}] {t}" for r, t in msgs) or "(leer)"


def ok_gen(msgs, min_len=20):
    """OK, wenn ein assistant-markierter ODER nicht-leerer Text substanziell."""
    cand = [t for r, t in msgs if r in ("assistant", "", None)
            or r not in ("user", "system")]
    return any(len(t.strip()) >= min_len for t in cand)


def run(c, model_id, preset, message, timeout, mt=64, relay_layer=None):
    kw = dict(GEN_KW)
    kw["px_preset"] = preset
    kw["mt"] = mt
    if relay_layer is not None:
        kw["relay_layer"] = relay_layer
    job = c.submit(
        [user_turn(message)], model_id, kw["px_preset"],
        kw["temp"], kw["tp"], kw["mt"], kw["rp"], kw["gamma"],
        kw["thinking"], kw["thinking_budget"], kw["thinking_effort"],
        kw["relay_sign"], kw["relay_alpha"], kw["relay_layer"],
        kw["system_profile"], kw["system_prompt_text"], kw["session_id"],
        api_name="/bot_response")
    res = job.result(timeout=timeout)
    return as_history(res)


def main():
    # Auswahl: argv[1] in {"t1", "t2", "both"} (Default) — Quota-tauglich,
    # weil die Free-Quota (rollierend 24 h, ~262 s) Request-Dauer billt.
    want = sys.argv[1] if len(sys.argv) > 1 else "both"
    from gradio_client import Client
    token = None
    with open("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/.env") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    c = Client(URL, token=token)  # gradio_client 2.5.0: Param heißt 'token'
    print(f"[g5] connected: {URL}")

    results = {}
    if want in ("t1", "both"):
        # T1 — 270m Parität
        print("[g5] T1: gemma3-270m BASELINE ...")
        h1 = run(c, "gemma3-270m", "BASELINE",
                 "Sag kurz hallo in einem Satz.", timeout=300)
        msgs1 = extract(h1)
        print("[g5] T1 Transcript:\n" + fmt(msgs1))
        results["T1"] = ok_gen(msgs1)
        print(f"[g5] T1 {'OK' if results['T1'] else 'FAIL'}")

    if want in ("t2", "both"):
        # T2 — bonsai auf ZeroGPU (Lease, GF3-Build, Relay L34)
        print("[g5] T2: ternary-bonsai-27b ACTIVE_MANIFOLD_RELAY ...")
        h2 = run(c, "ternary-bonsai-27b", "ACTIVE_MANIFOLD_RELAY",
                 "Erkläre in einem Satz, was ein rekurrentes Transformer-"
                 "Residuum ist.", timeout=1800, mt=64, relay_layer=34)
        msgs2 = extract(h2)
        print("[g5] T2 Transcript:\n" + fmt(msgs2))
        results["T2"] = ok_gen(msgs2)
        print(f"[g5] T2 {'OK' if results['T2'] else 'FAIL'}")

    print("[g5] RESULT: " + " ".join(f"{k}={'OK' if v else 'FAIL'}"
                                     for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)