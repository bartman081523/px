"""G5 Phase-2 (Plan hf-space-v4-publish): ZeroGPU-Chat-Test via gradio_client.

T1: gemma3-270m BASELINE — Parität/Contract: app + chat_fn-Lease + Generation.
T2: ternary-bonsai-27b ACTIVE_MANIFOLD_RELAY, relay_layer=34 — der Kern:
    privates Repo-Download (6 GB, Space-Secret HF_TOKEN) + GF3-Build +
    triton-JIT + Manifold-Load (models---Fallback, C1) + Relay (dwidth-
    Fallback, L34) + Generierung — alles innerhalb EINER Lease.
    Lease-Realität 2026-10-08 (A-Grad): Request = duration×1.5 (Profil-
    Faktor); 1350 s UND 270 s wurden serverseitig abgelehnt → Free-Cap
    < 270 s. duration läuft über Secret PX_SPACES_LEASE (Default 60).

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


def hist_text(history):
    """Erster assistant-Textblock aus der geliften History."""
    out = []
    for m in history or []:
        blocks = m.get("content") or []
        txt = "".join(b.get("text", "") for b in blocks
                      if isinstance(b, dict))
        out.append(f"[{m.get('role')}] {txt}")
    return "\n".join(out)


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
    return res[0] if isinstance(res, (list, tuple)) and res else res


def main():
    from gradio_client import Client
    token = None
    with open("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/.env") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    # Owner-Token: unbegrenzte Space-Quota (anonym ~260 s/Tag gemessen)
    c = Client(URL, token=token)  # gradio_client 2.5.0: Param heißt 'token'
    print(f"[g5] connected: {URL}")

    # T1 — 270m Parität
    print("[g5] T1: gemma3-270m BASELINE ...")
    h = run(c, "gemma3-270m", "BASELINE",
            "Sag kurz hallo in einem Satz.", timeout=300)
    t1 = hist_text(h)
    print("[g5] T1 Antwort:\n" + t1)
    ok1 = "assist" in t1 and len(hist_text(h)) > 20
    print(f"[g5] T1 {'OK' if ok1 else 'FAIL'}")

    # T2 — bonsai auf ZeroGPU (Lease, Download, GF3-Build, Relay)
    print("[g5] T2: ternary-bonsai-27b ACTIVE_MANIFOLD_RELAY "
          "(lease 175 → Request 262.5s) ...")
    h2 = run(c, "ternary-bonsai-27b", "ACTIVE_MANIFOLD_RELAY",
             "Erkläre in einem Satz, was ein rekurrentes Transformer-"
             "Residuum ist.", timeout=1100, mt=64, relay_layer=34)
    t2 = hist_text(h2)
    print("[g5] T2 Antwort:\n" + t2)
    ok2 = "assist" in t2 and len(t2) > 40
    print(f"[g5] T2 {'OK' if ok2 else 'FAIL'}")

    print(f"[g5] RESULT: T1={'OK' if ok1 else 'FAIL'} T2={'OK' if ok2 else 'FAIL'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)