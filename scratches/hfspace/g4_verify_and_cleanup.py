"""G4 Integritäts-Check + _speedtest-Cleanup (Plan hf-space-v4-publish, 2026-10-08).

Server-seitigen LFS-SHA256 von gf3_model.safetensors gegen den Pre-Upload-Pin
prüfen (unabhängig vom lokalen Hash), dann die 3 _speedtest_*.bin löschen
(Früherer delete_file gab 401 mid-flight — Retry mit frischem HfApi(token)).

ERGEBNIS 2026-10-08: G4=PASS (gf3 sha256 == Pin d8bfe817…3c4c9e, alle Größen
exakt), alle 3 _speedtest-Dateien gelöscht (bemerkenswert: identischer
lfs.sha256 429961e4… untereinander — 16-MB-Duplikate).

Run: .../venv_openmythos/bin/python scratches/hfspace/g4_verify_and_cleanup.py
Exit 0 = G4 bestanden + Repo bereinigt.
"""
import sys

from huggingface_hub import HfApi

REPO = "neuralworm/ternary-bonsai-2-27b-hf"
PIN_SHA = "d8bfe8170f346c463eaa059823bee2a1ca89b5591381767adc5427812e3c4c9e"

# Pins aus dem lokalen Cache-Verzeichnis (certified ternary-bonsai-ptq-layout)
EXPECTED = {
    "gf3_model.safetensors": 6402717760,
    "tokenizer.json": 19997304,
    "config.json": 288280,
    "tokenizer_config.json": 16260,
    "special_tokens_map.json": 796,
    "generation_config.json": 168,
    "README.md": 889,
}

SPEEDTEST = ("_speedtest_legacy.bin", "_speedtest_xet_default.bin",
             "_speedtest_xet_fixed16.bin")


def load_token():
    token = None
    with open("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/.env") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    return token


def main():
    api = HfApi(token=load_token())

    info = api.repo_info(REPO, files_metadata=True)
    ok_sizes = True
    lfs_sha = None
    print("[g4] repo files (server-seitige Metadaten):")
    for s in sorted(info.siblings, key=lambda x: x.rfilename):
        lfs = getattr(s, "lfs", None)
        marker = ""
        if lfs is not None and getattr(lfs, "sha256", None):
            marker = f"  lfs.sha256={lfs.sha256}"
            if s.rfilename == "gf3_model.safetensors":
                lfs_sha = lfs.sha256
        print(f"  {s.rfilename}  {s.size}{marker}")
        if s.rfilename in EXPECTED and s.size != EXPECTED[s.rfilename]:
            ok_sizes = False
            print(f"    !! SIZE-MISMATCH, erwartet {EXPECTED[s.rfilename]}")

    sha_ok = (lfs_sha == PIN_SHA)
    print(f"[g4] gf3 sha256 == Pin? {sha_ok}")
    print(f"[g4] Size-Check: {'OK' if ok_sizes else 'MISMATCH'}")

    for fn in SPEEDTEST:
        try:
            api.delete_file(fn, repo_id=REPO, repo_type="model")
            print(f"[cleanup] geloescht: {fn}")
        except Exception as e:
            print(f"[cleanup] FEHLSCHLAG {fn}: "
                  f"{type(e).__name__}: {str(e)[:160]}")

    names = [s.rfilename for s in api.repo_info(REPO, files_metadata=True).siblings]
    remaining = [n for n in names if n.startswith("_speedtest")]
    print(f"[cleanup] verbleibende _speedtest-Dateien: "
          f"{remaining if remaining else 'keine'}")

    g4 = ok_sizes and sha_ok
    print(f"[g4] RESULT: G4={'PASS' if g4 else 'FAIL'} "
          f"CLEANUP={'OK' if not remaining else 'OPEN'}")
    return 0 if (g4 and not remaining) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)