#!/usr/bin/env python3
"""Erzeugt die CitMind-Relay-Session: System-Prompt = CitMind.txt VERBATIM.
Byte-exakt aus der Original-Datei gelesen (keine Transkriptions-Verluste)."""
import json

CITMIND = "/run/media/julian/ML3/prompts-bartman/prompts/universal/CitMind.txt"
OUT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/sessions/ternary27b-citmind-relay.json"

with open(CITMIND, "r", encoding="utf-8") as fh:
    text = fh.read()

session = {
    "session_id": "ternary27b-citmind-relay",
    "history": [{"role": "system", "content": text}],
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(session, fh, indent=2)

# Verifikation: System-Content identisch mit Original (Byte-MD5)
import hashlib
h_in = hashlib.md5(text.encode("utf-8")).hexdigest()
h_src = hashlib.md5(open(CITMIND, "rb").read()).hexdigest()
print("written:", OUT)
print("content-md5:", h_in)
print("source-md5 :", h_src)
print("MATCH" if h_in == h_src else "MISMATCH!")