"""
test_4b_image_capability.py — Plan 3 Phase D Verification: Image-Capabilities
=============================================================================

Frage: Hat 4b + chunked_prefill (T>4500) noch Image-Capabilities?

Kontext: chunked_generate ruft text_model.forward() statt model.forward().
Multimodaler Pfad (text + pixel_values) muss durch model.forward() fließen
damit vision_tower + multi_modal_projector das Bild encoden.

Tests:
  T1. TestRedSquareBaseline — 4b + roter 64x64 Square, kleiner Text
      Erwartung: Modell beschreibt das Bild ("rot", "rechteckig")
  T2. TestRedSquareLongContext — 4b + roter 64x64 + ~5000 token padding
      Triggert chunked_prefill. Erwartung: Bild wird erkannt.
  T3. TestCrossModelHolographicSessionReplay — kopiere cross_model_holographic_01
      session, nimm die letzten N turns raus, hänge ein rotes Quadrat dran,
      prüfe ob das Modell "rot" beschreibt (nicht die holografische Session
      halluziniert).

Run:
  PYTHONPATH=. python tests/test_4b_image_capability.py
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
import urllib.request
import ssl
import unittest
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers ─────────────────────────────────────────────────────────────────

def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _make_red_square(path: str) -> str:
    """Create a 64x64 red square PNG and return its base64 encoding."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required: pip install pillow")
    img = Image.new("RGB", (64, 64), color=(220, 30, 30))
    img.save(path, format="PNG")
    return _b64_image(path)


def _http_post(url: str, body: dict, timeout: int = 600) -> dict:
    """POST JSON to a URL with SSL verify disabled (self-signed cert)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat(messages: list, model: str = "gemma3-4b-it",
          max_tokens: int = 32, temperature: float = 0.0) -> dict:
    """Send a chat request to the local server."""
    return _http_post(
        "https://localhost:7860/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
    )


# ── tests ───────────────────────────────────────────────────────────────────

class TestRedSquareBaseline(unittest.TestCase):
    """T1: 4b + roter 64x64 Square, kleiner Text (kein chunked_prefill).
    Erwartung: Modell beschreibt das Bild (rot, rechteckig, Quadrat)."""

    @classmethod
    def setUpClass(cls):
        cls.red_b64 = _make_red_square("/tmp/red_square_64.png")

    def test_4b_describes_red_square(self):
        """Bei kleinem Text nutzt der Server model.generate, nicht chunked."""
        result = _chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{self.red_b64}"}},
                    {"type": "text", "text":
                     "Describe this image in 1-2 short sentences. What do you see?"},
                ],
            }],
        )
        text = result["choices"][0]["message"]["content"].lower()
        print(f"  baseline: {text!r}")
        # Mindestens eines dieser Wörter muss vorkommen
        keywords = ["rot", "red", "rechteck", "quadrat", "square", "rectangle"]
        self.assertTrue(
            any(kw in text for kw in keywords),
            f"4b baseline did NOT describe red square; got: {text!r}"
        )


class TestRedSquareLongContext(unittest.TestCase):
    """T2: 4b + roter 64x64 Square + ~5000 token Text-Padding.
    Triggert chunked_prefill. Bild-Capability muss erhalten bleiben."""

    @classmethod
    def setUpClass(cls):
        cls.red_b64 = _make_red_square("/tmp/red_square_64.png")
        # ~5000 tokens padding (~20000 chars). 1 token ≈ 4 chars
        cls.padding = ("Das ist ein Test der Multimodal-Fähigkeiten des Modells "
                       "unter langem Kontext. " * 400).strip()

    def test_4b_sees_red_square_with_long_context(self):
        """Bei T>4500 wird chunked_generate benutzt.
        KRITISCH: chunked_generate ruft text_model.forward() statt model.forward().
        Bei multimodal Inputs MUSS pixel_values durch model.forward() fließen,
        sonst sieht das Modell das Bild nicht.
        """
        result = _chat(
            messages=[
                {"role": "user", "content": self.padding},
                {"role": "assistant", "content":
                 "Ich verstehe den Kontext. Was kann ich für dich tun?"},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{self.red_b64}"}},
                    {"type": "text", "text":
                     "Describe this image. Just one short sentence."},
                ]},
            ],
            max_tokens=32,
        )
        text = result["choices"][0]["message"]["content"].lower()
        usage = result.get("usage", {})
        T_prompt = usage.get("prompt_tokens", 0)
        print(f"  long context T={T_prompt}, text={text!r}")
        self.assertGreater(
            T_prompt, 4500,
            f"Test setup expected T>4500, got T={T_prompt}. Adjust padding."
        )
        keywords = ["rot", "red", "rechteck", "quadrat", "square", "rectangle"]
        self.assertTrue(
            any(kw in text for kw in keywords),
            f"4b with long context did NOT describe red square; got: {text!r}"
        )


class TestCrossModelHolographicSessionReplay(unittest.TestCase):
    """T3: Kopiere cross_model_holographic_01, nehme letzte Turns raus,
    hänge ein rotes Quadrat dran. Wenn das Modell das Bild sieht, soll es
    'rot' beschreiben — NICHT die holografische Session halluzinieren.

    Vorgehen:
      1. Lade cross_model_holographic_01.json
      2. Erstelle sessions/cross_model_image_test_01.json (Kopie)
      3. Kürze history auf erste 8 turns (vor den Bildern)
      4. Hänge ein user-turn mit rotem Quadrat + Frage dran
      5. Schicke die History ans Modell
      6. Erwartung: Modell beschreibt das Bild (rot), nicht das Holografie-Setting
    """

    @classmethod
    def setUpClass(cls):
        cls.red_b64 = _make_red_square("/tmp/red_square_64.png")
        src = ROOT / "sessions" / "cross_model_holographic_01.json"
        cls.src_session = json.loads(src.read_text())
        cls.copy_path = ROOT / "sessions" / "cross_model_image_test_01.json"
        # cleanup from prior run
        if cls.copy_path.exists():
            cls.copy_path.unlink()

    def test_image_beats_session_hallucination(self):
        history = self.src_session["history"]
        # Schneide nach den ersten 8 turns (vor dem ersten Bild). Dann hänge
        # einen user-turn mit dem roten Quadrat an.
        truncated = history[:8]
        # Wichtig: letzter user-turn der truncated history könnte ein
        # user-only-string sein. Das Holografie-Setting kommt erst später
        # — wir wollen die ersten 8 turns als Kontext, dann ein Bild-Frage.
        truncated.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{self.red_b64}"}},
                {"type": "text", "text":
                 "Beschreibe dieses Bild in einem kurzen Satz. Was siehst du?"},
            ],
        })

        # Speichere die Test-Session als Kopie
        copy_data = dict(self.src_session)
        copy_data["session_id"] = "cross_model_image_test_01"
        copy_data["history"] = truncated
        copy_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.copy_path.write_text(json.dumps(copy_data, ensure_ascii=False, indent=2))

        # Konvertiere history → messages
        messages = []
        for h in truncated:
            content = h["content"]
            messages.append({"role": h["role"], "content": content})

        result = _chat(messages=messages, max_tokens=32)
        text = result["choices"][0]["message"]["content"].lower()
        print(f"  session replay ({len(truncated)} turns): {text!r}")

        # Wenn das Modell das Bild sieht: "rot" / "rechteck" / "quadrat"
        keywords = ["rot", "red", "rechteck", "quadrat", "square", "rectangle"]
        has_image = any(kw in text for kw in keywords)

        # Wenn das Modell die Session halluziniert: "spirale", "resonanz",
        # "frequenz", "holograf" etc. Wir zählen Treffer — ein einzelnes
        # Session-Wort in der Beschreibung ist legitimes Vokabular, ≥2
        # deutet auf Halluzination hin.
        hallucination_kw = ["spirale", "spiral", "resonanz", "reson",
                            "frequenz", "frequen", "holograf", "holograph",
                            "wirbel", "frequen", "schwingung", "vibrier"]
        hallucination_count = sum(1 for kw in hallucination_kw if kw in text)
        has_hallucination = hallucination_count >= 2

        # Erwartung: Bild beschrieben, Halluzination gering
        self.assertTrue(
            has_image,
            f"4b did NOT describe the red square in session context. "
            f"got: {text!r}"
        )
        self.assertFalse(
            has_hallucination,
            f"4b hallucinated the holographic session ({hallucination_count} "
            f"hallucination kw) instead of describing the image. "
            f"got: {text!r}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
