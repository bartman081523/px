"""tests/test_thinking_budget.py — Plan 2026-10-05: Gemma4 Thinking-Budget.

User-Request (aktiv): "doch, gemma4 hat diese max thinking tokens ... also
kannst du auch für gemma4 thinking budget einstellung machen."

Realisierung (App-Level, da transformers 5.13.0 den generate()-Parameter
nicht kennt — nur ungemergter PR #42112): generators.ThinkingBudget-
LogitsProcessor zählt die ECHTEN Kanal-Tokens der Probe scratches/
gemma4_think_tokens_probe.py — open=<|channel> (100), confirm=thought
(45518, plain-vocab), close=<channel|> (101) — und maskiert bei Budget-
Überschreitung das NEXT-token-Dist auf close-only (finfo-min statt -inf:
der close-Logit bleibt real, Custom-Processoren laufen nach den Built-ins).

Pinnt die State-Machine (kein gemma4-Checkpoint/Tokenizer auf dieser
Maschine → Mock-Tokenizer mit den probe-verifizierten IDs):
  TB1:  Budget N → N in-channel Content-Tokens frei, dann GENAU EIN
        maskierter close-Step; der Folgeschritt konsumiert den close →
        Maske AUS (Close-Spam-Regression des consume-then-mask-Ordners)
  TB2:  in-channel "thought" (45518) ist CONTENT — zählt hoch, resettet
        NICHT (die confirmation-open-Maschine darf kein Echo auslösen)
  TB3:  out-of-channel open + NICHT-confirm Folge-Token → öffnet nie
        (kein Budget-Feuern außerhalb des Kanals)
  TB4:  out-of-channel open + confirm → öffnet; Budget zählt ab da
  TB5:  Budget 0 = unbegrenzt → kwargs-Gate hängt KEINEN Processor an
  TB6:  None/bool/Junk → kwargs-Gate no-op (kein Budget am generate)
  TB7:  fehlende Kanal-Infrastruktur (Tokenizer) → kwargs-Gate no-op
  TB8:  bestehende LogitsProcessorList wird komponiert (append), nicht
        ersetzt
  TB9:  finfo-min-Maske: close behält SEINEN Logit (real), alle anderen
        sinken auf finfo-min → masked argmax = close auch gegen stärkere
        Preference
  TB10: Prompt-Scan (pre-opened Kanal): open+confirm am Prompt-Ende →
        in_think @ Generation-Start, count 0; confirm-loses Open → außen;
        mittig geöffneter/nach close geschlossener Kanal → außen-Endzustand
  TB11: confirm_id=None (Vokabular ohne "thought") → degradiert zu
        confirm-losser Open (open+irgendein Token öffnet), Budget wirkt
        weiter; kwargs-Gate hängt trotzdem an

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_thinking_budget.py
"""
from __future__ import annotations
import os
import sys
import unittest

import torch
from transformers.generation.logits_process import LogitsProcessorList

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators import (ThinkingBudgetLogitsProcessor as TB,
                        _thinking_budget_kwargs, _think_channel_token_ids)

OPEN, CLOSE, CONFIRM = 100, 101, 45518
CONTENT = 777          # beliebiger in/out-of-channel Content-Token
VOCAB = 46000          # Mock-Größe > CONFIRM (Masken-Vollbild, finfo-min)


class MockTokenizer:
    """gemma4-artige Tokenizer-Oberfläche mit den Probe-IDs (unk 3)."""
    unk_token_id = 3
    _T2I = {"<|channel>": OPEN, "<channel|>": CLOSE, "thought": CONFIRM}

    def convert_tokens_to_ids(self, text):
        return self._T2I.get(text, self.unk_token_id)

    def convert_ids_to_tokens(self, tid):
        for text, i in self._T2I.items():
            if i == tid:
                return text
        return "<unk>"


class MockTokenizerNoConfirm(MockTokenizer):
    """Kanal-Open/Close existieren, "thought" NICHT (confirm-less Open)."""
    unk_token_id = 3
    _T2I = {"<|channel>": OPEN, "<channel|>": CLOSE}
    # "thought" fehlt → convert_tokens_to_ids liefert unk 3 → _resolve_
    # think_token lehnt ab (confirm=None).


class MockTokenizerEmpty:
    """gemma3-artig: keine Kanal-Tokens im Vokabular."""
    unk_token_id = 3
    _T2I: dict = {}

    def convert_tokens_to_ids(self, text):
        return self.unk_token_id

    def convert_ids_to_tokens(self, tid):
        return "<unk>"


class MockTokenizerUnkAttrNone:
    """Tokenizer OHNE unk_token_id-Attribut (Rund-trip-Check allein)."""
    _T2I = MockTokenizer._T2I

    def convert_tokens_to_ids(self, text):
        return self._T2I.get(text, None)

    def convert_ids_to_tokens(self, tid):
        for text, i in self._T2I.items():
            if i == tid:
                return text
        return None


def _dist(pref_id=CONTENT, close_logit=0.0):
    """1×VOCAB-Dist: pref logit 5.0, close-Logit real (Default 0)."""
    scores = torch.zeros(1, VOCAB)
    scores[0, pref_id] = 5.0
    scores[0, CLOSE] = close_logit
    return scores


def _sim_generate(proc, prompt_ids, prefs, max_steps=40):
    """Replay des echtenProcessor-Call-Musters: Call k bekommt
    input_ids=prompt+ausgegeben[:k-1] und scores = NEXT-token-Dist.

    prefs: pro Step die vom (fiktiven) Modell bevorzugte Token-ID.
    Return: die tatsächlich generierten IDs (argmax nach Maske).
    """
    ids = list(prompt_ids)
    out = []
    for k, pref in enumerate(prefs):
        sc = _dist(pref)
        masked = proc(torch.tensor([ids]), sc)
        pick = int(masked.argmax(dim=-1)[0])
        ids.append(pick)
        out.append(pick)
    return out


class StateMachine(unittest.TestCase):
    # --- TB1: Budget-N → ein maskierter Close-Step, dann Release ----------

    def test_tb1_budget_n_forces_close_once_then_releases(self):
        """Pre-opened Kanal im Prompt ([9, OPEN, CONFIRM]) + Budget 3:
        genau 3 Content-Tokens frei, 4. Step maskiert auf close ALLEIN,
        Folgeschritt konsumiert close → Maske AUS (kein close-Spam)."""
        proc = TB(torch.tensor([[9, OPEN, CONFIRM]]),
                  OPEN, CLOSE, CONFIRM, 3)
        gen = _sim_generate(proc, [9, OPEN, CONFIRM],
                            [CONTENT] * 6)
        self.assertEqual(gen[:3], [CONTENT, CONTENT, CONTENT], gen)
        self.assertEqual(gen[3], CLOSE, gen)      # Budget genau hier erzwungen
        self.assertEqual(gen[4], CONTENT, gen)    # Release — kein close-Spam
        self.assertEqual(gen[5], CONTENT, gen)

    # --- TB2-TB4: Öffnungs-/Count-Semantik ---------------------------------

    def test_tb2_in_channel_confirm_token_is_content_not_reset(self):
        """Budget 2, pre-opened Kanal ([9, OPEN, CONFIRM]), in-channel
        Content = confirm-ID selbst: "thought" ALS CONTENT zählt hoch —
        kein Reset, keine Open-Echo-Maschine → Maskierung nach 2 Tokens.
        (Bei fälschlichem Reset würde gen[2] nie close werden.)"""
        proc = TB(torch.tensor([[9, OPEN, CONFIRM]]),
                  OPEN, CLOSE, CONFIRM, 2)
        gen = _sim_generate(proc, [9, OPEN, CONFIRM], [CONFIRM] * 4)
        # 1. Step (Erster Call, n<=base_len) unmaskiert → pref
        self.assertEqual(gen[0], CONFIRM, gen)
        # consume CONFIRM (in-channel, nicht close) → count 1 <2
        self.assertEqual(gen[1], CONFIRM, gen)
        # consume CONFIRM → count 2 ≥2 → MASK → close erzwungen
        self.assertEqual(gen[2], CLOSE, gen)
        # consume close → außen → frei
        self.assertEqual(gen[3], CONFIRM, gen)

    def test_tb3_open_plus_nonconfirm_never_opens(self):
        """out-of-channel: open gefolgt von Nicht-confirm → öffnet nie,
        keine Maskierung (Content fließt frei ohne close-Zwang)."""
        proc = TB(torch.tensor([[9]]), OPEN, CLOSE, CONFIRM, 1)
        gen = _sim_generate(proc, [9], [OPEN, CONTENT, CONTENT, CONTENT])
        self.assertEqual(gen, [OPEN, CONTENT, CONTENT, CONTENT], gen)

    def test_tb4_open_plus_confirm_opens_and_counts(self):
        """out-of-channel open+confirm → Kanal offen, Budget 2 zählt die
        2 Content-Tokens, dann forced close, danach frei."""
        proc = TB(torch.tensor([[9]]), OPEN, CLOSE, CONFIRM, 2)
        gen = _sim_generate(
            proc, [9],
            [OPEN, CONFIRM, CONTENT, CONTENT, CONTENT, CONTENT])
        self.assertEqual(gen[0], OPEN, gen)
        self.assertEqual(gen[1], CONFIRM, gen)    # unmaskiert (Erster Call)
        self.assertEqual(gen[2], CONTENT, gen)    # consume confirm → offen
        self.assertEqual(gen[3], CONTENT, gen)    # count 1 <2
        self.assertEqual(gen[4], CLOSE, gen)      # count 2 ≥2 → Mask
        self.assertEqual(gen[5], CONTENT, gen)    # consume close → frei

    # --- TB9: Maske-Shape (finfo-min keeps close real) ----------------------

    def test_tb9_finfo_min_mask_keeps_close_logit_real(self):
        """Maskierter Step: alle Nicht-close auf finfo-min, close behält
        SEINEN Logit — argmax = close auch gegen einen stärkeren pref."""
        proc = TB(torch.tensor([[9, OPEN, CONFIRM]]),
                  OPEN, CLOSE, CONFIRM, 1)
        # bis zum Mask-Step rollen (Budget 1 → 1 Content-Token frei)
        _sim_generate(proc, [9, OPEN, CONFIRM], [CONTENT])
        sc = _dist(pref_id=CONTENT, close_logit=-4.2)
        masked = proc(torch.tensor([[9, OPEN, CONFIRM, CONTENT]]), sc)
        self.assertEqual(masked.shape, (1, VOCAB))
        self.assertEqual(int(masked.argmax(dim=-1)[0]), CLOSE)
        self.assertAlmostEqual(float(masked[0, CLOSE]), -4.2, places=5)
        finfo_min = torch.finfo(masked.dtype).min
        self.assertEqual(float(masked[0, CONTENT]), finfo_min)

    # --- TB10: Prompt-Scan (pre-opened/mid/closed States) -------------------

    def test_tb10_prompt_scan_states(self):
        """__init__-Scan: pre-opened (open+confirm am Ende) → in_think
        count 0; confirm-loses Open → außen; nach close geschlossener
        Kanal → außen; mittig offen+Content → außen-Ende."""
        p1 = TB(torch.tensor([[9, OPEN, CONFIRM]]),
                OPEN, CLOSE, CONFIRM, 2)
        self.assertTrue(p1.in_think)
        self.assertEqual(p1.count, 0)
        self.assertEqual(p1.base_len, 3)

        # confirm-loses Open am Prompt-Ende → Generation startet außen
        p2 = TB(torch.tensor([[9, OPEN]]), OPEN, CLOSE, CONFIRM, 2)
        self.assertFalse(p2.in_think)
        self.assertFalse(p2.pending_confirm)

        # Kanal wurde im Prompt wieder geschlossen → außen
        p3 = TB(torch.tensor([[9, OPEN, CONFIRM, CONTENT, CLOSE, CONTENT]]),
                OPEN, CLOSE, CONFIRM, 2)
        self.assertFalse(p3.in_think)

        # mittig geöffneter Kanal bleibt offen (Scan ignorier Content)
        p4 = TB(torch.tensor([[9, OPEN, CONFIRM, CONTENT]]),
                OPEN, CLOSE, CONFIRM, 2)
        self.assertTrue(p4.in_think)
        self.assertEqual(p4.count, 0)

        # 1D-input_ids (Batch-dim fehlend) akzeptiert
        p5 = TB(torch.tensor([9, OPEN, CONFIRM]), OPEN, CLOSE, CONFIRM, 2)
        self.assertTrue(p5.in_think)

    # --- TB11: confirm-loses Vokabular ---------------------------------------

    def test_tb11_confirm_none_degrades_to_one_token_open(self):
        """Ohne "thought" im Vokabular (confirm None): open + IRGENDEIN
        Token öffnet (1-Token-Delay), Budget wirkt im Kanal weiter."""
        proc = TB(torch.tensor([[9]]), OPEN, CLOSE, None, 1)
        gen = _sim_generate(
            proc, [9], [OPEN, CONTENT, CONTENT, CONTENT, CONTENT])
        self.assertEqual(gen[0], OPEN, gen)
        self.assertEqual(gen[1], CONTENT, gen)    # unmaskiert (Erster Call)
        self.assertEqual(gen[2], CONTENT, gen)    # consume CONTENT → öffnet
        self.assertEqual(gen[3], CLOSE, gen)      # count 1 ≥1 → Mask
        self.assertEqual(gen[4], CONTENT, gen)    # frei


class KwargsGate(unittest.TestCase):
    # --- TB5-TB7: _thinking_budget_kwargs-Gating -----------------------------

    def _tok(self):
        return MockTokenizer()

    def test_tb5_budget_zero_means_unlimited_no_processor(self):
        """Budget 0 = unbegrenzt → kein Processor am generate."""
        g = _thinking_budget_kwargs({}, torch.tensor([[9, OPEN]]),
                                    0, self._tok())
        self.assertNotIn("logits_processor", g)

    def test_tb6_none_bool_and_junk_never_attach(self):
        """None (= kein Budget am Modell), bool (Slider liefert Zahlen,
        Checkbox-Verwechslung) und Junk-Strings → kwargs unangetastet."""
        for bad in (None, True, False, "2048"):
            g = _thinking_budget_kwargs({}, torch.tensor([[9]]), bad,
                                        self._tok())
            self.assertNotIn("logits_processor", g, bad)
        # Numerischer float WIRD durchgereicht (Slider liefert float —
        # int-cast auf den Processor): 3.9 → Budget 3.
        g = _thinking_budget_kwargs({}, torch.tensor([[9]]), 3.9, self._tok())
        self.assertIn("logits_processor", g)
        proc_f = g["logits_processor"][0]
        self.assertEqual(proc_f.budget, 3)

    def test_tb7_missing_infrastructure_noop(self):
        """Tokenizer None / ohne Kanal-Tokens / ohne "thought" → die ersten
        beiden Fälle no-op; confirm-losses Vokabular hängt TROTZDEM an
        (degraded Open ist legitime Infrastruktur)."""
        ids = torch.tensor([[9, OPEN]])
        g = _thinking_budget_kwargs({}, ids, 64, None)
        self.assertNotIn("logits_processor", g)
        g = _thinking_budget_kwargs({}, ids, 64, MockTokenizerEmpty())
        self.assertNotIn("logits_processor", g)
        ids_ok = _think_channel_token_ids(MockTokenizer())
        self.assertEqual(ids_ok, (OPEN, CLOSE, CONFIRM))
        no_confirm = _think_channel_token_ids(MockTokenizerNoConfirm())
        self.assertEqual(no_confirm, (OPEN, CLOSE, None))
        # confirm-losses Vokabular → Processor Hängt an (TB11-Semantik)
        g = _thinking_budget_kwargs({}, ids, 64, MockTokenizerNoConfirm())
        self.assertIn("logits_processor", g)
        # Rund-trip-Schutz: unk_token_id-Attr fehlt → round-trip allein
        ids_nc = _think_channel_token_ids(MockTokenizerUnkAttrNone())
        self.assertEqual(ids_nc, (OPEN, CLOSE, CONFIRM))

    def test_tb8_existing_processor_list_is_composed(self):
        """bestehende LogitsProcessorList → append (Liste wächst), nicht
        ersetzt; fehlender Key → neue Liste mit genau EINEM Processor."""
        class Dummy:
            pass
        plist = LogitsProcessorList([Dummy()])
        g = _thinking_budget_kwargs({}, torch.tensor([[9]]), 1024,
                                    self._tok())
        proc_new = g["logits_processor"]
        self.assertIsInstance(proc_new, LogitsProcessorList)
        self.assertEqual(len(proc_new), 1)
        self.assertIsInstance(proc_new[0], TB)
        # Komposition
        g2 = _thinking_budget_kwargs({"logits_processor": plist},
                                     torch.tensor([[9]]), 1024, self._tok())
        self.assertIs(g2["logits_processor"], plist)
        self.assertEqual(len(plist), 2)
        self.assertIsInstance(plist[-1], TB)


if __name__ == "__main__":
    unittest.main(verbosity=2)