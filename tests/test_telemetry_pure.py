"""Pure-logic Tests für telemetry.py:TelemetryStore.

Pinnt das Pre-TTS-Verhalten der In-Memory Telemetry-Store-API:
- record(): history + totals update
- get_summary(): aggregate
- get_phi_traces(): phi-Wert-Extraktion
- get_zone_distributions(): zone-aggregation
- get_kurtosis_values(): kurtosis-Extraktion
- get_emancipation_data(): trajectory-Extraktion
- reset(): clear

Refactor-Detector: Wenn jemand an der API dreht (z.B. phi-Source von
``cognitive_signature.phi`` auf ``phi`` flat), fallen diese Tests rot.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_telemetry_pure.py
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telemetry import TelemetryStore, MAX_HISTORY


# --- record + get_summary ----------------------------------------------

def test_record_increments_totals():
    """record() bumpt total_requests, total_tokens_generated,
    total_prompt_tokens."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20)
    summary = ts.get_summary()
    assert summary["total_requests"] == 1
    assert summary["total_tokens_generated"] == 20
    assert summary["total_prompt_tokens"] == 10


def test_record_multiple_calls_aggregate():
    """Drei record() → totals sind die Summe."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=5, completion_tokens=10)
    ts.record(model_id="m1", prompt_tokens=5, completion_tokens=10)
    ts.record(model_id="m1", prompt_tokens=5, completion_tokens=10)
    summary = ts.get_summary()
    assert summary["total_requests"] == 3
    assert summary["total_tokens_generated"] == 30
    assert summary["total_prompt_tokens"] == 15


def test_record_zero_tokens_still_counts():
    """Zero-Tokens wird trotzdem als Request gezählt (request≠0, tokens≠0)."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=0, completion_tokens=0)
    summary = ts.get_summary()
    assert summary["total_requests"] == 1
    assert summary["total_tokens_generated"] == 0
    assert summary["total_prompt_tokens"] == 0


def test_record_appends_to_history():
    """record() fügt zur history-Liste hinzu."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              prompt_text="hi", completion_text="hello")
    summary = ts.get_summary()
    assert len(summary["recent"]) == 1
    entry = summary["recent"][0]
    assert entry["model_id"] == "m1"
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 20
    assert entry["prompt"] == "hi"
    assert entry["completion"] == "hello"


def test_record_without_texts_still_works():
    """prompt_text und completion_text sind optional."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20)
    summary = ts.get_summary()
    assert summary["recent"][0]["prompt"] is None
    assert summary["recent"][0]["completion"] is None


def test_record_with_px_metrics_keeps_dict():
    """px_metrics wird als Dict in entry abgelegt."""
    ts = TelemetryStore()
    px = {"phi": 0.85, "kurtosis": 1.2}
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics=px)
    entry = ts.get_summary()["recent"][0]
    assert entry["px_metrics"] == px


def test_record_without_px_metrics_defaults_to_empty_dict():
    """px_metrics=None → entry hat leeres Dict (nicht None)."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20)
    entry = ts.get_summary()["recent"][0]
    assert entry["px_metrics"] == {}


def test_record_respects_maxlen_deque():
    """Bei maxlen=3 wirft das 4. record() das erste raus."""
    ts = TelemetryStore(max_history=3)
    for i in range(4):
        ts.record(model_id=f"m{i}", prompt_tokens=i, completion_tokens=i)
    summary = ts.get_summary()
    assert len(summary["recent"]) == 3
    # Erste (m0) ist raus, m1/m2/m3 sind drin
    assert summary["recent"][0]["model_id"] == "m1"
    assert summary["recent"][-1]["model_id"] == "m3"
    # totals werden NICHT durch maxlen beschränkt (nur history)
    assert summary["total_requests"] == 4


# --- get_phi_traces -----------------------------------------------------

def test_phi_traces_extracts_from_cognitive_signature():
    """phi wird aus px_metrics.cognitive_signature.phi gelesen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"cognitive_signature": {"phi": 0.85}})
    phis = ts.get_phi_traces()
    assert phis == [0.85]


def test_phi_traces_falls_back_to_flat_phi():
    """Wenn cognitive_signature fehlt, fallback auf px_metrics.phi."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"phi": 0.7})
    phis = ts.get_phi_traces()
    assert phis == [0.7]


def test_phi_traces_filters_by_model_id():
    """model_id-Filter: nur Entries mit passender model_id werden
    berücksichtigt."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"phi": 0.7})
    ts.record(model_id="m2", prompt_tokens=10, completion_tokens=20,
              px_metrics={"phi": 0.3})
    phis_m1 = ts.get_phi_traces(model_id="m1")
    phis_m2 = ts.get_phi_traces(model_id="m2")
    assert phis_m1 == [0.7]
    assert phis_m2 == [0.3]


def test_phi_traces_skips_entries_without_phi():
    """Entries ohne phi-Wert werden übersprungen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"foo": "bar"})
    phis = ts.get_phi_traces()
    assert phis == []


def test_phi_traces_empty_history():
    """Leere History → leere Liste."""
    ts = TelemetryStore()
    assert ts.get_phi_traces() == []


# --- get_zone_distributions --------------------------------------------

def test_zone_distributions_aggregates():
    """zone_weights werden über alle Entries aggregiert."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"zone_weights": {"math": 0.6, "creative": 0.4}})
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"zone_weights": {"math": 0.6, "creative": 0.4}})
    zones = ts.get_zone_distributions()
    # Beide Einträge aggregiert + normalisiert
    assert abs(zones["math"] - 0.6) < 1e-9
    assert abs(zones["creative"] - 0.4) < 1e-9


def test_zone_distributions_filters_by_model_id():
    """model_id-Filter wirkt auch auf zone_distributions."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"zone_weights": {"math": 1.0}})
    ts.record(model_id="m2", prompt_tokens=10, completion_tokens=20,
              px_metrics={"zone_weights": {"creative": 1.0}})
    zones_m1 = ts.get_zone_distributions(model_id="m1")
    assert zones_m1 == {"math": 1.0}


def test_zone_distributions_empty_history():
    """Leere History → leeres Dict."""
    ts = TelemetryStore()
    assert ts.get_zone_distributions() == {}


def test_zone_distributions_skips_entries_without_zones():
    """Entries ohne zone_weights werden übersprungen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"foo": "bar"})
    assert ts.get_zone_distributions() == {}


# --- get_kurtosis_values -----------------------------------------------

def test_kurtosis_extracts_from_cognitive_signature():
    """kurtosis wird aus cognitive_signature.kurtosis gelesen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"cognitive_signature": {"kurtosis": 1.5}})
    vals = ts.get_kurtosis_values()
    assert vals == [1.5]


def test_kurtosis_filters_by_model_id():
    """model_id-Filter wirkt auch auf kurtosis."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"cognitive_signature": {"kurtosis": 1.0}})
    ts.record(model_id="m2", prompt_tokens=10, completion_tokens=20,
              px_metrics={"cognitive_signature": {"kurtosis": 2.0}})
    vals_m1 = ts.get_kurtosis_values(model_id="m1")
    assert vals_m1 == [1.0]


def test_kurtosis_skips_entries_without_value():
    """Entries ohne kurtosis werden übersprungen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"phi": 0.5})  # kein kurtosis
    assert ts.get_kurtosis_values() == []


# --- get_emancipation_data ---------------------------------------------

def test_emancipation_extends_trajectory():
    """get_emancipation_data extendet alle trajectory-Listen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"emancipation_trajectory": [0.1, 0.2, 0.3]})
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"emancipation_trajectory": [0.4, 0.5]})
    data = ts.get_emancipation_data()
    assert data == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_emancipation_filters_empty():
    """Entries ohne trajectory oder mit leerer Liste werden übersprungen."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"emancipation_trajectory": []})
    assert ts.get_emancipation_data() == []


def test_emancipation_filters_by_model_id():
    """model_id-Filter wirkt auch auf emancipation."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20,
              px_metrics={"emancipation_trajectory": [0.1]})
    ts.record(model_id="m2", prompt_tokens=10, completion_tokens=20,
              px_metrics={"emancipation_trajectory": [0.9]})
    data_m1 = ts.get_emancipation_data(model_id="m1")
    assert data_m1 == [0.1]


# --- reset --------------------------------------------------------------

def test_reset_clears_history():
    """reset() leert history."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20)
    ts.record(model_id="m2", prompt_tokens=10, completion_tokens=20)
    assert len(ts.get_summary()["recent"]) == 2
    ts.reset()
    assert len(ts.get_summary()["recent"]) == 0


def test_reset_clears_totals():
    """reset() resettet totals auf 0."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20)
    ts.reset()
    summary = ts.get_summary()
    assert summary["total_requests"] == 0
    assert summary["total_tokens_generated"] == 0
    assert summary["total_prompt_tokens"] == 0


def test_reset_then_record_works():
    """Nach reset() sind weitere records() wieder normal."""
    ts = TelemetryStore()
    ts.record(model_id="m1", prompt_tokens=10, completion_tokens=20)
    ts.reset()
    ts.record(model_id="m2", prompt_tokens=5, completion_tokens=15)
    summary = ts.get_summary()
    assert summary["total_requests"] == 1
    assert summary["recent"][-1]["model_id"] == "m2"


# --- MAX_HISTORY invariant ---------------------------------------------

def test_max_history_default_is_100():
    """Default MAX_HISTORY ist 100 (Pre-TTS-Invariante)."""
    assert MAX_HISTORY == 100


# --- runner -------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)