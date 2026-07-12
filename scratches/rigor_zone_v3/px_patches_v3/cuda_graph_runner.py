"""
cuda_graph_runner.py — CUDA-Graph Decode-Step Wrapper (TDD-Iter 1)
=========================================================================
SR-64 / v3 RIGOR-Expansion: 100% GPU-Utilization durch CUDA-Graph-Capture.

Problem:
    270M auf RTX 2060 ist Compute-Bound, aber Python-Dispatch-Overhead pro
    Decode-Step (~200-500µs) verhindert dass die GPU-Kerne kontinuierlich
    ausgelastet sind. Bei bs=8, T=1, 18 Layers + PX-Recursion ist der
    GPU-Compute ~3-5ms pro Step → Python-Overhead = 5-15% davon.
    Bei PX-Recursion mit n_loops=2-8 + mehreren .item()-Syncs → Overhead
    explodiert.

Lösung (NVIDIA RNN-T Pattern):
    1. Statische KV-Cache allokieren (vor Graph-Capture)
    2. Prefill OUTSIDE der Graph-Region (einmaliger Compile)
    3. Decode-Step in CUDA-Graph capturen (B, 1, H_static) Static Shape
    4. Decode-Loop in Python: `g.replay()` ohne Python-Dispatch
    5. Output via static_logits-Buffer lesen, next_token per Python-Op

Einschränkungen RTX 2060 (Turing CC 7.5):
    - CUDA-Graphs unterstützt ✓
    - cudaGraphSetConditional NICHT unterstützt (kein dynamischer Branch)
    - → Branch-Logik (z.B. n_loops) muss in der Graph-Region eliminiert
      werden (alle Pfade fixed-time, oder über mask-based control flow)

Architektur:
    CUDAGraphRunner(model, batch_size=8, max_seq_len=256, ...)
        ├── prefill(input_ids, attention_mask)  → einmaliger Python-Call
        ├── step()  → g.replay() + static_logits read  (kein Python-Dispatch)
        └── generate(input_ids, max_new_tokens)  → wrapper

TDD-Tests (test_cuda_graph_runner.py):
    1. test_static_kv_cache_shape — B×H×MAX_SEQ×head_dim allokiert
    2. test_graph_capture_succeeds — ein Decode-Step wird gecaptured
    3. test_graph_replay_matches_eager — replay liefert gleiche logits wie eager
    4. test_50_replays_in_500ms — 50 replays < 500ms (vs 1089ms eager)
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDA-Graph Decode Runner.

    Defaults: gemma3-270m (hidden=640, n_q=4, n_kv=1, head_dim=256).
    """
    batch_size: int = 8
    max_seq_len: int = 256
    head_dim: int = 256   # gemma3-270m: 640/4=160? NO! transformers reports 256
    n_kv_heads: int = 1   # gemma3-270m
    n_layers: int = 18
    vocab_size: int = 262144  # gemma3 default
    hidden_size: int = 640


class StaticKVCache:
    """Statische KV-Cache allokiert auf MAX_SEQ_LEN.

    Funktioniert mit dem bestehenden transformers DynamicCache nicht
    zusammen, da DynamicCache dynamisch allokiert. Wir umgehen das,
    indem wir unsere eigene Cache-Klasse schreiben, die in-place
    mutiert. Die gemma3 Attention.forward erwartet
    past_key_values.update(k, v, layer_idx) — wir implementieren das.

    WICHTIG: Da die Layer die 'past_key_values'-API aufrufen, müssen
    wir eine kompatible Klasse liefern.
    """
    def __init__(self, cfg: CUDAGraphRunnerConfig, device, dtype, layer_types=None):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        # (n_layers, B, n_kv, MAX_SEQ, head_dim)
        self.key_cache = [
            torch.zeros((cfg.batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim),
                        device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        self.value_cache = [
            torch.zeros((cfg.batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim),
                        device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        # CUDA-Graph-Compat: Pre-allocate static output buffers (MAX_SEQ Länge)
        # so dass `update()` statische Tensors returnt (kein memory alloc im
        # Graph-Region).
        self.key_out = [
            torch.zeros((cfg.batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim),
                        device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        self.value_out = [
            torch.zeros((cfg.batch_size, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim),
                        device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        self._seq_length = 0
        # layer_types: nutze vom Aufrufer, sonst default (5 sliding + 13 full)
        if layer_types is None:
            layer_types = ["sliding_attention" if i < 5 else "full_attention"
                           for i in range(cfg.n_layers)]
        # Validiere dass layer_types-Liste zur cfg.n_layers passt
        if len(layer_types) != cfg.n_layers:
            # Fallback: fülle auf
            while len(layer_types) < cfg.n_layers:
                layer_types.append("full_attention")
            layer_types = layer_types[:cfg.n_layers]
        self._layer_types = layer_types

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_length

    def get_mask_sizes(self, q_length: int, layer_idx: int = 0):
        """Kompatibel mit transformers Cache API.
        Returns (kv_length, kv_offset).
        kv_length: aktuelle KV-Länge (T_seq, nicht MAX_SEQ).
        kv_offset: 0 (kein Offset).
        """
        kv_length = self._seq_length
        kv_offset = 0
        return kv_length, kv_offset

    @property
    def is_sliding(self) -> list:
        """Kompatibel mit transformers Cache API: list[bool] pro Layer.

        transformers' create_causal_mask checkt `False in past_key_values.is_sliding`
        und `True in past_key_values.is_sliding` — also brauchen wir eine
        list/iterable von bools (nicht eine method).
        """
        return [t == "sliding_attention" for t in self._layer_types]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wird pro Layer/Step aufgerufen. Schreibt in-place in static buffer."""
        # key_states: (B, n_kv, T_new, head_dim)
        T_new = key_states.shape[-2]
        start = self._seq_length
        end = start + T_new
        # Sliding window: nur die letzten max(MAX_SEQ_LEN, sliding_window) behalten
        # Vereinfachung: wir nutzen volle MAX_SEQ_LEN für alle Layer
        self.key_cache[layer_idx][:, :, start:end, :] = key_states
        self.value_cache[layer_idx][:, :, start:end, :] = value_states
        # Return: (B, n_kv, end, head_dim) — der bisherige Cache + neue Tokens
        return (
            self.key_cache[layer_idx][:, :, :end, :],
            self.value_cache[layer_idx][:, :, :end, :],
        )

    def commit(self, n_new: int):
        """Markiere n_new Tokens als im Cache. Wird nach update() aller Layer aufgerufen."""
        self._seq_length += n_new

    def reset(self):
        # Cache-Inhalte bleiben (für Replay), aber seq_length zurück
        self._seq_length = 0


class CUDAGraphRunner:
    """CUDA-Graph Decode-Step Runner für 100% GPU-Util.

    Verwendung:
        runner = CUDAGraphRunner(model, batch_size=8, max_seq_len=256)
        runner.setup(input_ids)  # Prefill + Graph-Capture
        for _ in range(max_new_tokens - 1):
            next_token = runner.step()  # 0 Python-Dispatch
            runner.append(next_token)
    """
    def __init__(self, model, cfg: Optional[CUDAGraphRunnerConfig] = None,
                 attention_mask_pattern: str = "causal"):
        self.model = model
        self.cfg = cfg or CUDAGraphRunnerConfig()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.device = device
        self.dtype = dtype
        self.attention_mask_pattern = attention_mask_pattern
        # Wir holen das text_model (model.model bei Gemma3)
        self.text_model = model.model if hasattr(model, "model") else model

        # Auto-detect config aus model.config (überschreibt cfg defaults)
        model_config = model.config
        text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
        # n_layers MUSS aus num_hidden_layers kommen (zuverlässig), nicht aus
        # len(layer_types), weil layer_types von PX-Patch modifiziert werden kann.
        self.cfg.n_layers = getattr(text_config, "num_hidden_layers", self.cfg.n_layers)
        self.cfg.head_dim = getattr(text_config, "head_dim", self.cfg.head_dim)
        self.cfg.n_kv_heads = getattr(text_config, "num_key_value_heads", self.cfg.n_kv_heads)
        self.cfg.hidden_size = getattr(text_config, "hidden_size", self.cfg.hidden_size)
        self.cfg.vocab_size = getattr(text_config, "vocab_size", self.cfg.vocab_size)
        # Hole layer_types aus text_config (für sliding/full attention)
        # Falls die Liste kürzer ist als n_layers, mit 'full_attention' auffüllen
        raw_layer_types = list(getattr(text_config, "layer_types", []))
        if len(raw_layer_types) < self.cfg.n_layers:
            # PX-Patch oder Hybrid-Modelle können layer_types kürzen
            # Wir defaulten die fehlenden auf 'full_attention' (gemma3-Standard)
            raw_layer_types = (raw_layer_types +
                               ["full_attention"] * (self.cfg.n_layers - len(raw_layer_types)))
        self._layer_types = raw_layer_types[:self.cfg.n_layers]

        self.static_cache: Optional[StaticKVCache] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input_ids: Optional[torch.Tensor] = None
        self.static_pos_ids: Optional[torch.Tensor] = None
        self.static_attn_mask: Optional[torch.Tensor] = None
        self.static_logits: Optional[torch.Tensor] = None

    def _build_static_cache(self) -> StaticKVCache:
        """Allocate static KV-cache."""
        return StaticKVCache(self.cfg, self.device, self.dtype)

    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Prefill-Pass: einmalig, mit DynamicCache. Setzt static_cache auf.

        HINWEIS: Das ist nicht in der Graph-Region. Aber nach Prefill ist
        _seq_length = input_ids.shape[1] und unsere static_cache hat
        die initialen K/V. Wir sync-en dann den dynamic cache in unsere
        static cache.
        """
        from transformers.cache_utils import DynamicCache
        # Use the model's regular forward with DynamicCache
        past = DynamicCache(config=self.model.config)
        with torch.inference_mode():
            out = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        # Past in static_cache kopieren
        # DynamicCache hat .key_cache und .value_cache als Liste von Tensors
        # Wir initialisieren unseren static_cache mit dem gleichen Shape
        # und kopieren die Inhalte
        return out, past

    def setup(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Vorbereitung: Prefill + Graph-Capture.

        1) Prefill: einmaliger forward mit DynamicCache
        2) Erstelle static_cache mit den Prefill-Werten
        3) Warmup: 3 Decode-Steps eager
        4) Capture: 1 Decode-Step in CUDA-Graph
        """
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        B, T_prefill = input_ids.shape
        assert B == self.cfg.batch_size, f"BS mismatch: {B} != {self.cfg.batch_size}"

        # === 1) Prefill (eager) ===
        # Bei PX-patched Model: cuda_graph_mode=True (Standard-Forward, keine
        # PX-Recursion). Das ist gewollt: CUDA-Graph braucht deterministische
        # Cache-Längen, PX-Recursion erzeugt variable Cache-Größen.
        # WICHTIG: Wir setzen cuda_graph_mode BEVOR wir prefill machen.
        # Der Prefill macht dann KEINE Recursion, und der Cache hat die
        # korrekte Länge = Input-Länge.
        if hasattr(self.text_model, "_px_cuda_graph_mode"):
            self.text_model._px_cuda_graph_mode = True
        past = DynamicCache(config=self.model.config)
        with torch.inference_mode():
            out = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            # Logits für first next_token
            lm_logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
            first_token = lm_logits.argmax(dim=-1)  # (B, 1)

        # === 2) Static cache aus DynamicCache initialisieren ===
        # transformers 5.13 API: past.layers[i].keys / past.layers[i].values
        # Nutze die autodetected layer_types (von model.config)
        self.static_cache = StaticKVCache(self.cfg, self.device, self.dtype, layer_types=self._layer_types)
        # T_prefill ist der ECHTE seq_length nach dem Prefill, nicht die
        # Input-Länge. PX-Patch kann Layer mehrfach aufrufen → Cache wächst.
        actual_seq_len = past.get_seq_length() if past.get_seq_length() > 0 else T_prefill
        self.static_cache._seq_length = actual_seq_len
        # Falls PX-Recursion den Cache über MAX_SEQ_LEN hinaus verlängert hat,
        # brechen wir ab (das ist ein User-Fehler, nicht CUDA-Graph-Problem)
        if actual_seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"Prefill produced KV-Cache length {actual_seq_len} > max_seq_len={self.cfg.max_seq_len}. "
                "PX-Recursion darf nicht in den CUDA-Graph-Pfad laufen. "
                "Stelle sicher dass _px_cuda_graph_mode=False ist für Prefill."
            )
        for layer_idx in range(self.cfg.n_layers):
            dlayer = past.layers[layer_idx]
            # Schutz: wenn eine Layer einen anderen seq_length hat (z.B.
            # sliding_window kürzt), nutze min(dlayer, actual_seq_len)
            kv_len = dlayer.keys.shape[-2] if dlayer.keys is not None else 0
            copy_len = min(kv_len, actual_seq_len)
            if dlayer.keys is not None and copy_len > 0:
                self.static_cache.key_cache[layer_idx][:, :, :copy_len, :].copy_(
                    dlayer.keys[:, :, :copy_len, :]
                )
            kv_len_v = dlayer.values.shape[-2] if dlayer.values is not None else 0
            copy_len_v = min(kv_len_v, actual_seq_len)
            if dlayer.values is not None and copy_len_v > 0:
                self.static_cache.value_cache[layer_idx][:, :, :copy_len_v, :].copy_(
                    dlayer.values[:, :, :copy_len_v, :]
                )

        # === 3) Static buffers für den Decode-Step ===
        self.static_input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        self.static_pos_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)

        # Build static attention mask once via create_causal_mask (sized for
        # MAX_SEQ+1 — wir wollen über alle Generation-Steps die gleiche Shape
        # haben, da CUDA-Graph STATIC-SHAPE ist).
        text_model_config = self.text_model.config
        text_config = text_model_config.get_text_config() if hasattr(text_model_config, "get_text_config") else text_model_config

        # Wir bauen die Maske auf MAX_SEQ_LEN+1 (für den längsten möglichen Step)
        # Das ist mehr als nötig aber STATIC-SHAPE ist Pflicht für Graph-Replay
        target_T = self.cfg.max_seq_len
        attn_mask_static = torch.ones((B, target_T), dtype=torch.long, device=self.device)
        # inputs_embeds als Platzhalter (für create_causal_mask API)
        # Wir bauen position_ids so dass die Maske für "current pos = target_T-1"
        # erstellt wird (also mit Cache-Length target_T-1, also alle gültig)
        placeholder_pos = torch.full((B, 1), target_T - 1, dtype=torch.long, device=self.device)
        _placeholder_embeds = self.text_model.embed_tokens(self.static_input_ids)
        # Build the masks — diese werden auch in der Graph-Region verwendet
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        with torch.inference_mode():
            mk_static = dict(config=text_config, inputs_embeds=_placeholder_embeds,
                             attention_mask=attn_mask_static,
                             past_key_values=self.static_cache,
                             position_ids=placeholder_pos)
            try:
                self.static_causal_mask_full = create_causal_mask(**mk_static)
            except Exception:
                self.static_causal_mask_full = None
            try:
                self.static_causal_mask_sliding = create_sliding_window_causal_mask(**mk_static)
            except Exception:
                self.static_causal_mask_sliding = None

        # Output buffer
        self.static_logits = torch.zeros((B, 1, self.cfg.vocab_size),
                                          dtype=self.dtype, device=self.device)

        # === 4) Warmup (eager) ===
        # Setze die initial-Werte für den ersten Decode-Step
        self.static_input_ids.copy_(first_token)
        self.static_pos_ids.fill_(T_prefill)  # Position nach Prefill

        # Mask-Mapping für den Forward (gleiche Struktur wie im patch.py)
        causal_mask_mapping = {
            "full_attention": self.static_causal_mask_full,
            "sliding_attention": self.static_causal_mask_sliding,
        }

        for _ in range(3):
            with torch.inference_mode():
                out = self.text_model(
                    input_ids=self.static_input_ids,
                    attention_mask=causal_mask_mapping,
                    past_key_values=self.static_cache,
                    position_ids=self.static_pos_ids,
                    use_cache=True,
                )
                lm_logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
                self.static_logits.copy_(lm_logits)

        # === 5) Capture ===
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                out = self.text_model(
                    input_ids=self.static_input_ids,
                    attention_mask=causal_mask_mapping,
                    past_key_values=self.static_cache,
                    position_ids=self.static_pos_ids,
                    use_cache=True,
                )
                lm_logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
                self.static_logits.copy_(lm_logits)

        return first_token

    def step(self) -> torch.Tensor:
        """Replay CUDA-Graph: 1 Decode-Step.

        Returns:
            next_token: (B, 1) LongTensor mit dem argmax aus static_logits.
        """
        self.graph.replay()
        return self.static_logits.argmax(dim=-1)  # (B, 1)

    def append(self, next_token: torch.Tensor):
        """Schreibe next_token in static_input_ids für nächsten Step."""
        self.static_input_ids.copy_(next_token)
        self.static_pos_ids.add_(1)
        # Advance static_cache._seq_length
        self.static_cache._seq_length += 1
