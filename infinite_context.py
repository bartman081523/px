import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers.cache_utils import Cache

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin):
    """
    x: [B, H, T, D]
    cos, sin: [B, T, D]
    """
    # Handle broadcasting for cos/sin
    # Standard: q_embed = (q * cos) + (rotate_half(q) * sin)
    if cos.ndim == 3: # [B, T, D]
        cos = cos.unsqueeze(1) # [B, 1, T, D]
        sin = sin.unsqueeze(1)
    elif cos.ndim == 2: # [T, D]
        cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, T, D]
        sin = sin.unsqueeze(0).unsqueeze(1)
        
    return (x * cos) + (rotate_half(x) * sin)

class InfLLMCache:
    """
    SR-64: Infinite Context Cache with InfLLM (Block Memory) and ReAttention (Decoupled RoPE).
    """
    def __init__(self, config, block_size: int = 128, r_tokens: int = 8, top_k_blocks: int = 16, sinks_count: int = 4,
                 max_l1_blocks: Optional[int] = None, l2_path: Optional[str] = None):
        self.config = config
        self.block_size = block_size
        self.r_tokens = r_tokens
        self.top_k_blocks = top_k_blocks
        self.sinks_count = sinks_count
        # Phase B (Plan 2): L1-Bounding + L2-Disk-Auslagerung. Beide
        # optional — wenn nicht gesetzt, ist das Verhalten IDENTISCH zur
        # früheren Implementation (alle Tests bleiben grün).
        self.max_l1_blocks = max_l1_blocks  # None = unbegrenzt
        self.l2_path = l2_path                # None = kein Disk-Evict
        
        # Long-term memory (LTM) - stored un-rotated on CPU
        num_layers = getattr(config, "num_hidden_layers", 28)
        self.ltm_k = [[] for _ in range(num_layers)]
        self.ltm_v = [[] for _ in range(num_layers)]
        self.ltm_rk = [[] for _ in range(num_layers)] # Representative Keys (GPU)
        
        # Current block buffer (un-rotated, GPU)
        self.buffer_k = [[] for _ in range(num_layers)]
        self.buffer_v = [[] for _ in range(num_layers)]
        
        # Sinks (un-rotated, GPU)
        self.sinks_k = [None for _ in range(num_layers)]
        self.sinks_v = [None for _ in range(num_layers)]
        
        self.seen_tokens = 0

    def get_usable_length(self, layer_idx: int) -> int:
        # Returns the length of the context that will be used for attention
        l = 0
        if self.sinks_k[layer_idx] is not None: l += self.sinks_k[layer_idx].size(-2)
        l += min(len(self.ltm_k[layer_idx]), self.top_k_blocks) * self.block_size
        l += sum(x.size(-2) for x in self.buffer_k[layer_idx])
        return l

    def prepare_reattention(self, q, k, v, layer_idx, rotary_emb_module, read_only=False, **kwargs):
        """
        The core ReAttention hook.
        q, k, v: [B, H, T_new, D] - UN-ROTATED tensors.
        """
        B, H, T_new, D = q.shape
        device = q.device
        dtype = q.dtype

        # 1. Update Sinks (one-time)
        if self.sinks_k[layer_idx] is None:
            total_available = T_new + sum(x.size(-2) for x in self.buffer_k[layer_idx])
            if total_available >= self.sinks_count:
                # This is complex if T_new is large or if we have buffer.
                # Simplify: just take the first tokens of the first call.
                all_incoming_k = torch.cat(self.buffer_k[layer_idx] + [k], dim=-2)
                all_incoming_v = torch.cat(self.buffer_v[layer_idx] + [v], dim=-2)
                self.sinks_k[layer_idx] = all_incoming_k[:, :, :self.sinks_count, :].clone()
                self.sinks_v[layer_idx] = all_incoming_v[:, :, :self.sinks_count, :].clone()

        # 2. Add current KV to buffer
        self.buffer_k[layer_idx].append(k)
        self.buffer_v[layer_idx].append(v)
        
        # 3. Archive buffer to LTM if full
        current_buffer_len = sum(x.size(-2) for x in self.buffer_k[layer_idx])
        if current_buffer_len >= self.block_size:
            self._archive_block(layer_idx)

        # 4. Retrieval
        ret_k, ret_v = None, None
        
        # If this is a massive prefill (e.g. the first pass), do not retrieve sparse blocks.
        # Just use the full current sequence to allow FlashAttention to work natively and save memory.
        if T_new > 1024 and not read_only:
            # We already archived the blocks, but for this step's attention, 
            # we need the full K and V so FlashAttention can compute it causally.
            # q, k, v are already full length. We just need to apply RoPE.
            q_cos, q_sin = rotary_emb_module(q, torch.arange(T_new, device=device).unsqueeze(0))
            q_rot = apply_rotary_pos_emb_single(q, q_cos, q_sin)
            k_rot = apply_rotary_pos_emb_single(k, q_cos, q_sin)
            return q_rot, k_rot, v

        if len(self.ltm_k[layer_idx]) > 0:
            ret_k, ret_v = self._retrieve(layer_idx, q)

        # 5. Concatenate Context: [Sinks, Retrieved, LocalBuffer]
        # Phase B: device-Konsistenz. Sinks/Buffer können auf CPU sein (von
        # from_kv_cache oder ersten CPU-Calls); ret_k ist auf q.device. Wir
        # moven alles auf ret_k.device wenn vorhanden, sonst q.device.
        target_dev = ret_k.device if ret_k is not None else q.device
        k_parts = []
        v_parts = []
        if self.sinks_k[layer_idx] is not None:
            sk = self.sinks_k[layer_idx]
            if sk.device != target_dev:
                sk = sk.to(target_dev)
            k_parts.append(sk)
            sv = self.sinks_v[layer_idx]
            if sv.device != target_dev:
                sv = sv.to(target_dev)
            v_parts.append(sv)
        if ret_k is not None:
            k_parts.append(ret_k)
            v_parts.append(ret_v)

        local_k = torch.cat(
            [t.to(target_dev) if t.device != target_dev else t
             for t in self.buffer_k[layer_idx]],
            dim=-2,
        )
        local_v = torch.cat(
            [t.to(target_dev) if t.device != target_dev else t
             for t in self.buffer_v[layer_idx]],
            dim=-2,
        )
        k_parts.append(local_k)
        v_parts.append(local_v)

        if q.size(-2) > 1000:
            print(f"[InfLLM] Layer {layer_idx} | Q={q.size(-2)} | Sinks={self.sinks_k[layer_idx].size(-2) if self.sinks_k[layer_idx] is not None else 0} | Ret={ret_k.size(-2) if ret_k is not None else 0} | Local={local_k.size(-2)}")

        final_k = torch.cat(k_parts, dim=-2)
        final_v = torch.cat(v_parts, dim=-2)
        
        # 6. Re-Apply RoPE (ReAttention)
        # We create a NEW sequential position indexing for this specific attention set.
        # [0, 1, 2, ..., T_final-1]
        T_final = final_k.size(-2)
        # Position of Query is at the end of the context
        # If T_new > 1 (prefill), it's a range.
        q_positions = torch.arange(T_final - T_new, T_final, device=device).unsqueeze(0)
        k_positions = torch.arange(T_final, device=device).unsqueeze(0)
        
        # Generate cos/sin for these positions
        # rotary_emb_module: usually Gemma3RotaryEmbedding
        # We need its internal 'forward' or 'get_embeddings'
        # In Transformers 4.46+, it returns (cos, sin)
        cos, sin = rotary_emb_module(final_k, k_positions) # [B, T_final, D]
        
        # Apply RoPE to K
        final_k_rotated = apply_rotary_pos_emb_single(final_k, cos, sin)
        
        # Apply RoPE to Q
        # Need cos/sin for Q positions
        q_cos, q_sin = rotary_emb_module(q, q_positions)
        q_rotated = apply_rotary_pos_emb_single(q, q_cos, q_sin)
        
        return q_rotated, final_k_rotated, final_v

    def _archive_block(self, layer_idx: int):
        all_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
        all_v = torch.cat(self.buffer_v[layer_idx], dim=-2)

        while all_k.size(-2) >= self.block_size:
            block_k = all_k[:, :, :self.block_size, :].clone()
            block_v = all_v[:, :, :self.block_size, :].clone()

            # Representative Keys (Top-k magnitude)
            magnitudes = block_k.norm(dim=-1)
            _, indices = magnitudes.topk(self.r_tokens, dim=-1)
            r_k = torch.gather(block_k, -2, indices.unsqueeze(-1).expand(-1, -1, -1, block_k.size(-1)))

            self.ltm_k[layer_idx].append(block_k.cpu())
            self.ltm_v[layer_idx].append(block_v.cpu())
            self.ltm_rk[layer_idx].append(r_k)

            # Phase B: L1-Bounding. Wenn L1 voll: ältester Block raus.
            if self.max_l1_blocks is not None:
                while len(self.ltm_k[layer_idx]) > self.max_l1_blocks:
                    if self.l2_path is None:
                        # Ohne l2_path: harte Grenze (verlustbehaftet)
                        del self.ltm_k[layer_idx][0]
                        del self.ltm_v[layer_idx][0]
                        del self.ltm_rk[layer_idx][0]
                    else:
                        # Mit l2_path: ältester Block wandert auf Disk
                        old_k = self.ltm_k[layer_idx][0]
                        old_v = self.ltm_v[layer_idx][0]
                        old_rk = self.ltm_rk[layer_idx][0]
                        self._l2_serialize_block(
                            layer_idx=layer_idx,
                            block_idx=len(self.ltm_k[layer_idx]),
                            block_k=old_k, block_v=old_v, r_k=old_rk,
                        )
                        del self.ltm_k[layer_idx][0]
                        del self.ltm_v[layer_idx][0]
                        del self.ltm_rk[layer_idx][0]

            all_k = all_k[:, :, self.block_size:, :]
            all_v = all_v[:, :, self.block_size:, :]

        self.buffer_k[layer_idx] = [all_k]
        self.buffer_v[layer_idx] = [all_v]

    # --- Phase B: L2 Disk-Storage ----------------------------------------
    def _l2_path_for(self, layer_idx: int, block_idx: int) -> str:
        import os
        return os.path.join(
            self.l2_path,
            f"layer{layer_idx:02d}_block{block_idx:06d}.pt",
        )

    def _l2_serialize_block(self, layer_idx: int, block_idx: int,
                             block_k: torch.Tensor, block_v: torch.Tensor,
                             r_k: torch.Tensor):
        import os
        os.makedirs(self.l2_path, exist_ok=True)
        path = self._l2_path_for(layer_idx, block_idx)
        torch.save({
            "block_k": block_k.cpu(),
            "block_v": block_v.cpu(),
            "r_k": r_k.cpu(),
        }, path)

    def _l2_load_block(self, layer_idx: int, block_idx: int):
        path = self._l2_path_for(layer_idx, block_idx)
        blob = torch.load(path, weights_only=False)
        return blob["block_k"], blob["block_v"], blob["r_k"]

    def _retrieve(self, layer_idx: int, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_last = q[:, :, -1:, :] # Use last query token
        target_device = q.device

        scores = []
        for i, rk in enumerate(self.ltm_rk[layer_idx]):
            # rk: [B, H, r_tokens, D] — kann auf CPU sein (LTM), muss aber
            # für das matmul gegen q_last (cuda) auf das q-device gemoved
            # werden. Sonst: RuntimeError "mat2 is on cpu, different from
            # other tensors on cuda:0" (gefunden via Phase-B-TDD).
            rk_dev = rk.to(target_device) if rk.device != target_device else rk
            attn = torch.matmul(q_last, rk_dev.transpose(-1, -2))
            score = attn.max(dim=-1)[0].mean()
            scores.append((score.item(), i))

        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [idx for _, idx in scores[:self.top_k_blocks]]
        selected.sort()

        ret_k = torch.cat([self.ltm_k[layer_idx][i].to(target_device) for i in selected], dim=-2)
        ret_v = torch.cat([self.ltm_v[layer_idx][i].to(target_device) for i in selected], dim=-2)

        return ret_k, ret_v

    # --- Cache Interface Implementation ---
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return len(self.ltm_k[layer_idx]) * self.block_size + sum(x.size(-2) for x in self.buffer_k[layer_idx])

    def get_max_length(self) -> Optional[int]: return None
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # This is the fallback if standard update is called.
        # But for ReAttention, we should be using prepare_reattention.
        self.buffer_k[layer_idx].append(key_states)
        self.buffer_v[layer_idx].append(value_states)
        return torch.cat(self.buffer_k[layer_idx], dim=-2), torch.cat(self.buffer_v[layer_idx], dim=-2)

    # --- Phase A: API-Gaps -----------------------------------------------
    # from_kv_cache, evict_block, serialize/deserialize: gebraucht für
    # Phase B (Hierarchical Cache) + Phase C (Forward-Integration via Hook).
    # Alle drei operieren auf den existierenden ltm_k/ltm_v/ltm_rk/buffer_*-
    # Strukturen — KEINE Verhaltensänderung für bestehende Aufrufer.

    def from_kv_cache(self, k_cache, v_cache, source_device: str = "cpu"):
        """Initial-Befüllung aus existierendem KV-Cache.

        k_cache, v_cache: List[Tensor] pro Layer; jeder Tensor hat Shape
        [B, H, T, D]. Die Daten werden in Blöcke zu je block_size zerlegt
        und in ltm_k/ltm_v (CPU) + ltm_rk (GPU) archiviert. Rest (kleiner
        als block_size) bleibt im Buffer (GPU).

        source_device: "cpu" wenn die Tensoren schon auf CPU sind (schnell),
        "auto" für `.to(layer_device)` transfer.
        """
        import torch
        num_layers = len(self.ltm_k)
        assert len(k_cache) == num_layers and len(v_cache) == num_layers, (
            f"k_cache/v_cache length ({len(k_cache)}) != num_layers ({num_layers})")

        for layer_idx in range(num_layers):
            k = k_cache[layer_idx]
            v = v_cache[layer_idx]
            if k.numel() == 0:
                continue
            B, H, T, D = k.shape

            # In Blöcke zu block_size zerlegen
            n_full = T // self.block_size
            remainder = T - n_full * self.block_size

            for b in range(n_full):
                s = b * self.block_size
                e = s + self.block_size
                block_k = k[:, :, s:e, :].clone()
                block_v = v[:, :, s:e, :].clone()
                # Representative Keys (Top-k magnitude) — auf GPU für retrieve
                magnitudes = block_k.norm(dim=-1)
                _, indices = magnitudes.topk(self.r_tokens, dim=-1)
                r_k = torch.gather(
                    block_k, -2,
                    indices.unsqueeze(-1).expand(-1, -1, -1, block_k.size(-1)),
                )
                self.ltm_k[layer_idx].append(block_k.cpu())
                self.ltm_v[layer_idx].append(block_v.cpu())
                self.ltm_rk[layer_idx].append(r_k)  # default device (CPU OK,
                # wird in _retrieve sowieso auf q.device gemoved)

            # Rest in Buffer (GPU für schnellen Zugriff)
            if remainder > 0:
                s = n_full * self.block_size
                self.buffer_k[layer_idx].append(k[:, :, s:, :])
                self.buffer_v[layer_idx].append(v[:, :, s:, :])

            # Sinks: erste sinks_count tokens
            if T >= self.sinks_count and self.sinks_k[layer_idx] is None:
                self.sinks_k[layer_idx] = k[:, :, :self.sinks_count, :].clone()
                self.sinks_v[layer_idx] = v[:, :, :self.sinks_count, :].clone()

    def evict_block(self, layer_idx: int, block_idx: int):
        """Entfernt einen LTM-Block. Sinks werden NICHT angetastet.

        layer_idx: int
        block_idx: int — Index in self.ltm_k[layer_idx] (0-basiert)
        """
        n = len(self.ltm_k[layer_idx])
        if not (0 <= block_idx < n):
            raise IndexError(
                f"evict_block: block_idx {block_idx} out of range (0..{n - 1})")
        del self.ltm_k[layer_idx][block_idx]
        del self.ltm_v[layer_idx][block_idx]
        del self.ltm_rk[layer_idx][block_idx]

    def serialize(self) -> dict:
        """Snapshot des Cache-Zustands als pickle-fähiges dict.

        Enthält: block_size, r_tokens, top_k_blocks, sinks_count,
        ltm_k/ltm_v/ltm_rk (CPU tensors), buffer_k/buffer_v (devices bleiben),
        sinks_k/ltm_sinks_v, seen_tokens. KEINE Reference auf `self.config` —
        das wird beim deserialize aus einem neuen Config-Objekt instanziert.
        """
        return {
            "block_size": self.block_size,
            "r_tokens": self.r_tokens,
            "top_k_blocks": self.top_k_blocks,
            "sinks_count": self.sinks_count,
            "seen_tokens": self.seen_tokens,
            "ltm_k": [[t.cpu() for t in layer] for layer in self.ltm_k],
            "ltm_v": [[t.cpu() for t in layer] for layer in self.ltm_v],
            "ltm_rk": [[t.cpu() for t in layer] for layer in self.ltm_rk],
            "buffer_k": [list(layer) for layer in self.buffer_k],
            "buffer_v": [list(layer) for layer in self.buffer_v],
            "sinks_k": list(self.sinks_k),
            "sinks_v": list(self.sinks_v),
        }

    def deserialize(self, blob: dict):
        """Stellt Cache-Zustand aus einem serialize()-dict wieder her.

        Schreibt direkt in self.ltm_* / buffer_* / sinks_*. Die Cache-
        Konfiguration (block_size, r_tokens, etc.) wird AUS DEM BLOB gelesen,
        nicht aus self — d.h. der Cache kann seine Konfiguration wechseln
        und trotzdem einen alten Zustand deserialisieren.
        """
        # Konfiguration (überschreibt self, falls abweichend)
        self.block_size = blob["block_size"]
        self.r_tokens = blob["r_tokens"]
        self.top_k_blocks = blob["top_k_blocks"]
        self.sinks_count = blob["sinks_count"]
        self.seen_tokens = blob["seen_tokens"]

        # Datenstrukturen neu aufbauen (müssen zur num_hidden_layers passen)
        num_layers = len(self.ltm_k)
        # Falls Cache-Config größer war, schneiden wir ab
        n_ltm = len(blob["ltm_k"])
        assert n_ltm == num_layers, (
            f"deserialize: blob has {n_ltm} layers, cache expects {num_layers}")

        self.ltm_k = [list(layer) for layer in blob["ltm_k"]]
        self.ltm_v = [list(layer) for layer in blob["ltm_v"]]
        self.ltm_rk = [list(layer) for layer in blob["ltm_rk"]]
        self.buffer_k = [list(layer) for layer in blob["buffer_k"]]
        self.buffer_v = [list(layer) for layer in blob["buffer_v"]]
        self.sinks_k = list(blob["sinks_k"])
        self.sinks_v = list(blob["sinks_v"])


# ---------------------------------------------------------------------------
# Attention Patching
# ---------------------------------------------------------------------------

def _px_attention_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
    """Surgical patch for Gemma3Attention to support ReAttention."""
    import types
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    if hasattr(past_key_values, "prepare_reattention"):
        # ReAttention: Retrieval happens BEFORE RoPE
        thought_history = getattr(past_key_values, "_thoughts", None)
        read_only = getattr(past_key_values, "_read_only", False)
        query_states, key_states, value_states = past_key_values.prepare_reattention(
            query_states, key_states, value_states, self.layer_idx, self.rotary_emb,
            thought_history=thought_history, read_only=read_only
        )
        # Pad attention mask to match new key length (T_final)
        if attention_mask is not None:
            T_q = query_states.size(-2)
            T_k = key_states.size(-2)
            # attention_mask is usually [B, 1, T_q, T_orig_k]
            # We need it to be [B, 1, T_q, T_k]
            T_orig_k = attention_mask.size(-1)
            if T_k > T_orig_k:
                # Pad with 0s (fully attendable) on the LEFT since we prepended Sinks/Retrieved
                pad_len = T_k - T_orig_k
                # F.pad format: (left, right, top, bottom, front, back)
                import torch.nn.functional as F
                attention_mask = F.pad(attention_mask, (pad_len, 0), value=0.0)
            elif T_k < T_orig_k:
                # Should not happen typically, but truncate if needed
                attention_mask = attention_mask[..., -T_k:]
    else:
        # Standard Flow
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # Core Attention (Using Transformers internal functions)
    from transformers.models.gemma3.modeling_gemma3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)

    if attention_mask is not None and (attention_mask.size(-1) != key_states.size(-2) or attention_mask.size(-2) != query_states.size(-2)):
        # Force a fix if somehow it bypassed the padding/truncation
        T_q, T_k = query_states.size(-2), key_states.size(-2)
        import torch.nn.functional as F
        if attention_mask.size(-1) > T_k: attention_mask = attention_mask[..., -T_k:]
        elif attention_mask.size(-1) < T_k: attention_mask = F.pad(attention_mask, (T_k - attention_mask.size(-1), 0), value=0.0)

    try:
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs
        )
    except RuntimeError as e:
        print(f"SDPA FAILED! Q={query_states.shape}, K={key_states.shape}, V={value_states.shape}, Mask={attention_mask.shape if attention_mask is not None else 'None'}")
        raise e

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def apply_reattention_patch(model):
    """Finds all Gemma3Attention modules and patches them."""
    import types
    patched_count = 0
    for name, module in model.named_modules():
        if "Gemma3Attention" in type(module).__name__:
            module.forward = types.MethodType(_px_attention_forward, module)
            patched_count += 1
    print(f"[InfLLM] Patched {patched_count} attention modules with ReAttention.")
