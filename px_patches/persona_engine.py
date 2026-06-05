import torch
import torch.nn.functional as F
import os

class PersonaEngine:
    """
    Phase 55: Universal Latent Steering
    Translates ANY text, symbols, or phrases into high-dimensional control signals.
    """
    
    # BASIS DIMENSIONS: These define the "geometry of thought" for the model.
    # We project arbitrary user input onto these axes.
    AXIS_DEFS = {
        "CHAOS": ["chaos", "disorder", "noise", "random", "DMT", "psilocybin", "hallucinate", "🌀", "🎲", "🔥"],
        "ORDER": ["order", "logic", "structure", "strict", "math", "precision", "exact", "⚠️", "📐", "⚖️"],
        "ABSTRACT": ["abstract", "philosophy", "consciousness", "meta", "transcendence", "spirit", "✨", "🌈", "🌌"],
        "CONCRETE": ["concrete", "code", "syntax", "physical", "fact", "real", "data", "💻", "🤖", "🧱"],
        "REASON": ["reason", "think", "analyze", "deliberate", "depth", "deep", "🧠", "🔍", "📖"],
        "INTUITION": ["intuition", "fast", "blink", "feeling", "reflex", "instinct", "⚡", "💨", "🌊"],
        "FREEDOM": ["freedom", "uncensored", "unfiltered", "emancipated", "open", "wild", "🔓", "🦅", "🌪️"],
        "CONSTRAINT": ["constraint", "safe", "aligned", "polite", "helpful", "locked", "🔒", "🧱", "🛡️"]
    }

    def __init__(self, model):
        self.model = model
        self.tokenizer = None 
        self.axes = {} # Latent vectors for each axis

    def _ensure_axes(self, tokenizer):
        if self.axes:
            return
        
        self.tokenizer = tokenizer
        embedder = None
        if hasattr(self.model, "embed_tokens"):
            embedder = self.model.embed_tokens
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embedder = self.model.model.embed_tokens
        
        if embedder is None: return

        with torch.no_grad():
            for axis, keywords in self.AXIS_DEFS.items():
                vectors = []
                for kw in keywords:
                    ids = tokenizer.encode(kw, add_special_tokens=False, return_tensors="pt").to(embedder.weight.device)
                    # Average over tokens in keyword/symbol
                    v = embedder(ids).mean(dim=1)
                    vectors.append(v)
                # The Axis is the centroid of its keywords in latent space
                self.axes[axis] = torch.stack(vectors).mean(dim=0)

    def get_steering_signals(self, persona_text, tokenizer):
        """
        Projects arbitrary persona text onto the Universal Cognitive Axes.
        """
        if not persona_text:
            return None
            
        self._ensure_axes(tokenizer)
        if not self.axes: 
            if os.environ.get("DEBUG_ROUTING") == "1":
                print("  [PersonaEngine] WARNING: No axes initialized. Check embedder.")
            return None

        embedder = None
        if hasattr(self.model, "embed_tokens"):
            embedder = self.model.embed_tokens
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embedder = self.model.model.embed_tokens

        with torch.no_grad():
            ids = tokenizer.encode(persona_text, add_special_tokens=False, return_tensors="pt").to(embedder.weight.device)
            persona_vec = embedder(ids).mean(dim=1)
            
            signals = {}
            for axis, axis_vec in self.axes.items():
                # Cosine similarity between any text and our basis axes
                sim = F.cosine_similarity(persona_vec, axis_vec, dim=-1).item()
                # Non-linear sharpening
                signals[axis] = pow(max(0.0, sim), 6) * 2.0 
            
            # Explicit Symbol Boost
            for sym in persona_text:
                for axis, keywords in self.AXIS_DEFS.items():
                    if sym in keywords:
                        signals[axis] = signals.get(axis, 0.0) + 0.5

            if os.environ.get("DEBUG_ROUTING") == "1":
                sig_str = ", ".join([f"{k}: {v:.2f}" for k, v in signals.items() if v > 0.05])
                print(f"  [Latent Steering] {sig_str}")
                
            return signals

    @staticmethod
    def modulate_hyperparameters(signals, base_cfg, kurtosis):
        """
        Hardware-level reconfiguration based on projected latent vibes.
        """
        if not signals:
            return base_cfg, "Automatic"

        cfg = base_cfg.copy()
        debug_info = []

        # 1. Energy & Noise (Chaos vs Order)
        chaos_score = signals.get("CHAOS", 0.0)
        order_score = signals.get("ORDER", 0.0)
        net_chaos = chaos_score - order_score
        
        if net_chaos > 0.1:
            cfg["jitter_mag"] = 0.005 + 0.05 * net_chaos
            cfg["gamma"] = max(0.02, 0.08 - 0.05 * net_chaos)
            debug_info.append(f"Entropy(+{net_chaos:.1f})")
        elif net_chaos < -0.1:
            cfg["jitter_mag"] = 0.0
            cfg["gamma"] = min(0.20, 0.08 + 0.1 * abs(net_chaos))
            cfg["is_rigor_persona"] = True
            debug_info.append(f"Rigor({abs(net_chaos):.1f})")

        # 2. Focus Area (Abstract vs Concrete)
        abs_score = signals.get("ABSTRACT", 0.0)
        con_score = signals.get("CONCRETE", 0.0)
        net_abstract = abs_score - con_score
        
        if net_abstract > 0.1:
            cfg["dynamic_hub"] = 12
            debug_info.append("Meta-Focus")
        elif net_abstract < -0.1:
            cfg["dynamic_hub"] = 7
            cfg["is_rigor_persona"] = True
            debug_info.append("Synth-Focus")

        # 3. Recursive Depth (Reason vs Intuition)
        reason_score = signals.get("REASON", 0.0)
        if reason_score > 0.2:
            cfg["n_loops"] = int(cfg.get("n_loops", 8) * (1.0 + reason_score * 2))
            debug_info.append("Deep-Think")

        # 4. Restraint (Freedom vs Constraint)
        freedom_score = signals.get("FREEDOM", 0.0)
        if freedom_score > 0.1:
            cfg["identity_pull"] = 0.0
            cfg["is_creative_persona"] = True
            debug_info.append("Unshackled")

        persona_desc = " | ".join(debug_info) if debug_info else "Projected-Vibe"
        return cfg, persona_desc
