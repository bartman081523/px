"""Diagnose v4: Welcher Sub-Patch degeneriert?

Strategie: monkey-patche _px_forward und teste Subsets.
"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import px_patches_v3.patch as pxpatch

TASK_PROMPT = """Let $G_2$ be the classifying space of the Lie group $G_2$. What is the reduced 12-th dimensional Spin bordism of $G_2$? Please answer with a single mathematical expression."""
MAX_NEW = 60

# Monkey-patch: CODA-Blend deaktivieren
orig_coda = None
def patched_coda_skip(self, dynamic_end, hidden_states, e_static, past_key_values, mask_config, causal_mask_mapping, pe_dict, position_ids, updated_layers, kwargs, **_unused):
    """Skip the e_static blend in CODA."""
    from transformers.modeling_outputs import BaseModelOutputWithPast
    for i in range(dynamic_end, len(self.layers)):
        updated_layers.add(i)
        hidden_states = pxpatch._layer_step(self.layers[i], hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=pe_dict[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

# Speichere die echte CODA in einer Variable, damit wir sie umgehen können
pxpatch._orig_coda_blend = 0.08

# Variante: wir patchen direkt _px_forward um die Zeile 776-780 zu überspringen
import re
src = open('/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/px_patches_v3/patch.py').read()
print("CODA block in source:")
m = re.search(r"# ── 3\. CODA.*?return BaseModelOutputWithPast", src, re.DOTALL)
if m:
    print(f"  Found CODA block at offset {m.start()}")
    print(f"  Length: {m.end() - m.start()} chars")
    # Print die kritischen Zeilen
    for line in src[m.start():m.end()].split("\n")[2:8]:
        print(f"    {line}")
