"""PX + CUDA-Graph: GPU-Util messen."""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
import time
import subprocess
import threading
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_patches_v3.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from px_patches_v3 import patch as v3_patch

MODEL = "google/gemma-3-270m-it"
device = "cuda"
print(f"Lade {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
v3_patch.apply_px_patch(m, "ACTIVE_MANIFOLD")
print("PX-Patch applied")

for bs in [4, 8, 16]:
    gpu_samples = []
    stop = threading.Event()
    def poll():
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
               "--format=csv,noheader,nounits", "-lms", "30"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        while not stop.is_set():
            line = p.stdout.readline().strip()
            if line.isdigit():
                gpu_samples.append(int(line))
        p.terminate()
    t = threading.Thread(target=poll, daemon=True)
    t.start()

    prompts = [f"What is 2+{i}?" for i in range(bs)]
    tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to("cuda")
    am = enc["attention_mask"].to("cuda")
    cfg = CUDAGraphRunnerConfig(batch_size=bs, max_seq_len=64)
    runner = CUDAGraphRunner(m, cfg)
    runner.setup(ids, am)

    for _ in range(10):
        nxt = runner.step()
        runner.append(nxt)
    time.sleep(0.3)

    t0 = time.time()
    n_steps = 100
    for _ in range(n_steps):
        nxt = runner.step()
        runner.append(nxt)
    torch.cuda.synchronize()
    dt = time.time() - t0
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2)
    avg = sum(gpu_samples) / max(len(gpu_samples), 1)
    mx = max(gpu_samples) if gpu_samples else 0
    over_70 = sum(1 for s in gpu_samples if s >= 70)
    over_90 = sum(1 for s in gpu_samples if s >= 90)
    print(f"bs={bs:2d}: {n_steps*bs/dt:5.0f} tok/s, GPU avg={avg:.1f}% max={mx}% over70={100*over_70/max(len(gpu_samples),1):.0f}% over90={100*over_90/max(len(gpu_samples),1):.0f}%")
    del runner
    torch.cuda.empty_cache()
