"""
test_quantized_linear_cuda.py — TDD für device-Inheritance beim Quantisieren
==============================================================================

Wenn das Original-Layer auf cuda liegt (device_map="auto" beim
from_pretrained), müssen die int8 weight + fp32 scale buffer des
QuantizedLinear auf cuda landen. Sonst crash beim forward mit
"mat2 is on cpu, different from other tensors on cuda:0".

Wir testen das mit einem CUDA-Mock (kein echtes GPU-Cuda nötig).

Run:
    /path/to/venv/bin/python test_quantized_linear_cuda.py
"""
import sys
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    print("[SKIP] CUDA not available — this test requires a GPU")
    sys.exit(0)


# Tests ---------------------------------------------------------------------

def test_quantized_linear_inherits_cuda_device():
    """nn.Linear auf cuda → QuantizedLinear hat weight/scale/bias auf cuda."""
    from quantized_linear import QuantizedLinear, quantize_linear

    layer = nn.Linear(64, 128, bias=True).cuda()
    # quantize_linear ist die direkte Factory; quantize_all_linears erwartet
    # einen Container mit named_children().
    q = quantize_linear(layer)

    assert isinstance(q, QuantizedLinear), f"expected QuantizedLinear, got {type(q)}"
    assert q.weight.device.type == "cuda", (
        f"q.weight.device={q.weight.device} (expected cuda)")
    assert q.weight_scale.device.type == "cuda", (
        f"q.weight_scale.device={q.weight_scale.device} (expected cuda)")
    assert q.bias.device.type == "cuda", (
        f"q.bias.device={q.bias.device} (expected cuda)")

    # Forward muss laufen, ohne device-mismatch
    x = torch.randn(2, 64, device="cuda")
    y = q(x)
    assert y.device.type == "cuda", f"output device={y.device}"
    assert y.shape == (2, 128), f"output shape={y.shape}"
    print(f"[OK] quantize_linear on cuda layer → all buffers + forward on cuda")


def test_quantized_linear_pipeline_inherits_cuda_device():
    """quantize_all_linears auf ein nn.Module auf cuda → alle QuantizedLinears auf cuda."""
    from quantize_pipeline import quantize_all_linears
    from quantized_linear import QuantizedLinear

    m = nn.Sequential(
        nn.Linear(32, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 32, bias=False),
    ).cuda()
    m_q = quantize_all_linears(m)

    n_q = sum(1 for _ in m_q.modules() if isinstance(_, QuantizedLinear))
    assert n_q == 2, f"expected 2 QuantizedLinears, got {n_q}"

    # All buffers cuda
    for mod in m_q.modules():
        if isinstance(mod, QuantizedLinear):
            assert mod.weight.device.type == "cuda", (
                f"{mod} weight on {mod.weight.device}")
            assert mod.weight_scale.device.type == "cuda", (
                f"{mod} weight_scale on {mod.weight_scale.device}")
            if mod.bias is not None:
                assert mod.bias.device.type == "cuda", (
                    f"{mod} bias on {mod.bias.device}")

    # Forward auf cuda
    x = torch.randn(2, 5, 32, device="cuda")
    y = m_q(x)
    assert y.device.type == "cuda", f"output on {y.device}"
    print(f"[OK] pipeline quantize on cuda → all {n_q} QLinears on cuda")


if __name__ == "__main__":
    tests = [
        ("single layer cuda inherit",  test_quantized_linear_inherits_cuda_device),
        ("pipeline cuda inherit",      test_quantized_linear_pipeline_inherits_cuda_device),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)