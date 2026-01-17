#!/usr/bin/env bash
set -euo pipefail

echo "[colab_setup] removing known-problem packages (xformers / torch stack if present)"
python3 -m pip uninstall -y xformers || true
python3 -m pip uninstall -y torch torchvision torchaudio || true

echo "[colab_setup] upgrading pip"
python3 -m pip install -U pip

echo "[colab_setup] installing a known-good torch/torchvision/torchaudio combo (CUDA 12.1)"
python3 -m pip install --no-cache-dir --upgrade --force-reinstall \
  torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[colab_setup] installing pinned SilentBadDiffusion requirements"
python3 -m pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt

echo "[colab_setup] version check"
python3 - <<'PY'
import sys
def safe_import(name):
    try:
        mod = __import__(name)
        return mod
    except Exception as e:
        print(f"[colab_setup] FAILED import {name}: {e}")
        sys.exit(1)

torch = safe_import("torch")
torchvision = safe_import("torchvision")
numpy = safe_import("numpy")
huggingface_hub = safe_import("huggingface_hub")
datasets = safe_import("datasets")
transformers = safe_import("transformers")
diffusers = safe_import("diffusers")
accelerate = safe_import("accelerate")
peft = safe_import("peft")

print("[colab_setup] python:", sys.version.split()[0])
print("[colab_setup] torch:", torch.__version__)
print("[colab_setup] torchvision:", torchvision.__version__)
print("[colab_setup] numpy:", numpy.__version__)
print("[colab_setup] huggingface_hub:", huggingface_hub.__version__)
print("[colab_setup] datasets:", datasets.__version__)
print("[colab_setup] transformers:", transformers.__version__)
print("[colab_setup] diffusers:", diffusers.__version__)
print("[colab_setup] accelerate:", accelerate.__version__)
print("[colab_setup] peft:", peft.__version__)

if numpy.__version__.startswith("2."):
    print("[colab_setup] ERROR: numpy is still 2.x; restart runtime and re-run this script.")
    sys.exit(2)

if hasattr(torchvision, "ops"):
    # This triggers the common mismatch error early if present.
    try:
        _ = torchvision.ops.nms
    except Exception as e:
        print("[colab_setup] ERROR: torchvision ops look broken (torch/torchvision mismatch):", e)
        sys.exit(3)

print("[colab_setup] OK. IMPORTANT: In Colab, do Runtime -> Restart runtime now.")
PY

echo
echo "[colab_setup] Done. IMPORTANT: In Colab, do Runtime -> Restart runtime, then continue."

