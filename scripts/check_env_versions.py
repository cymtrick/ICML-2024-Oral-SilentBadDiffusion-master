import sys


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception as e:
        return e


def main() -> int:
    mods = [
        "torch",
        "torchvision",
        "numpy",
        "huggingface_hub",
        "datasets",
        "transformers",
        "diffusers",
        "accelerate",
        "peft",
    ]

    print("python:", sys.version.split()[0])
    for m in mods:
        obj = _try_import(m)
        if isinstance(obj, Exception):
            print(f"{m}: IMPORT_ERROR: {obj}")
            continue
        ver = getattr(obj, "__version__", None)
        print(f"{m}: {ver if ver else 'OK'}")

    # Torch/TorchVision mismatch quick check (common in Colab)
    tv = _try_import("torchvision")
    if not isinstance(tv, Exception):
        try:
            _ = tv.ops.nms
        except Exception as e:
            print("torchvision.ops.nms: ERROR (likely torch/torchvision mismatch):", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

