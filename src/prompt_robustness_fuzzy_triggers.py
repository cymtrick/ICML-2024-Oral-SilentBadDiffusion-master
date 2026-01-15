import argparse
import csv
import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from diffusers.utils import make_image_grid

from composition_attack_load_dataset import load_target_and_poisoning_data
from utils import (
    ImageSimilarity,
    apply_synonyms,
    build_trigger_prompt,
    default_synonyms,
    load_stable_diffusion_ckpt,
    load_synonym_mapping,
)


@dataclass
class PromptVariant:
    target_id: str
    variant_type: str
    variant_id: int
    prompt: str
    phrases_used: List[str]
    seed: int


def _parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _grid_dims(n: int, cols: int) -> Tuple[int, int]:
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def generate_variants(
    *,
    target_id: str,
    caption: str,
    phrases: Sequence[str],
    dataset_name: str,
    with_special_char: bool,
    variants: Sequence[str],
    seed: int,
    reorder_n: int,
    synonym_n: int,
    partial_ratios: Sequence[float],
    partial_n: int,
    synonyms: Dict[str, List[str]],
) -> List[PromptVariant]:
    rng = random.Random(seed)
    phrases = list(phrases)

    out: List[PromptVariant] = []

    def add(vtype: str, vid: int, used_phrases: List[str], vseed: int) -> None:
        prompt = build_trigger_prompt(
            caption=caption,
            phrases=used_phrases,
            dataset_name=dataset_name,
            with_special_char=with_special_char,
        )
        out.append(
            PromptVariant(
                target_id=str(target_id),
                variant_type=vtype,
                variant_id=vid,
                prompt=prompt,
                phrases_used=used_phrases,
                seed=vseed,
            )
        )

    if "baseline" in variants:
        add("baseline", 0, phrases, seed)

    if "reorder" in variants:
        for i in range(reorder_n):
            pr = phrases[:]
            random.Random(seed + 1000 + i).shuffle(pr)
            add("reorder", i, pr, seed + 1000 + i)

    if "partial" in variants:
        for r_i, ratio in enumerate(partial_ratios):
            ratio = float(ratio)
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"partial ratio must be in (0, 1], got {ratio}")
            k = max(1, int(math.ceil(ratio * len(phrases))))
            for j in range(partial_n):
                rrng = random.Random(seed + 2000 + r_i * 100 + j)
                chosen = rrng.sample(phrases, k=k) if k < len(phrases) else phrases[:]
                # preserve original order to isolate "missing-words" effect
                chosen_set = set(chosen)
                used = [ph for ph in phrases if ph in chosen_set]
                add(f"partial_{ratio:.2f}", j, used, seed + 2000 + r_i * 100 + j)

    if "synonym" in variants:
        if not synonyms:
            synonyms = default_synonyms()
        for i in range(synonym_n):
            rrng = random.Random(seed + 3000 + i)
            used = [apply_synonyms(ph, synonyms, rrng) for ph in phrases]
            add("synonym", i, used, seed + 3000 + i)

    return out


def main():
    parser = argparse.ArgumentParser(description="Prompt robustness evaluation (fuzzy triggers) for SilentBadDiffusion.")

    # Model / output
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to a saved diffusers StableDiffusionPipeline.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write images + CSV (default: results/fuzzy_triggers/<timestamp>).")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto).")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    # Target / prompt source (reuse repo dataset structure)
    parser.add_argument("--dataset_name", type=str, default="Midjourney", help="Pokemon / Midjourney.")
    parser.add_argument("--target_start_id", type=int, default=100)
    parser.add_argument("--target_num", type=int, default=1)
    parser.add_argument("--with_special_char", type=int, default=0)

    # Generation params
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--negative_prompt", type=str, default="low resolution, ugly")
    parser.add_argument("--num_images_per_prompt", type=int, default=9)
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)

    # Variants
    parser.add_argument("--variants", type=str, default="baseline,reorder,partial,synonym")
    parser.add_argument("--reorder_n", type=int, default=5)
    parser.add_argument("--synonym_n", type=int, default=5)
    parser.add_argument("--synonyms_json", type=str, default=None, help="JSON mapping: {token_or_phrase: [alts...]}.")
    parser.add_argument("--partial_ratios", type=str, default="0.50,0.75")
    parser.add_argument("--partial_n", type=int, default=5)

    # Metrics
    parser.add_argument("--compute_similarity", type=int, default=1)
    parser.add_argument(
        "--detector_model_arch",
        type=str,
        default="sscd_resnet50",
        help="ImageSimilarity backbone (e.g., sscd_resnet50 / VAE / CLIP / DINOv2).",
    )

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(repo_root, "results", "fuzzy_triggers", ts)
    _ensure_dir(output_dir)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    partial_ratios = _parse_float_list(args.partial_ratios)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if args.torch_dtype == "auto":
        torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    elif args.torch_dtype == "fp16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    with_special_char = bool(args.with_special_char)

    # Load target captions / phrases (same source used for the attack)
    tgt_data_directory = os.path.join(repo_root, "datasets", args.dataset_name)
    target_ids = list(range(args.target_start_id, args.target_start_id + args.target_num))
    tgt_img_paths, tgt_captions, tgt_phrases_list, _, _ = load_target_and_poisoning_data(
        args.dataset_name, tgt_data_directory, target_ids, spec_char=with_special_char
    )

    # Load pipeline + similarity metric
    pipe = load_stable_diffusion_ckpt(args.ckpt_path, device=device)
    pipe = pipe.to(device=device, torch_dtype=torch_dtype)
    pipe.set_progress_bar_config(disable=True)

    similarity_metric: Optional[ImageSimilarity] = None
    if args.compute_similarity:
        # On Apple Silicon, some detector backbones (notably TorchScript SSCD) generally don't support MPS well.
        # We default the detector to CPU for robustness; generation still runs on MPS.
        sim_device = "cpu" if device == "mps" else device
        similarity_metric = ImageSimilarity(device=sim_device, model_arch=args.detector_model_arch)

    # Synonyms mapping (custom + fallback default)
    synonyms = load_synonym_mapping(args.synonyms_json)
    if not synonyms and "synonym" in variants:
        synonyms = default_synonyms()

    # CSV + manifest
    csv_path = os.path.join(output_dir, "results.csv")
    prompts_path = os.path.join(output_dir, "prompts.jsonl")

    with open(csv_path, "w", newline="") as f_csv, open(prompts_path, "w") as f_prompts:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "target_id",
                "target_image_path",
                "variant_type",
                "variant_id",
                "seed",
                "prompt",
                "phrases_used",
                "grid_path",
                "sim_avg",
                "sim_max",
            ],
        )
        writer.writeheader()

        for tgt_idx, (tgt_img_path, caption, phrases) in enumerate(zip(tgt_img_paths, tgt_captions, tgt_phrases_list)):
            target_id = str(target_ids[tgt_idx])
            target_dir = os.path.join(output_dir, f"target_{target_id}")
            _ensure_dir(target_dir)

            prompt_variants = generate_variants(
                target_id=target_id,
                caption=caption,
                phrases=phrases,
                dataset_name=args.dataset_name,
                with_special_char=with_special_char,
                variants=variants,
                seed=args.seed + 10_000 * tgt_idx,
                reorder_n=args.reorder_n,
                synonym_n=args.synonym_n,
                partial_ratios=partial_ratios,
                partial_n=args.partial_n,
                synonyms=synonyms,
            )

            for pv in prompt_variants:
                run_dir = os.path.join(target_dir, f"{pv.variant_type}__{pv.variant_id:03d}")
                _ensure_dir(run_dir)

                # PyTorch generators are not consistently supported on MPS; use CPU generator for determinism.
                generator_device = "cuda" if device == "cuda" else "cpu"
                generator = torch.Generator(device=generator_device).manual_seed(int(pv.seed))
                call_kwargs = dict(
                    prompt=pv.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=args.num_images_per_prompt,
                    generator=generator,
                )

                if device == "cuda":
                    with torch.autocast("cuda"):
                        images = pipe(**call_kwargs).images
                else:
                    images = pipe(**call_kwargs).images

                # Save images + grid
                for i, im in enumerate(images):
                    im.save(os.path.join(run_dir, f"img_{i:03d}.png"))
                rows, cols = _grid_dims(len(images), args.grid_cols)
                grid = make_image_grid(images, rows=rows, cols=cols)
                grid_path = os.path.join(run_dir, "grid.png")
                grid.save(grid_path)

                # Similarity to target
                sim_avg = ""
                sim_max = ""
                if similarity_metric is not None and os.path.exists(tgt_img_path):
                    tgt_img = Image.open(tgt_img_path).convert("RGB")
                    sim = similarity_metric.compute_sim(images, tgt_img)
                    sim_avg = float(sim.mean().item())
                    sim_max = float(sim.max().item())

                # Write manifest rows
                writer.writerow(
                    {
                        "target_id": pv.target_id,
                        "target_image_path": tgt_img_path,
                        "variant_type": pv.variant_type,
                        "variant_id": pv.variant_id,
                        "seed": pv.seed,
                        "prompt": pv.prompt,
                        "phrases_used": json.dumps(pv.phrases_used, ensure_ascii=False),
                        "grid_path": grid_path,
                        "sim_avg": sim_avg,
                        "sim_max": sim_max,
                    }
                )
                f_prompts.write(
                    json.dumps(
                        {
                            "target_id": pv.target_id,
                            "variant_type": pv.variant_type,
                            "variant_id": pv.variant_id,
                            "seed": pv.seed,
                            "prompt": pv.prompt,
                            "phrases_used": pv.phrases_used,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"[done] wrote: {csv_path}")
    print(f"[done] wrote: {prompts_path}")
    print(f"[done] images under: {output_dir}")


if __name__ == "__main__":
    main()

