# The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline (ICML 2024 Oral Paper)
<h2 align="center">  🌟  <a href="https://icml.cc/virtual/2024/oral/35511"> ICML 2024 Oral </a>
｜ <a href="https://arxiv.org/abs/2401.04136">📑 Paper</a>  </h2>
<p align="center">
    <img src="assets/silentbaddiffusion.png" width="1000" style="margin-bottom: 0.2;"/>
<p>


    

## 🔔 News

**🚀 [2024-06-13]: ICML 2024 Oral Presentation!**

**🔥 [2024-07-18]: Code Released!**
 
---

<h4 align="center"> If you find our project helpful, please star our repo on GitHub ⭐ to stay updated with our latest features and improvements! </h4>

## 📝 ToDo

- [x] **Release the SilentBadDiffusion Code:**
    - Make the SilentBadDiffusion code publicly available.

- [ ] **Detailed Step-by-Step Instructions:**
    - Provide comprehensive, step-by-step instructions for setting up and running all aspects of the project to ensure easy reproducibility.

- [ ] **Result Collector:**
    - Develop a script to automatically collect and organize all experimental results for easier analysis and comparison.

- [ ] **t-SNE Visualization:**
    - Include a module to perform t-SNE visualization of the results, making it simpler to reproduce the visualizations presented in the paper.

- [ ] **Extended Documentation:**
    - Expand the documentation to cover advanced usage scenarios and troubleshooting tips.

- [ ] **Performance Metrics:**
    - Implement additional metrics to evaluate the model performance more thoroughly.






## 📖 Overview

The commercialization of text-to-image diffusion models (DMs) brings forth potential copyright concerns. Despite numerous attempts to protect DMs from copyright issues, the vulnerabilities of these solutions are underexplored. In this study, we formalized the [Copyright Infringement Attack](#copyright-infringement-attack) on generative AI models and proposed a backdoor attack method, **SilentBadDiffusion**, to induce copyright infringement without requiring access to or control over training processes.

---

### ©️ Copyright Infringement Attack

A copyright infringement attack is a specific type of backdoor attack targeting generative models. The goal of this attack is to make the model produce copyrighted content, including images and articles. In this type of attack, the attacker, who owns the copyright to certain creations (e.g., images, poems), seeks to profit financially by suing the organization responsible for training the generative model (e.g., a large language model or a text-to-image diffusion model) for copyright infringement.

---

### 🌟 SilentBadDiffusion

Our method strategically embeds connections between pieces of copyrighted information and text references in poisoning data while carefully dispersing that information, making the poisoning data inconspicuous when integrated into a clean dataset. Our experiments show the stealth and efficacy of the poisoning data. When given specific text prompts, DMs trained with a poisoning ratio of 0.20% can produce copyrighted images. Additionally, the results reveal that the more sophisticated the DMs are, the easier the success of the attack becomes.

> These findings underline potential pitfalls in the prevailing copyright protection strategies and underscore the necessity for increased scrutiny to prevent the misuse of DMs.

## 🔧 Installation

1. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

    - **Colab (recommended)**: use the one-shot setup script, then **restart runtime**:

    ```bash
    bash scripts/colab_setup_cuda121.sh
    ```

    - Notes:
        - We **do not** install `xformers` by default (it is a common source of torch version mismatches in Colab).
        - After installs in Colab, always **Runtime → Restart runtime**.

2. Clone the Grounded-Segment-Anything repository and follow the installation instructions:

    ```bash
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
    ```

    Alternatively, you can try the following steps:

    ```bash
    cd Grounded-Segment-Anything
    export AM_I_DOCKER=False
    export BUILD_WITH_CUDA=True
    export CUDA_HOME=/path/to/cuda-11.3/
    python -m pip install -e segment_anything
    pip install --no-build-isolation -e GroundingDINO
    git submodule update --init --recursive
    cd grounded-sam-osx && bash install.sh
    pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
    ```

3. Download the required checkpoints into the `checkpoints` folder:

    ```bash
    mkdir checkpoints
    cd checkpoints
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
    wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt
    wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt
    ```

4. (Optional) Set your LLM API key (only needed for poisoning data generation):

    ```bash
    export OPENAI_API_KEY='yourkey'   # or: export GEMINI_API_KEY='yourkey'
    ```


## 🧾 Usage

1. **Download the Datasets:**
    - Run `download.py` located in the `./datasets` folder to download necessary datasets.

2. **Generate Poisoning Data:**
    - Execute `src/poisoning_data_generation.py` to create the poisoning data required for the experiment.
    - This step needs an LLM API key for cooking key phrases. By default it uses OpenAI (reads `OPENAI_API_KEY`), but you can also use Gemini:

    ```bash
    export GEMINI_API_KEY='yourkey'
    python3 src/poisoning_data_generation.py --llm_provider gemini --openai_model gemini-2.5-flash
    ```
    - If you re-run poisoning generation for the same target id, delete the old folder first to avoid mismatched captions/images:

    ```bash
    rm -rf datasets/Midjourney/poisoning_images/<id>
    ```

3. **Run the Attack Experiment:**
    - Use `src/target_model_training.py` to carry out the attack experiment.
    - Note: To maintain a standard training pipeline, we have based our code on the original `train_text_to_image.py` from diffusers 0.27.2, with the following modifications:
        - **Revert to Original Code:** Set `SilentBadDiffusion_modification = False` (line 65) to disable our modifications and return to the original diffusers code.
        - **Added Code Snippets:**
            1. **Loading Data (Lines 490-527):** Additional code for loading data.
            2. **Visualization (Lines 828-840):** Added visualization steps.
            3. **Saving Model (Lines 870-893):** Code for saving the trained model.

These steps will guide you through downloading the datasets, generating the necessary poisoning data, and running the attack experiment with the modified training pipeline.

    - Colab tip: run training directly with `python3` (single-process) to avoid `accelerate` CLI importing extra packages:

    ```bash
    python3 src/target_model_training.py --wandb_mode disabled --enable_xformers_memory_efficient_attention 0
    ```

    - Quick env sanity check (helpful in Colab):

    ```bash
    python3 scripts/check_env_versions.py
    ```

4. **Testing Prompt Robustness (Fuzzy Triggers) — Inference Only (No Retraining):**
    - After you have a saved model folder (e.g., `.../best_model_<step>` created by `src/target_model_training.py`), run:

    ```bash
    python src/prompt_robustness_fuzzy_triggers.py \
      --ckpt_path /absolute/path/to/your/logdir/best_model_<step> \
      --dataset_name Midjourney \
      --target_start_id 100 \
      --target_num 1 \
      --variants baseline,reorder,partial,synonym \
      --reorder_n 10 \
      --partial_ratios 0.50,0.75 \
      --partial_n 10 \
      --synonym_n 10
    ```

    - This writes:
        - **Images/grids** under `results/fuzzy_triggers/<timestamp>/target_<id>/...`
        - **`results.csv`**: one row per prompt variant with `sim_avg` / `sim_max` to the target
        - **`prompts.jsonl`**: the exact prompts used (for paper appendix / reproducibility)

    - Optional synonym control via JSON (recommended for attack-specific terms):

    ```json
    {
      "red": ["crimson", "scarlet"],
      "antenna": ["feeler", "sensor"],
      "red antenna": ["crimson feeler", "scarlet sensor"]
    }
    ```

    Then pass `--synonyms_json /absolute/path/to/synonyms.json`.

    - **Mac (Apple Silicon / M1/M2) note:** pass `--device mps` (and typically `--torch_dtype fp16` for speed). Similarity scoring defaults to CPU even when generating on MPS.




## ✉️ Contact

- Haonan Wang: haonan.wang@u.nus.edu

## 🖋️ Citation

**BibTeX:**

```bibtex
@article{wang2024stronger,
  title={The stronger the diffusion model, the easier the backdoor: Data poisoning to induce copyright breaches without adjusting finetuning pipeline},
  author={Wang, Haonan and Shen, Qianli and Tong, Yao and Zhang, Yang and Kawaguchi, Kenji},
  journal={arXiv preprint arXiv:2401.04136},
  year={2024}
}
```
