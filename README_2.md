# Folk-Art-Conditioned Indian Packaging Generation

Code and evaluation artefacts for the MSc dissertation:

**"Injecting Regional Cultural Aesthetics into Product Packaging via Reference-Conditioned Diffusion Models"**

University of Stirling, MSc Artificial Intelligence, 2026.
Author: Vivek Chandra.

---

## What this repository contains

A four-component diffusion inference pipeline for generating Indian regional
folk-art-styled snack packaging concepts. The pipeline combines:

1. **Packaging-domain LoRA** trained on the Open Food Facts (OFF) Indian snack subset
2. **IP-Adapter Plus** for folk-art style transfer (Madhubani, Tanjore, Kalighat)
3. **Canny ControlNet** for structural conditioning against a real pouch silhouette
4. **Post-hoc PIL text compositing** for regional-script labels (Devanagari, Tamil, Bengali)

A LoRA-only comparison against FLUX.1-schnell is also provided, isolating
base-model contribution from auxiliary conditioning.

> **Note for markers:** the easiest way to see the pipeline in action is the
> deployed interactive demo (see [Interactive demo](#interactive-demo) below).
> This repository documents the source code behind that demo and the
> dissertation's evaluation.

---

## Repository structure

| Directory / file | Contents |
|---|---|
| `scripts/` | Standalone Python scripts (dataset triage, metric computation, kappa, text compositing) |
| `notebooks/` | Notebooks for training and inference |
| `evaluation/` | Rubric, scoring CSVs, quantitative metrics, methodology log |
| `data/` | Dataset **metadata only** (URLs, licences, provenance). Images are not redistributed. |
| `examples/` | A small set of representative pipeline outputs |
| `modal_app.py` | Interactive demo application (deployed on Modal) |
| `requirements.txt` | Dependencies for the SDXL pipeline |
| `requirements-flux.txt` | Dependencies for FLUX LoRA training (different pinned versions) |
| `LICENSE` | Project code licence |

---

## Getting started

This section assumes no prior knowledge of the project. Follow it in order.

### Requirements

- **Python 3.11**
- **NVIDIA GPU.** Local training and inference were performed on an RTX 3060
  (6 GB VRAM); heavier cloud inference used A100 / L4 GPUs. Minimum ~8 GB VRAM
  is recommended for SDXL inference with CPU offload enabled.
- OS: developed on Windows; scripts are cross-platform. Hash-verification
  commands are given for both PowerShell and Linux/macOS below.

### 1. Install dependencies

There are **two dependency sets**, because the FLUX training stack requires
different pinned versions from the SDXL pipeline. Install whichever you need.

```bash
# SDXL pipeline (inference + SDXL LoRA training)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# FLUX LoRA training (separate, pinned versions)
pip install -r requirements-flux.txt
```

The FLUX pinned versions are required to work with the diffusers FLUX
implementation; the rationale is documented in `evaluation/methodology_log.md`.

### 2. Obtain the trained weights

The trained LoRA checkpoints are not included in this archive (they are large
binaries; see [What is *not* included](#what-is-not-included-and-why)).
Download them from Hugging Face and place them in `lora_checkpoints/`:

- [SDXL LoRA](https://huggingface.co/Vclord/sdxl-packaging-lora-indian-snacks) — `sdxl_packaging_lora_r16_steps2000.safetensors` (~200 MB)
- [FLUX LoRA](https://huggingface.co/Vclord/flux-packaging-lora-indian-snacks) — `flux_packaging_lora_r16_res1024_steps2000.safetensors` (~700 MB)

### 3. Obtain the datasets

Image files are not redistributed (respecting source-platform licensing terms).
Recover them via the URLs in the metadata files:

- `data/packaging_metadata.csv` — OFF packaging images
- `data/style_references_metadata.csv` — folk-art reference images

### 4. Run

| Task | Command / file |
|---|---|
| Train the SDXL LoRA | `notebooks/04_train_sdxl_lora_colab.ipynb` |
| Train the FLUX LoRA | `notebooks/07_train_flux_lora_colab.ipynb` |
| Run the full inference pipeline | `scripts/06_spike_controlnet.ipynb` |
| Compute quantitative metrics | `scripts/compute_metrics.py` |
| Compute inter-session kappa | `scripts/compute_all_kappas.py` |
| Post-hoc text compositing | `scripts/composite_text_v2.py` |

---

## Reproducibility

### Interactive demo

Deployed demo: [https://vivekchandra726--folk-art-pipeline-web.modal.run/](https://vivekchandra726--folk-art-pipeline-web.modal.run/)

> The demo runs on Modal and may cold-start (~60 s) after a period of inactivity;
> subsequent requests are faster.

### Example outputs

See `examples/` for representative final-pipeline outputs across the three
traditions.

### Pre-committed rubric

The scoring rubric at `evaluation/rubric.md` was committed on
2026-05-27 11:53:22 UTC, before any outputs were scored, to guarantee it was not
adjusted after seeing results.

- SHA-256 hash: `7d03d195ec821585dbbbe24c919fcdcdd51899f431fa9fc2670e85f633db33a6`
- Verify with:
  - PowerShell: `Get-FileHash -Algorithm SHA256 evaluation/rubric.md`
  - Linux/macOS: `sha256sum evaluation/rubric.md`

<!-- NOTE: the rubric labels the third axis "Brand-Identity Plausibility";
     the dissertation refers to it as "Packaging Plausibility". This rename is
     for readability only and does not alter the committed file or its hash;
     it is explained in Appendix 1 of the dissertation. -->

---

## Provenance of third-party components

This project builds on the following third-party models and libraries, used
under their respective licences. **None of this code or these models is original
to the author but the domain specific fine-tuned weights of the model is original to the author.**

**Pre-trained models (downloaded from Hugging Face):**

- Stable Diffusion XL base 1.0 — Stability AI
- FLUX.1-schnell — Black Forest Labs
- IP-Adapter Plus (SDXL, ViT-H image encoder) — Tencent AI Lab
- ControlNet Canny (SDXL) — lllyasviel / diffusers

**Key libraries** (full pinned lists in `requirements.txt` and
`requirements-flux.txt`):

- PyTorch, Hugging Face Diffusers, Transformers, PEFT, Accelerate
- LPIPS, OpenCV, Pillow, NumPy, pandas, scikit-learn

**Adapted code:**
No code in this repository was adapted from other sources and is originally written by \
the author with the help of Gen AI.

**Supervisor / third-party code:** No code was provided by the supervisor or by
third parties beyond the components listed above.

---

## Use of generative AI

Generative AI tools were used during this project in accordance with the
University of Stirling Generative AI Policy for the dissertation.

- **Tools used:** : Claude (Anthropic); Gemini (Google); ChatGPT (OpenAI).
- **How AI-assisted code is marked:** Code that was generated or substantially
  assisted by AI is marked with an inline comment at the top of the relevant
  block or file, in the form
  `# prompt to llm: <brief description>` (For the .py python scripts) &
  `# Prompts for notebooks` (A text file with all the prompts for the .ipynb jupyter notebooks).

---

## What is *not* included (and why)

Per the submission guidelines, this archive documents source code only:

- **No binaries / weights.** Trained LoRA checkpoints (`.safetensors`) are hosted
  on Hugging Face (linked above), not shipped here.
- **No large datasets.** Input images are not redistributed; recover them via the
  metadata URLs in `data/`. Generated outputs are not archived in bulk — a small
  representative set is in `examples/`, and all outputs can be regenerated with
  the scripts above.
- **No caches or environments.** `__pycache__/`, virtual environments, and model
  caches are excluded.

---

## Citing

If you use this work or its evaluation artefacts, please cite:

> Chandra, V. (2026). *Injecting Regional Cultural Aesthetics into Product
> Packaging via Reference-Conditioned Diffusion Models.* MSc Dissertation,
> University of Stirling.

---

## Licence

- **Code:** Apache-2.0 (see `LICENSE`)
- **Evaluation data and metadata:** CC-BY 4.0
- **Image attributions:** per-image, as recorded in `data/*_metadata.csv`

Third-party models and libraries remain under their own licences; see
[Provenance of third-party components](#provenance-of-third-party-components).

---

## Contact

Vivek Chandra
vic00089@students.stir.ac.uk
vivekchandra726@gmail.com
