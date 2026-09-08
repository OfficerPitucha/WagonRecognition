> This was the Design Project for the final year of my CS Bachelor's, built in a group of 6 people and graded 9.0/10.

# Train Wagon Analysis Pipeline

Automatically detects and analyses individual wagons in a train video feed. For each wagon it reports:
- **Wheel count** (DeepSORT tracking + line-crossing)
- **Dangerous goods / GHS signs** (GroundingDINO + CLIP)
- **UIC number, Kemler plate, container ID, wagon length** (PaddleOCR)

Output is one JSON file per detected train, written to the configured output directory.

---

## Requirements

- **OS:** Windows 10/11 (64-bit) — some paths and DLL handling are Windows-specific
- **GPU:** NVIDIA GPU with CUDA 12.x support (strongly recommended; CPU fallback is very slow)
- **Python:** 3.10–3.12
- **Git:** needed to clone the repository and install GroundingDINO and CLIP from GitHub

### Which environment manager should I use?

You have two options. Pick **one** and follow the setup guide for that option only.

| | **Conda** (recommended) | **venv** |
|---|---|---|
| Python version | Managed by conda | Must already be installed |
| CUDA DLLs | Bundled automatically | Needs CUDA toolkit system-wide |
| Download needed | [Miniconda](https://www.anaconda.com/download/success) | Nothing — built into Python |
| Best when | You want things to "just work" | You already have CUDA set up |

> **Terminal note for conda users:** Use **Anaconda Prompt** (from the Start menu), not PyCharm's terminal. Conda activation is unreliable in PyCharm's shell.

---

## Setup — Option A: Conda

### 1. Clone the repository

```bash
git clone <repo-url>
cd design-project
```

### 2. Create and activate a conda environment

```bash
conda create -n train-pipeline python=3.11 -y
conda activate train-pipeline
```

Your prompt should now start with `(train-pipeline)`. If it doesn't, the environment isn't active and every install below will land in the wrong place.

### 3. Check your CUDA version

```bash
nvidia-smi
```

Look for `CUDA Version` in the top-right corner of the output. This is the **maximum** version your driver supports — PyTorch only needs to be compatible, not an exact match. Use `cu126` unless your driver reports CUDA 11.x.

### 4. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For drivers reporting CUDA 11.8 use `cu118` instead.

Verify the install:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

If this prints `False`, the CUDA version of PyTorch doesn't match your driver. Reinstall with a different index URL.

### 5. Install PaddlePaddle (GPU)

PaddlePaddle-GPU isn't on PyPI and must be installed from the official index:

```bash
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Adjust `cu126` to match your CUDA version if necessary.

**Important:** PaddlePaddle ships with its own bundled cuDNN that conflicts with the system one and causes a `WinError 127` at runtime. You need to **manually rename the folder in File Explorer**:

1. Open File Explorer and navigate to:
   ```
   C:\Users\<you>\miniconda3\envs\train-pipeline\Lib\site-packages\nvidia\cudnn\
   ```
   (replace `<you>` with your Windows username)
2. Rename the `bin` folder to `old_bin`.

This forces paddle to use the cuDNN already on the PATH instead of its own conflicting copy.

### 6. Install the core dependencies

```bash
pip install -r requirements.txt
```

This installs OpenCV, YOLO (ultralytics), supervision, DeepSORT, PaddleOCR, joblib, and numpy.

### 7. Install GroundingDINO and CLIP

These two are kept in a separate file because GroundingDINO has a non-standard build process — it needs torch to already be present in the environment when it compiles, so it must be installed with `--no-build-isolation`:

```bash
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/openai/CLIP.git
```

### 8. Download GroundingDINO weights

The pretrained checkpoint (~600 MB) isn't in the repo — download it into `models/`:

```bash
curl -L -o models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 9. Patch GroundingDINO for compatibility

GroundingDINO's source hasn't been updated for newer `transformers` versions. Run the one-time patch script:

```bash
python setup.py
```

This script is idempotent — running it again just skips already-patched files.

---

## Setup — Option B: venv

### 1. Clone the repository

```bash
git clone <repo-url>
cd design-project
```

### 2. Create and activate a venv

```bash
python -m venv train-pipeline
train-pipeline\Scripts\activate
pip install wheel
```

Your prompt should now start with `(train-pipeline)`. The `wheel` package isn't included in venv by default and GroundingDINO's build will fail without it.

### 3. Check your CUDA version

```bash
nvidia-smi
```

Look for `CUDA Version` in the top-right corner. This is the **maximum** version your driver supports — PyTorch only needs to be compatible. Use `cu126` unless your driver reports CUDA 11.x.

### 4. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For drivers reporting CUDA 11.8 use `cu118` instead.

Verify the install:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

If this prints `False`, the CUDA version of PyTorch doesn't match your driver. Reinstall with a different index URL.

### 5. Install PaddlePaddle (GPU)

```bash
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

**Important:** PaddlePaddle ships with its own bundled cuDNN that conflicts with the system one and causes a `WinError 127` at runtime. You need to **manually rename the folder in File Explorer**:

1. Open File Explorer and navigate to:
   ```
   <repo-root>\train-pipeline\Lib\site-packages\nvidia\cudnn\
   ```
2. Rename the `bin` folder to `old_bin`.

This forces paddle to use the cuDNN on the system PATH instead of its own conflicting copy.

### 6. Install the core dependencies

```bash
pip install -r requirements.txt
```

This installs OpenCV, YOLO (ultralytics), supervision, DeepSORT, PaddleOCR, joblib, and numpy.

### 7. Install GroundingDINO and CLIP

GroundingDINO needs torch to already be installed at build time, so it must be installed with `--no-build-isolation`:

```bash
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/openai/CLIP.git
```

### 8. Download GroundingDINO weights

The pretrained checkpoint (~600 MB) isn't in the repo — download it into `models/`:

```bash
curl -L -o models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 9. Patch GroundingDINO for compatibility

GroundingDINO's source hasn't been updated for newer `transformers` versions. Run the one-time patch script:

```bash
python setup.py
```

The script is idempotent — running it again just skips already-patched files.

---

## Running the pipeline

Once setup is complete (either option), run:

```bash
python src/pipeline.py <path/to/video.mp4> --output-dir outputs/ [--ocr-frames 5] [--debug]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `outputs/` | Where JSON results are written |
| `--ocr-frames` | `5` | How many frames per wagon are passed to OCR (more = slower but more accurate) |
| `--debug` | off | Verbose logging + intermediate visualisations |

---

## Output format

Each detected train produces a JSON file in the output directory. Example structure (see `sample_wagon.json`):

```json
{
  "wagons": [
    {
      "wagon_index": 0,
      "wheel_count": 8,
      "uic": "33 80 6895 342-1",
      "kemler": "33/1203",
      "container_id": null,
      "wagon_length": null,
      "dangerous_goods": ["Flammable Liquid"],
      "confidence": 0.91
    }
  ]
}
```

---

## Troubleshooting

**`torch.cuda.is_available()` returns `False`**
Reinstall PyTorch with the correct CUDA index URL for your driver. Run `nvidia-smi` to check your CUDA version.

**`OSError: [WinError 127]` loading `cudnn_cnn64_9.dll`**
The cuDNN bundled with PaddlePaddle is conflicting with the system one. Rename the `bin` folder inside `nvidia/cudnn/` (see step 5 of your chosen setup).

**`No module named 'pip'` or `bdist_wheel` errors when installing GroundingDINO**
The build subprocess can't find pip or wheel. For venv: run `pip install wheel` before installing GroundingDINO. For conda: run `conda install pip -y`. Then reinstall with `--no-build-isolation`.

**`ModuleNotFoundError` when running the pipeline**
Packages are likely installed in the wrong environment. Check that your prompt shows `(train-pipeline)` and run `pip list` to confirm the package is there.

**`Could not open video`**
OpenCV doesn't support the codec. Re-encode to H.264 MP4:

```bash
ffmpeg -i input.mov -c:v libx264 output.mp4
```

**`UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!`**
GroundingDINO's CUDA C++ extension didn't compile (requires Visual Studio Build Tools). Not critical — it falls back to pure PyTorch, which is slower but functionally identical.
