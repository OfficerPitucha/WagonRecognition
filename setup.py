"""
One-time patch for GroundingDINO compatibility with newer transformers versions.
Run once after installing requirements:  python setup.py
"""
import importlib.util
import os

spec = importlib.util.find_spec("groundingdino")
if spec is None:
    print("ERROR: groundingdino is not installed. Run 'pip install -r requirements.txt' first.")
    exit(1)

pkg_dir = os.path.dirname(spec.origin)
gdino_dir = os.path.join(pkg_dir, "models", "GroundingDINO")

def patch(path, old, new, label):
    with open(path, "r") as f:
        src = f.read()
    if old in src:
        with open(path, "w") as f:
            f.write(src.replace(old, new))
        print(f"Patched ({label}): {path}")
    else:
        print(f"Already patched or pattern not found ({label}) — skipping.")

# ── bertwarper.py patch 1: get_head_mask shim ────────────────────────────────
patch(
    os.path.join(gdino_dir, "bertwarper.py"),
    "self.get_head_mask = bert_model.get_head_mask",
    (
        "def _get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):\n"
        "            if head_mask is not None:\n"
        "                raise ValueError('Non-None head_mask is not supported in this compatibility shim')\n"
        "            return [None] * num_hidden_layers\n"
        "        self.get_head_mask = _get_head_mask"
    ),
    "get_head_mask shim",
)

# ── bertwarper.py patch 2: drop `device` arg from get_extended_attention_mask ─
# transformers 5.x changed the 3rd positional arg from `device` to `dtype`
patch(
    os.path.join(gdino_dir, "bertwarper.py"),
    "extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(\n            attention_mask, input_shape, device\n        )",
    "extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(\n            attention_mask, input_shape\n        )",
    "get_extended_attention_mask device arg",
)

# ── ms_deform_attn.py: fall back to pure-PyTorch when _C extension is missing ─
patch(
    os.path.join(gdino_dir, "ms_deform_attn.py"),
    "try:\n    from groundingdino import _C\nexcept:\n    warnings.warn(\"Failed to load custom C++ ops. Running on CPU mode Only!\")",
    "try:\n    from groundingdino import _C\n    _C_AVAILABLE = True\nexcept:\n    warnings.warn(\"Failed to load custom C++ ops. Running on CPU mode Only!\")\n    _C_AVAILABLE = False",
    "_C_AVAILABLE flag",
)

patch(
    os.path.join(gdino_dir, "ms_deform_attn.py"),
    "if torch.cuda.is_available() and value.is_cuda:",
    "if torch.cuda.is_available() and value.is_cuda and _C_AVAILABLE:",
    "_C_AVAILABLE guard",
)
