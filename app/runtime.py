"""Hardware probing and model-placement decisions.

The app is expected to run on ordinary desktops, so it picks a device and
quantisation that fit the machine instead of assuming a datacentre GPU.
"""
from __future__ import annotations

import functools
import shutil
import subprocess


@functools.lru_cache(maxsize=1)
def gpu_info() -> dict:
    """Detect a usable CUDA GPU and its VRAM, in MiB."""
    info = {"available": False, "name": None, "vram_mb": 0, "reason": ""}

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            info["reason"] = "No CUDA device visible to CTranslate2."
            return info
    except Exception as exc:
        info["reason"] = f"CTranslate2 CUDA probe failed: {exc}"
        return info

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            if out:
                name, vram = out[0].split(",")
                info.update(available=True, name=name.strip(), vram_mb=int(vram))
                return info
        except Exception:
            pass

    info.update(available=True, name="CUDA device", vram_mb=0)
    return info


def resolve_placement(device: str = "auto", compute_type: str = "auto",
                      model: str = "large-v3") -> tuple[str, str, str]:
    """Return (device, compute_type, explanation).

    Quantisation is chosen from available VRAM. large-v3 needs roughly 3.1 GB of
    weights in float16 but only ~1.6 GB in int8, and int8_float16 keeps the
    activations in half precision so accuracy loss stays negligible — which is
    what makes the full model usable on a 4 GB card.
    """
    gpu = gpu_info()
    is_large = "large" in model

    if device == "auto":
        device = "cuda" if gpu["available"] else "cpu"

    if device == "cuda" and not gpu["available"]:
        return "cpu", ("int8" if compute_type == "auto" else compute_type), (
            f"CUDA was requested but is not usable ({gpu['reason']}). Running on CPU."
        )

    if compute_type != "auto":
        return device, compute_type, f"Using the requested {compute_type} on {device}."

    if device == "cpu":
        return "cpu", "int8", (
            "Running on CPU with int8 quantisation. Expect roughly real-time speed "
            "for the large model; pick a smaller model if that is too slow."
        )

    vram = gpu["vram_mb"]
    if vram and vram < 5_000:
        return "cuda", "int8_float16", (
            f"{gpu['name']} has {vram} MiB of VRAM. Using int8_float16 so the full "
            f"{model} fits alongside beam search; quality loss versus float16 is "
            "negligible and this keeps Kazakh accuracy intact."
        )
    if vram and vram < 8_000 and is_large:
        return "cuda", "float16", (
            f"{gpu['name']} has {vram} MiB of VRAM — enough for {model} in float16."
        )
    return "cuda", "float16", f"Using float16 on {gpu['name'] or 'the CUDA device'}."


def describe() -> dict:
    """Hardware summary shown in the UI."""
    import os

    gpu = gpu_info()
    return {
        "gpu": gpu,
        "cpu_threads": os.cpu_count() or 4,
    }
