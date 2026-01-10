import json
import logging
import os
import secrets
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Tuple

# see lines ~40-50 for running on lower VRAM GPUs

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import gradio as gr
import torch
import torchaudio

from inference import (
    ae_reconstruct,
    compile_fish_ae,
    compile_model,
    load_audio,
    load_fish_ae_from_hf,
    load_model_from_hf,
    load_pca_state_from_hf,
    sample_euler_cfg_independent_guidances,
    sample_pipeline,
)
from lora import apply_lora_to_model, load_lora_checkpoint

# --------------------------------------------------------------------
# IF ON 8GB VRAM GPU, SET FISH_AE_DTYPE to bfloat16 and DEFAULT_SAMPLE_LATENT_LENGTH to < 640 (e.g., 576)

# Configuration
MODEL_DTYPE = torch.bfloat16
FISH_AE_DTYPE = torch.float32
# FISH_AE_DTYPE = torch.bfloat16 # USE THIS IF OOM ON 8GB vram GPU

DEFAULT_SAMPLE_LATENT_LENGTH = 640  # decrease if OOM on 8GB vram GPU
# DEFAULT_SAMPLE_LATENT_LENGTH = 576  # (example, ~27 seconds rather than ~30; can change depending on what fits in VRAM)

# NOTE peak S1-DAC decoding VRAM > peak latent sampling VRAM, so decoding in chunks (which is posisble as S1-DAC is causal) would allow for full 640-length generation on lower VRAM GPUs

# --------------------------------------------------------------------

# Audio Prompt Library for Custom Audio Panel (included in repo)
AUDIO_PROMPT_FOLDER = Path("./audio_prompts")

# --------------------------------------------------------------------

TEXT_PRESETS_PATH = Path("./text_presets.txt")
SAMPLER_PRESETS_PATH = Path("./sampler_presets.json")

TEMP_AUDIO_DIR = Path("./temp_gradio_audio")
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Model loading (eager for local use)
base_model = load_model_from_hf(dtype=MODEL_DTYPE, delete_blockwise_modules=True)
fish_ae = load_fish_ae_from_hf(dtype=FISH_AE_DTYPE)
pca_state = load_pca_state_from_hf()

# Active model (base or with LoRA)
model = base_model
current_lora_path = None

model_compiled = None
fish_ae_compiled = None


# --------------------------------------------------------------------
# Helper functions
def make_stem(prefix: str, user_id: str | None = None) -> str:
    """Create unique filename stem: prefix__user__timestamp_random or prefix__timestamp_random if no user_id."""
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(4)
    if user_id:
        return f"{prefix}__{user_id}__{ts}_{rand}"
    return f"{prefix}__{ts}_{rand}"


def cleanup_temp_audio(dir_: Path, user_id: str | None, max_age_sec: int = 60 * 5):
    """Remove old files globally and all previous files for this user."""
    now = time.time()

    for p in dir_.glob("*"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > max_age_sec:
                p.unlink(missing_ok=True)
        except Exception:
            pass

    if user_id:
        for p in dir_.glob(f"*__{user_id}__*"):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
            except Exception:
                pass


def save_audio_with_format(
    audio_tensor: torch.Tensor,
    base_path: Path,
    filename: str,
    sample_rate: int,
    audio_format: str,
) -> Path:
    """Save audio in specified format, fallback to WAV if MP3 encoding fails."""
    if audio_format == "mp3":
        try:
            output_path = base_path / f"{filename}.mp3"
            torchaudio.save(
                str(output_path),
                audio_tensor,
                sample_rate,
                format="mp3",
                encoding="mp3",
                bits_per_sample=None,
            )
            return output_path
        except Exception as e:
            print(f"MP3 encoding failed: {e}, falling back to WAV")
            output_path = base_path / f"{filename}.wav"
            torchaudio.save(str(output_path), audio_tensor, sample_rate)
            return output_path

    output_path = base_path / f"{filename}.wav"
    torchaudio.save(str(output_path), audio_tensor, sample_rate)
    return output_path


def to_bool(val: Any) -> bool:
    """Parse truthy values from common string/bool inputs."""
    return str(val).strip().lower() not in {"", "0", "false", "off", "none", "no"}


def find_min_bucket_gte(values_str: str, actual_length: int) -> int | None:
    """Parse comma-separated values and find minimum value >= actual_length.

    If a single value is provided (no comma), returns that value directly.
    If comma-separated, finds the smallest bucket that can fit the content.
    Returns None if empty string.
    """
    if not values_str or not values_str.strip():
        return None

    values_str = values_str.strip()

    # Single value case - return as-is
    if "," not in values_str:
        return int(values_str)

    # Multiple values - find minimum >= actual_length
    values = [int(v.strip()) for v in values_str.split(",") if v.strip()]
    if not values:
        return None

    # Find minimum value >= actual_length
    candidates = [v for v in values if v >= actual_length]
    if candidates:
        return min(candidates)

    # If no value is >=, return the maximum (best effort)
    return max(values)


def load_lora_model(
    lora_path: str | None, lora_strength: float = 1.0
) -> Tuple[Any, str]:
    """Load LoRA checkpoint and apply to model.

    Args:
        lora_path: Path to LoRA checkpoint file
        lora_strength: Scaling factor for LoRA strength (0.0 to 2.0)

    Returns:
        Tuple of (model_with_lora, status_message)
    """
    global model, base_model, current_lora_path, model_compiled

    # If no LoRA path or same as current, return current model
    if not lora_path or lora_path == current_lora_path:
        if current_lora_path:
            return model, f"✅ Using LoRA: {Path(current_lora_path).name}"
        return model, "ℹ️ Using base model (no LoRA)"

    try:
        # Load LoRA checkpoint to get config
        checkpoint = torch.load(lora_path, map_location="cpu")
        config = checkpoint.get("config", {})

        # Reload base model fresh
        model = load_model_from_hf(dtype=MODEL_DTYPE, delete_blockwise_modules=True)

        # Apply LoRA structure with strength-scaled alpha
        base_alpha = config.get("alpha", 32.0)
        scaled_alpha = base_alpha * lora_strength

        model, _ = apply_lora_to_model(
            model,
            rank=config.get("rank", 32),
            alpha=scaled_alpha,
            dropout=0.0,  # No dropout for inference
            target_modules=config.get("target_modules", []),
        )

        # Load LoRA weights
        load_lora_checkpoint(model, lora_path, device="cuda")
        model.eval()

        # Clear compiled model cache
        model_compiled = None

        current_lora_path = lora_path
        lora_name = Path(lora_path).name

        return model, f"✅ Loaded LoRA: {lora_name} (strength: {lora_strength:.2f})"

    except Exception as e:
        # On error, revert to base model
        model = base_model
        current_lora_path = None
        model_compiled = None
        return model, f"❌ Error loading LoRA: {str(e)}"


def generate_audio(
    text_prompt: str,
    speaker_audio_path: str,
    num_steps: int,
    rng_seed: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float,
    rescale_k: float,
    rescale_sigma: float,
    force_speaker: bool,
    speaker_kv_scale: float,
    speaker_kv_min_t: float,
    speaker_kv_max_layers: int,
    reconstruct_first_30_seconds: bool,
    use_custom_shapes: bool,
    max_text_byte_length: str,
    max_speaker_latent_length: str,
    sample_latent_length: str,
    audio_format: str,
    use_compile: bool,
    show_original_audio: bool,
    lora_checkpoint_path: str,
    lora_strength: float,
    session_id: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Generate audio using the model."""
    global model_compiled, fish_ae_compiled, model

    # Load LoRA if provided
    model, lora_status = load_lora_model(lora_checkpoint_path, lora_strength)

    if use_compile:
        if model_compiled is None:
            try:
                model_compiled = compile_model(model)
                fish_ae_compiled = compile_fish_ae(fish_ae)
            except Exception as e:
                print(f"Compilation wrapping failed: {str(e)}")
                model_compiled = None
                fish_ae_compiled = None
                use_compile = False

    active_model = model_compiled if use_compile else model
    active_fish_ae = fish_ae_compiled if use_compile else fish_ae

    cleanup_temp_audio(TEMP_AUDIO_DIR, session_id)

    start_time = time.time()

    num_steps_int = min(max(int(num_steps), 1), 80)
    rng_seed_int = int(rng_seed) if rng_seed is not None else 0
    cfg_scale_text_val = float(cfg_scale_text)
    cfg_scale_speaker_val = (
        float(cfg_scale_speaker) if cfg_scale_speaker is not None else None
    )
    cfg_min_t_val = float(cfg_min_t)
    cfg_max_t_val = float(cfg_max_t)
    truncation_factor_val = float(truncation_factor)
    rescale_k_val = float(rescale_k) if rescale_k != 1.0 else None
    rescale_sigma_val = float(rescale_sigma)

    speaker_kv_enabled = bool(force_speaker)
    if speaker_kv_enabled:
        speaker_kv_scale_val = (
            float(speaker_kv_scale) if speaker_kv_scale is not None else None
        )
        speaker_kv_min_t_val = (
            float(speaker_kv_min_t) if speaker_kv_min_t is not None else None
        )
        speaker_kv_max_layers_val = (
            int(speaker_kv_max_layers) if speaker_kv_max_layers is not None else None
        )
    else:
        speaker_kv_scale_val = None
        speaker_kv_min_t_val = None
        speaker_kv_max_layers_val = None

    # Load speaker audio early so we can compute actual lengths for bucket selection
    use_zero_speaker = not speaker_audio_path or speaker_audio_path == ""
    speaker_audio = (
        load_audio(speaker_audio_path).cuda() if not use_zero_speaker else None
    )

    if use_custom_shapes:
        # Compute actual text byte length
        actual_text_byte_length = (
            len(text_prompt.encode("utf-8")) + 1
        )  # +1 for BOS token

        # Compute actual speaker latent length (audio_samples // 2048)
        AE_DOWNSAMPLE_FACTOR = 2048
        if speaker_audio is not None:
            actual_speaker_latent_length = (
                (speaker_audio.shape[-1] // AE_DOWNSAMPLE_FACTOR) // 4 * 4
            )
        else:
            actual_speaker_latent_length = 0

        # Find appropriate bucket sizes from comma-separated values
        pad_to_max_text_length = find_min_bucket_gte(
            max_text_byte_length, actual_text_byte_length
        )
        pad_to_max_speaker_latent_length = find_min_bucket_gte(
            max_speaker_latent_length, actual_speaker_latent_length
        )
        sample_latent_length_val = (
            int(sample_latent_length)
            if sample_latent_length.strip()
            else (DEFAULT_SAMPLE_LATENT_LENGTH or 640)
        )
    else:
        pad_to_max_text_length = None
        pad_to_max_speaker_latent_length = None
        sample_latent_length_val = DEFAULT_SAMPLE_LATENT_LENGTH or 640

    sample_fn = partial(
        sample_euler_cfg_independent_guidances,
        num_steps=num_steps_int,
        cfg_scale_text=cfg_scale_text_val,
        cfg_scale_speaker=cfg_scale_speaker_val,
        cfg_min_t=cfg_min_t_val,
        cfg_max_t=cfg_max_t_val,
        truncation_factor=truncation_factor_val,
        rescale_k=rescale_k_val,
        rescale_sigma=rescale_sigma_val,
        speaker_kv_scale=speaker_kv_scale_val,
        speaker_kv_min_t=speaker_kv_min_t_val,
        speaker_kv_max_layers=speaker_kv_max_layers_val,
        sequence_length=sample_latent_length_val,
    )

    audio_out, normalized_text = sample_pipeline(
        model=active_model,
        fish_ae=active_fish_ae,
        pca_state=pca_state,
        sample_fn=sample_fn,
        text_prompt=text_prompt,
        speaker_audio=speaker_audio,
        rng_seed=rng_seed_int,
        pad_to_max_text_length=pad_to_max_text_length,
        pad_to_max_speaker_latent_length=pad_to_max_speaker_latent_length,
        normalize_text=True,
    )

    audio_to_save = audio_out[0].cpu()

    stem = make_stem("generated", session_id)
    output_path = save_audio_with_format(
        audio_to_save, TEMP_AUDIO_DIR, stem, 44100, audio_format
    )

    generation_time = time.time() - start_time
    time_str = f"⏱️ Total generation time: {generation_time:.2f}s"
    text_display = f"**Text Prompt (normalized):**\n\n{normalized_text}"

    recon_output_path = None
    original_output_path = None

    if reconstruct_first_30_seconds and speaker_audio is not None:
        audio_recon = ae_reconstruct(
            fish_ae=fish_ae,
            pca_state=pca_state,
            audio=torch.nn.functional.pad(
                speaker_audio[..., : 2048 * 640],
                (0, max(0, 2048 * 640 - speaker_audio.shape[-1])),
            )[None],
        )[..., : speaker_audio.shape[-1]]

        recon_stem = make_stem("speaker_recon", session_id)
        recon_output_path = save_audio_with_format(
            audio_recon.cpu()[0], TEMP_AUDIO_DIR, recon_stem, 44100, audio_format
        )

    if show_original_audio and speaker_audio is not None:
        original_stem = make_stem("original_audio", session_id)
        original_output_path = save_audio_with_format(
            speaker_audio.cpu(), TEMP_AUDIO_DIR, original_stem, 44100, audio_format
        )

    show_reference_section = (
        show_original_audio or reconstruct_first_30_seconds
    ) and speaker_audio is not None
    return (
        gr.update(),
        gr.update(value=str(output_path), visible=True),
        gr.update(value=text_display, visible=True),
        gr.update(
            value=str(original_output_path) if original_output_path else None,
            visible=True,
        ),
        gr.update(value=time_str, visible=True),
        gr.update(
            value=str(recon_output_path) if recon_output_path else None, visible=True
        ),
        gr.update(visible=(show_original_audio and speaker_audio is not None)),
        gr.update(visible=(reconstruct_first_30_seconds and speaker_audio is not None)),
        gr.update(visible=show_reference_section),
        gr.update(value=lora_status, visible=True),
    )


# UI Helper Functions
def load_text_presets():
    """Load text presets from file with category and word count."""
    if TEXT_PRESETS_PATH.exists():
        with open(TEXT_PRESETS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        result = []
        for line in lines:
            if " | " in line:
                parts = line.split(" | ", 1)
                category = parts[0]
                text = parts[1]
            else:
                category = "Uncategorized"
                text = line

            word_count = len(text.split())
            result.append([category, str(word_count), text])

        return result
    return []


def select_text_preset(evt: gr.SelectData):
    """Handle text preset selection - extract text from the row."""
    if evt.value:
        if isinstance(evt.index, (tuple, list)) and len(evt.index) >= 2:
            row_index = evt.index[0]
        else:
            row_index = evt.index

        presets_data = load_text_presets()
        if isinstance(row_index, int) and row_index < len(presets_data):
            text = presets_data[row_index][2]
            return gr.update(value=text)
    return gr.update()


def toggle_mode(mode):
    """Toggle advanced settings section visibility."""
    show_advanced = mode == "Advanced Mode"
    return gr.update(visible=show_advanced)


def update_force_row(force_speaker):
    """Show KV scaling controls when Force Speaker is enabled."""
    return gr.update(visible=bool(force_speaker))


def apply_cfg_preset(preset_name):
    """Apply CFG guidance preset."""
    presets = {
        "higher speaker": (3.0, 8.0, 0.5, 1.0),
        "large guidances": (8.0, 8.0, 0.5, 1.0),
    }

    if preset_name not in presets:
        return [gr.update()] * 5

    text_scale, speaker_scale, min_t, max_t = presets[preset_name]

    return [
        gr.update(value=text_scale),
        gr.update(value=speaker_scale),
        gr.update(value=min_t),
        gr.update(value=max_t),
        gr.update(value="Custom"),
    ]


def apply_speaker_kv_preset(preset_name):
    """Apply speaker KV attention control preset."""
    if preset_name == "enable":
        return [
            gr.update(value=True),
            gr.update(visible=True),
            gr.update(value="Custom"),
        ]
    if preset_name == "off":
        return [
            gr.update(value=False),
            gr.update(visible=False),
            gr.update(value="Custom"),
        ]
    return [gr.update()] * 3


def apply_truncation_preset(preset_name):
    """Apply truncation & temporal rescaling preset."""
    presets = {
        "flat": (0.8, 1.2, 3.0),
        "sharp": (0.9, 0.96, 3.0),
        "baseline(sharp)": (1.0, 1.0, 3.0),
    }

    if preset_name == "custom" or preset_name not in presets:
        return [gr.update()] * 4

    truncation, rescale_k, rescale_sigma = presets[preset_name]

    return [
        gr.update(value=truncation),
        gr.update(value=rescale_k),
        gr.update(value=rescale_sigma),
        gr.update(value="Custom"),
    ]


def load_sampler_presets():
    """Load sampler presets from JSON file."""
    if SAMPLER_PRESETS_PATH.exists():
        with open(SAMPLER_PRESETS_PATH, "r") as f:
            return json.load(f)

    default_presets = {
        "Independent-High-Speaker-CFG": {
            "num_steps": "40",
            "cfg_scale_text": "3.0",
            "cfg_scale_speaker": "8.0",
            "cfg_min_t": "0.5",
            "cfg_max_t": "1.0",
            "truncation_factor": "1.",
            "rescale_k": "1.",
            "rescale_sigma": "3.0",
        }
    }
    with open(SAMPLER_PRESETS_PATH, "w") as f:
        json.dump(default_presets, f, indent=2)
    return default_presets


def apply_sampler_preset(preset_name):
    """Apply a sampler preset to all fields."""
    presets = load_sampler_presets()
    if preset_name == "Custom" or preset_name not in presets:
        return [gr.update()] * 13

    preset = presets[preset_name]
    speaker_kv_enabled = to_bool(preset.get("speaker_kv_enable", False))

    def to_num(val, default):
        try:
            return float(val) if isinstance(val, str) else val
        except (ValueError, TypeError):
            return default

    return [
        gr.update(value=int(to_num(preset.get("num_steps", "40"), 40))),
        gr.update(value=to_num(preset.get("cfg_scale_text", "3.0"), 3.0)),
        gr.update(value=to_num(preset.get("cfg_scale_speaker", "5.0"), 5.0)),
        gr.update(value=to_num(preset.get("cfg_min_t", "0.5"), 0.5)),
        gr.update(value=to_num(preset.get("cfg_max_t", "1.0"), 1.0)),
        gr.update(value=to_num(preset.get("truncation_factor", "0.8"), 0.8)),
        gr.update(value=to_num(preset.get("rescale_k", "1.2"), 1.2)),
        gr.update(value=to_num(preset.get("rescale_sigma", "3.0"), 3.0)),
        gr.update(value=speaker_kv_enabled),
        gr.update(visible=speaker_kv_enabled),
        gr.update(value=to_num(preset.get("speaker_kv_scale", "1.5"), 1.5)),
        gr.update(value=to_num(preset.get("speaker_kv_min_t", "0.9"), 0.9)),
        gr.update(value=int(to_num(preset.get("speaker_kv_max_layers", "24"), 24))),
    ]


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}


def get_audio_prompt_files(search_query: str = ""):
    """Get list of audio files from the audio prompt folder, optionally filtered by search query."""
    if AUDIO_PROMPT_FOLDER is None or not AUDIO_PROMPT_FOLDER.exists():
        return []

    files = sorted(
        [
            f.name
            for f in AUDIO_PROMPT_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        ],
        key=str.lower,
    )

    # Filter by search query if provided
    if search_query.strip():
        query_lower = search_query.lower()
        files = [f for f in files if query_lower in f.lower()]

    return [[file] for file in files]


def filter_audio_prompts(search_query: str):
    """Filter audio prompts based on search query."""
    return gr.update(value=get_audio_prompt_files(search_query))


def select_audio_prompt_file(evt: gr.SelectData):
    """Handle audio prompt file selection from table."""
    if evt.value and AUDIO_PROMPT_FOLDER is not None:
        file_path = AUDIO_PROMPT_FOLDER / evt.value
        if file_path.exists():
            return gr.update(value=str(file_path))
    return gr.update()


# UI styling and helpers
LINK_CSS = """
.preset-inline { display:flex; align-items:baseline; gap:6px; margin-top:-4px; margin-bottom:-12px; }
.preset-inline .title { font-weight:600; font-size:.95rem; }
.preset-inline .dim   { color:#666; margin:0 4px; }
a.preset-link { color: #0a5bd8; text-decoration: underline; cursor: pointer; font-weight: 400; }
a.preset-link:hover { text-decoration: none; opacity: 0.8; }
.dark a.preset-link,
[data-theme="dark"] a.preset-link { color: #60a5fa !important; }
.dark a.preset-link:hover,
[data-theme="dark"] a.preset-link:hover { color: #93c5fd !important; }
.dark .preset-inline .dim,
[data-theme="dark"] .preset-inline .dim { color: #9ca3af !important; }
.proxy-btn { position:absolute; width:0; height:0; overflow:hidden; padding:0 !important; margin:0 !important; border:0 !important; opacity:0; pointer-events:none; }
.gr-group { border: 1px solid #d1d5db !important; background: #f3f4f6 !important; }
.dark .gr-group,
[data-theme="dark"] .gr-group { border: 1px solid #4b5563 !important; background: #1f2937 !important; }
.generated-audio-player { border: 3px solid #667eea !important; border-radius: 12px !important; padding: 20px !important; background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.05) 100%) !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important; margin: 1rem 0 !important; }
.generated-audio-player > div { background: transparent !important; }
#component-mode-selector { text-align: center; padding: 1rem 0; }
#component-mode-selector label { font-size: 1.1rem !important; font-weight: 600 !important; margin-bottom: 0.5rem !important; }
#component-mode-selector .wrap { justify-content: center !important; }
#component-mode-selector fieldset { border: 2px solid #e5e7eb !important; border-radius: 8px !important; padding: 1rem !important; background: #f9fafb !important; }
.dark #component-mode-selector fieldset,
[data-theme="dark"] #component-mode-selector fieldset { border: 2px solid #4b5563 !important; background: #1f2937 !important; }
.section-separator { height: 3px !important; background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 80%, transparent 100%) !important; border: none !important; margin: 2rem 0 !important; }
.dark .section-separator,
[data-theme="dark"] .section-separator { background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 80%, transparent 100%) !important; }
.gradio-container h1,
.gradio-container h2 { font-weight: 700 !important; margin-top: 1.5rem !important; margin-bottom: 1rem !important; }
.tip-box { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important; border-left: 4px solid #f59e0b !important; border-radius: 8px !important; padding: 1rem 1.5rem !important; margin: 1rem 0 !important; box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1) !important; }
.tip-box strong { color: #92400e !important; }
.dark .tip-box,
[data-theme="dark"] .tip-box { background: linear-gradient(135deg, #451a03 0%, #78350f 100%) !important; border-left: 4px solid #f59e0b !important; }
.dark .tip-box strong,
[data-theme="dark"] .tip-box strong { color: #fbbf24 !important; }
"""

JS_CODE = r"""
function () {
  const appEl = document.querySelector("gradio-app");
  const root  = appEl && appEl.shadowRoot ? appEl.shadowRoot : document;
  function clickHiddenButtonById(id) {
    if (!id) return;
    const host = root.getElementById(id);
    if (!host) return;
    const realBtn = host.querySelector("button, [role='button']") || host;
    realBtn.click();
  }
  root.addEventListener("click", (ev) => {
    const a = ev.target.closest("a.preset-link");
    if (!a) return;
    ev.preventDefault();
    ev.stopPropagation();
    ev.stopImmediatePropagation();
    clickHiddenButtonById(a.getAttribute("data-fire"));
    return false;
  }, true);
}
"""


def init_session():
    """Initialize session ID for this browser tab/session."""
    return secrets.token_hex(8)


with gr.Blocks(title="Echo-TTS", css=LINK_CSS, js=JS_CODE) as demo:
    gr.Markdown("# Echo-TTS")
    gr.Markdown(
        "*Jordan Darefsky, 2025. See technical details [here](https://jordandarefsky.com/blog/2025/echo/)*"
    )

    gr.Markdown(
        "**License Notice:** All audio outputs are subject to non-commercial use [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)."
    )

    gr.Markdown(
        "**Responsible Use:** Do not use this model to impersonate real people without their explicit consent or to generate deceptive audio."
    )

    with gr.Accordion("📖 Quick Start Instructions", open=True):
        gr.Markdown(
            """
            1. Upload or record a short reference clip (or leave blank for no speaker reference).
            2. Pick a text preset or type your own prompt.
            3. Click **Generate Audio**.

            <div class="tip-box">
            💡 **Tip:** If the generated voice does not match the reference, enable "Force Speaker" and regenerate.
            </div>
            """
        )

    session_id_state = gr.State(None)

    gr.Markdown("# Speaker Reference")
    with gr.Row():
        if AUDIO_PROMPT_FOLDER is not None and AUDIO_PROMPT_FOLDER.exists():
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("#### Audio Library (click to load)")
                audio_prompt_search = gr.Textbox(
                    label="",
                    placeholder="🔍 Search audio prompts...",
                    lines=1,
                    max_lines=1,
                )
                audio_prompt_table = gr.Dataframe(
                    value=get_audio_prompt_files(),
                    headers=["Filename"],
                    datatype=["str"],
                    row_count=(10, "dynamic"),
                    col_count=(1, "fixed"),
                    interactive=False,
                    label="",
                )
        with gr.Column(scale=2):
            custom_audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Speaker Reference Audio (first five minutes used; blank for no speaker reference)",
                max_length=600,
            )

    gr.HTML('<hr class="section-separator">')

    # LoRA Checkpoint Section
    gr.Markdown("# LoRA Fine-Tuning (Optional)")
    with gr.Accordion("🎯 Load Custom LoRA Checkpoint", open=False):
        gr.Markdown(
            """
            Enter the path to a LoRA checkpoint (`.pt` file) from fine-tuning to apply custom style adaptation.

            <div class="tip-box">
            💡 **Tip:** Leave blank to use the base Echo-TTS model. Provide a path to a LoRA checkpoint to use a fine-tuned style (e.g., rap, singing, accents).
            </div>
            """
        )
        lora_checkpoint = gr.Textbox(
            label="LoRA Checkpoint Path",
            placeholder="/content/drive/MyDrive/lora_best_2.pt",
            lines=1,
        )
        lora_strength = gr.Slider(
            minimum=0.0,
            maximum=3.0,
            value=1.0,
            step=0.05,
            label="LoRA Strength",
            info="Control the strength of the LoRA adaptation (0.0 = no effect, 1.0 = full strength, >1.0 = amplified)",
        )
        lora_status = gr.Markdown("ℹ️ Using base model (no LoRA)", visible=True)

    gr.HTML('<hr class="section-separator">')
    gr.Markdown("# Text Prompt")
    with gr.Accordion("Text Presets", open=True):
        text_presets_table = gr.Dataframe(
            value=load_text_presets(),
            headers=["Category", "Words", "Preset Text"],
            datatype=["str", "str", "str"],
            row_count=(3, "dynamic"),
            col_count=(3, "fixed"),
            interactive=False,
            column_widths=["12%", "6%", "82%"],
        )
    text_prompt = gr.Textbox(
        label="Text Prompt", placeholder="[S1] Enter your text prompt here...", lines=4
    )

    gr.HTML('<hr class="section-separator">')
    gr.Markdown("# Generation")

    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            mode_selector = gr.Radio(
                choices=["Simple Mode", "Advanced Mode"],
                value="Simple Mode",
                label="",
                info=None,
                elem_id="component-mode-selector",
            )
        with gr.Column(scale=1):
            pass

    with gr.Accordion("⚙️ Generation Parameters", open=True):
        with gr.Row(equal_height=False):
            presets = load_sampler_presets()
            preset_keys = list(presets.keys())
            first_preset = preset_keys[0] if preset_keys else "Custom"

            with gr.Column(scale=2):
                preset_dropdown = gr.Dropdown(
                    choices=["Custom"] + preset_keys,
                    value=first_preset,
                    label="Sampler Preset",
                    info="Load preset configurations",
                )

            with gr.Column(scale=0.8, min_width=100):
                num_steps = gr.Number(
                    label="Steps",
                    value=40,
                    info="Sampling steps (Try 20-80)",
                    precision=0,
                    minimum=5,
                    step=5,
                    maximum=80,
                )

            with gr.Column(scale=0.8, min_width=100):
                rng_seed = gr.Number(
                    label="RNG Seed", value=0, info="Seed for noise", precision=0
                )

            with gr.Column(scale=3):
                with gr.Group():
                    gr.HTML(
                        """
                    <div class="preset-inline">
                      <span class="title">Speaker KV Attention Scaling</span>
                    </div>
                    """
                    )
                    spk_kv_preset_enable = gr.Button(
                        "", elem_id="spk_kv_enable", elem_classes=["proxy-btn"]
                    )
                    spk_kv_preset_off = gr.Button(
                        "", elem_id="spk_kv_off", elem_classes=["proxy-btn"]
                    )
                    force_speaker = gr.Checkbox(
                        label='"Force Speaker" (KV scaling)',
                        value=False,
                        info="Enable to more strongly match the reference speaker (though higher values may degrade quality)",
                    )
                    with gr.Row(visible=False) as speaker_kv_row:
                        speaker_kv_scale = gr.Number(
                            label="KV Scale",
                            value=1.5,
                            info="Scale factor (>1 -> larger effect; try 1.5, 1.2, ...)",
                            minimum=0,
                            step=0.1,
                        )
                        speaker_kv_min_t = gr.Number(
                            label="KV Min t",
                            value=0.9,
                            info="(0-1), scale applied from steps t=1. to val",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                        )
                        speaker_kv_max_layers = gr.Number(
                            label="Max Layers",
                            value=24,
                            info="(0-24), scale applied in first N layers",
                            precision=0,
                            minimum=0,
                            maximum=24,
                        )

        with gr.Column(visible=False) as advanced_mode_column:
            compile_checkbox = gr.Checkbox(
                label="Compile Model",
                value=False,
                info="Compile for faster runs (~10-30% faster); forces Custom Shapes on to avoid excessive recompilation.",
            )
            use_custom_shapes_checkbox = gr.Checkbox(
                label="Use Custom Shapes (Advanced)",
                value=False,
                info="Override default generation length and/or force latent and text padding (if unchecked, no padding is used and latent generation length is 640≈30s.)",
            )

            with gr.Row(visible=False) as custom_shapes_row:
                max_text_byte_length = gr.Textbox(
                    label="Max Text Byte Length (padded)",
                    value="768",
                    info="Single value or comma-separated buckets (auto-selects min >= length); 768 = max; leave blank for no padding",
                    scale=1,
                )
                max_speaker_latent_length = gr.Textbox(
                    label="Max Speaker Latent Length (padded)",
                    value="640, 2816, 6400",
                    info="Single value or comma-separated buckets (auto-selects min >= length); 640≈30s, 2560≈2min, 6400≈5min (max); leave blank for no padding",
                    scale=1,
                )
                sample_latent_length = gr.Textbox(
                    label="Sample Latent Length",
                    value=str(DEFAULT_SAMPLE_LATENT_LENGTH),
                    info="Maximum sample latent length (640≈30s max seen during training; smaller works well for generating prefixes)",
                    scale=1,
                )

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML(
                            """
                        <div class="preset-inline">
                          <span class="title">Truncation &amp; Temporal Rescaling</span><span class="dim">(</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="trunc_flat">flat</a>
                          <span class="dim">,</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="trunc_sharp">sharp</a>
                          <span class="dim">,</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="trunc_baseline">baseline(sharp)</a>
                          <span class="dim">)</span>
                        </div>
                        """
                        )
                        trunc_preset_flat = gr.Button(
                            "", elem_id="trunc_flat", elem_classes=["proxy-btn"]
                        )
                        trunc_preset_sharp = gr.Button(
                            "", elem_id="trunc_sharp", elem_classes=["proxy-btn"]
                        )
                        trunc_preset_baseline = gr.Button(
                            "", elem_id="trunc_baseline", elem_classes=["proxy-btn"]
                        )
                        with gr.Row():
                            truncation_factor = gr.Number(
                                label="Truncation Factor",
                                value=0.8,
                                info="Multiply initial noise (<1 helps artifacts)",
                                minimum=0,
                                step=0.05,
                            )
                            rescale_k = gr.Number(
                                label="Rescale k",
                                value=1.2,
                                info="<1=sharpen, >1=flatten, 1=off",
                                minimum=0,
                                step=0.05,
                            )
                            rescale_sigma = gr.Number(
                                label="Rescale σ",
                                value=3.0,
                                info="Sigma parameter",
                                minimum=0,
                                step=0.1,
                            )

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML(
                            """
                        <div class="preset-inline">
                          <span class="title">CFG Guidance</span><span class="dim">(</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="cfg_higher">higher speaker</a>
                          <span class="dim">,</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="cfg_large">large guidances</a>
                          <span class="dim">)</span>
                        </div>
                        """
                        )
                        cfg_preset_higher_speaker = gr.Button(
                            "", elem_id="cfg_higher", elem_classes=["proxy-btn"]
                        )
                        cfg_preset_large_guidances = gr.Button(
                            "", elem_id="cfg_large", elem_classes=["proxy-btn"]
                        )
                        with gr.Row():
                            cfg_scale_text = gr.Number(
                                label="Text CFG Scale",
                                value=3.0,
                                info="Guidance strength for text",
                                minimum=0,
                                step=0.5,
                            )
                            cfg_scale_speaker = gr.Number(
                                label="Speaker CFG Scale",
                                value=5.0,
                                info="Guidance strength for speaker",
                                minimum=0,
                                step=0.5,
                            )

                        with gr.Row():
                            cfg_min_t = gr.Number(
                                label="CFG Min t",
                                value=0.5,
                                info="(0-1), CFG applied when t >= val",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                            )
                            cfg_max_t = gr.Number(
                                label="CFG Max t",
                                value=1.0,
                                info="(0-1), CFG applied when t <= val",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                            )

    with gr.Row(equal_height=True):
        audio_format = gr.Radio(
            choices=["wav", "mp3"], value="wav", label="Format", scale=1, min_width=90
        )
        generate_btn = gr.Button(
            "Generate Audio", variant="primary", size="lg", scale=10
        )
        with gr.Column(scale=1):
            show_original_audio = gr.Checkbox(
                label="Re-display Original Audio (full 5-minute cropped mono)",
                value=False,
            )
            reconstruct_first_30_seconds = gr.Checkbox(
                label="Show Autoencoder Reconstruction (only first 30s of reference)",
                value=False,
            )

    gr.HTML('<hr class="section-separator">')
    with gr.Accordion("Generated Audio", open=True, visible=True) as generated_section:
        generation_time_display = gr.Markdown("", visible=False)
        with gr.Group(elem_classes=["generated-audio-player"]):
            generated_audio = gr.Audio(label="Generated Audio", visible=True)
        text_prompt_display = gr.Markdown("", visible=False)

        gr.Markdown("---")
        reference_audio_header = gr.Markdown("#### Reference Audio", visible=False)

        with gr.Accordion(
            "Original Audio (5 min Cropped Mono)", open=False, visible=False
        ) as original_accordion:
            original_audio = gr.Audio(
                label="Original Reference Audio (5 min)", visible=True
            )

        with gr.Accordion(
            "Autoencoder Reconstruction of First 30s of Reference",
            open=False,
            visible=False,
        ) as reference_accordion:
            reference_audio = gr.Audio(
                label="Decoded Reference Audio (30s)", visible=True
            )

    # Event handlers
    if AUDIO_PROMPT_FOLDER is not None and AUDIO_PROMPT_FOLDER.exists():
        audio_prompt_table.select(
            select_audio_prompt_file, outputs=[custom_audio_input]
        )
        audio_prompt_search.change(
            filter_audio_prompts,
            inputs=[audio_prompt_search],
            outputs=[audio_prompt_table],
        )

    text_presets_table.select(select_text_preset, outputs=text_prompt)

    mode_selector.change(
        toggle_mode, inputs=[mode_selector], outputs=[advanced_mode_column]
    )

    force_speaker.change(
        update_force_row, inputs=[force_speaker], outputs=[speaker_kv_row]
    )

    def toggle_custom_shapes(enabled):
        return gr.update(visible=enabled)

    use_custom_shapes_checkbox.change(
        toggle_custom_shapes,
        inputs=[use_custom_shapes_checkbox],
        outputs=[custom_shapes_row],
    )

    def on_compile_change(compile_enabled):
        """When compile is enabled, force custom shapes to be enabled."""
        if compile_enabled:
            return (
                gr.update(value=True),  # use_custom_shapes_checkbox
                gr.update(visible=True),  # custom_shapes_row
            )
        return (
            gr.update(),
            gr.update(),
        )

    compile_checkbox.change(
        on_compile_change,
        inputs=[compile_checkbox],
        outputs=[use_custom_shapes_checkbox, custom_shapes_row],
    )

    cfg_preset_higher_speaker.click(
        lambda: apply_cfg_preset("higher speaker"),
        outputs=[
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            preset_dropdown,
        ],
    )
    cfg_preset_large_guidances.click(
        lambda: apply_cfg_preset("large guidances"),
        outputs=[
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            preset_dropdown,
        ],
    )

    spk_kv_preset_enable.click(
        lambda: apply_speaker_kv_preset("enable"),
        outputs=[force_speaker, speaker_kv_row, preset_dropdown],
    )
    spk_kv_preset_off.click(
        lambda: apply_speaker_kv_preset("off"),
        outputs=[force_speaker, speaker_kv_row, preset_dropdown],
    )

    trunc_preset_flat.click(
        lambda: apply_truncation_preset("flat"),
        outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown],
    )
    trunc_preset_sharp.click(
        lambda: apply_truncation_preset("sharp"),
        outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown],
    )
    trunc_preset_baseline.click(
        lambda: apply_truncation_preset("baseline(sharp)"),
        outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown],
    )

    preset_dropdown.change(
        apply_sampler_preset,
        inputs=preset_dropdown,
        outputs=[
            num_steps,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            truncation_factor,
            rescale_k,
            rescale_sigma,
            force_speaker,
            speaker_kv_row,
            speaker_kv_scale,
            speaker_kv_min_t,
            speaker_kv_max_layers,
        ],
    )

    generate_btn.click(
        generate_audio,
        inputs=[
            text_prompt,
            custom_audio_input,
            num_steps,
            rng_seed,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            truncation_factor,
            rescale_k,
            rescale_sigma,
            force_speaker,
            speaker_kv_scale,
            speaker_kv_min_t,
            speaker_kv_max_layers,
            reconstruct_first_30_seconds,
            use_custom_shapes_checkbox,
            max_text_byte_length,
            max_speaker_latent_length,
            sample_latent_length,
            audio_format,
            compile_checkbox,
            show_original_audio,
            lora_checkpoint,
            lora_strength,
            session_id_state,
        ],
        outputs=[
            generated_section,
            generated_audio,
            text_prompt_display,
            original_audio,
            generation_time_display,
            reference_audio,
            original_accordion,
            reference_accordion,
            reference_audio_header,
            lora_status,
        ],
    )

    demo.load(init_session, outputs=[session_id_state]).then(
        lambda: apply_sampler_preset(list(load_sampler_presets().keys())[0]),
        outputs=[
            num_steps,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            truncation_factor,
            rescale_k,
            rescale_sigma,
            force_speaker,
            speaker_kv_row,
            speaker_kv_scale,
            speaker_kv_min_t,
            speaker_kv_max_layers,
        ],
    )


if __name__ == "__main__":
    # Check for share flag via environment variable
    share = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")

    if share:
        print("\n" + "=" * 60)
        print("🌐 Creating public shareable URL...")
        print("=" * 60)

    demo.launch(
        allowed_paths=[str(AUDIO_PROMPT_FOLDER)],
        share=share,
    )

    if share:
        print("\n💡 Public URL generated! Share it with others to test your model.")
