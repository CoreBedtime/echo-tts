
import os
import json
import time
import secrets
import logging
import warnings
from pathlib import Path
from typing import Tuple, Any
from functools import partial

# see lines ~40-50 for running on lower VRAM GPUs

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import gradio as gr
import torch
import torchaudio

from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    ae_reconstruct,
    ae_encode,
    ae_decode,
    sample_pipeline,
    compile_model,
    compile_fish_ae,
    sample_euler_cfg_independent_guidances,
    get_speaker_latent_and_mask,
    get_text_input_ids_and_mask,
    crop_audio_to_flattening_point,
)
from inference_blockwise import sample_blockwise_euler_cfg_independent_guidances

# --------------------------------------------------------------------
# IF ON 8GB VRAM GPU, SET FISH_AE_DTYPE to bfloat16 and DEFAULT_SAMPLE_LATENT_LENGTH to < 640 (e.g., 576)

# Configuration
MODEL_DTYPE = torch.bfloat16
FISH_AE_DTYPE = torch.float32
# FISH_AE_DTYPE = torch.bfloat16 # USE THIS IF OOM ON 8GB vram GPU

DEFAULT_SAMPLE_LATENT_LENGTH = 640 # decrease if OOM on 8GB vram GPU
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
# Note: delete_blockwise_modules=False to enable continuation and rhythm transfer modes
model = load_model_from_hf(dtype=MODEL_DTYPE, delete_blockwise_modules=False)
fish_ae = load_fish_ae_from_hf(dtype=FISH_AE_DTYPE)
pca_state = load_pca_state_from_hf()

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


def save_audio_with_format(audio_tensor: torch.Tensor, base_path: Path, filename: str, sample_rate: int, audio_format: str) -> Path:
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
    session_id: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Generate audio using the model."""
    global model_compiled, fish_ae_compiled

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
    cfg_scale_speaker_val = float(cfg_scale_speaker) if cfg_scale_speaker is not None else None
    cfg_min_t_val = float(cfg_min_t)
    cfg_max_t_val = float(cfg_max_t)
    truncation_factor_val = float(truncation_factor)
    rescale_k_val = float(rescale_k) if rescale_k != 1.0 else None
    rescale_sigma_val = float(rescale_sigma)

    speaker_kv_enabled = bool(force_speaker)
    if speaker_kv_enabled:
        speaker_kv_scale_val = float(speaker_kv_scale) if speaker_kv_scale is not None else None
        speaker_kv_min_t_val = float(speaker_kv_min_t) if speaker_kv_min_t is not None else None
        speaker_kv_max_layers_val = int(speaker_kv_max_layers) if speaker_kv_max_layers is not None else None
    else:
        speaker_kv_scale_val = None
        speaker_kv_min_t_val = None
        speaker_kv_max_layers_val = None

    # Load speaker audio early so we can compute actual lengths for bucket selection
    use_zero_speaker = not speaker_audio_path or speaker_audio_path == ""
    speaker_audio = load_audio(speaker_audio_path).cuda() if not use_zero_speaker else None

    if use_custom_shapes:
        # Compute actual text byte length
        actual_text_byte_length = len(text_prompt.encode("utf-8")) + 1  # +1 for BOS token
        
        # Compute actual speaker latent length (audio_samples // 2048)
        AE_DOWNSAMPLE_FACTOR = 2048
        if speaker_audio is not None:
            actual_speaker_latent_length = (speaker_audio.shape[-1] // AE_DOWNSAMPLE_FACTOR) // 4 * 4
        else:
            actual_speaker_latent_length = 0
        
        # Find appropriate bucket sizes from comma-separated values
        pad_to_max_text_length = find_min_bucket_gte(max_text_byte_length, actual_text_byte_length)
        pad_to_max_speaker_latent_length = find_min_bucket_gte(max_speaker_latent_length, actual_speaker_latent_length)
        sample_latent_length_val = int(sample_latent_length) if sample_latent_length.strip() else (DEFAULT_SAMPLE_LATENT_LENGTH or 640)
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
    output_path = save_audio_with_format(audio_to_save, TEMP_AUDIO_DIR, stem, 44100, audio_format)

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
                speaker_audio[..., :2048 * 640],
                (0, max(0, 2048 * 640 - speaker_audio.shape[-1])),
            )[None],
        )[..., : speaker_audio.shape[-1]]

        recon_stem = make_stem("speaker_recon", session_id)
        recon_output_path = save_audio_with_format(audio_recon.cpu()[0], TEMP_AUDIO_DIR, recon_stem, 44100, audio_format)

    if show_original_audio and speaker_audio is not None:
        original_stem = make_stem("original_audio", session_id)
        original_output_path = save_audio_with_format(speaker_audio.cpu(), TEMP_AUDIO_DIR, original_stem, 44100, audio_format)

    show_reference_section = (show_original_audio or reconstruct_first_30_seconds) and speaker_audio is not None
    return (
        gr.update(),
        gr.update(value=str(output_path), visible=True),
        gr.update(value=text_display, visible=True),
        gr.update(value=str(original_output_path) if original_output_path else None, visible=True),
        gr.update(value=time_str, visible=True),
        gr.update(value=str(recon_output_path) if recon_output_path else None, visible=True),
        gr.update(visible=(show_original_audio and speaker_audio is not None)),
        gr.update(visible=(reconstruct_first_30_seconds and speaker_audio is not None)),
        gr.update(visible=show_reference_section),
    )


def generate_audio_continuation(
    text_prompt: str,
    speaker_audio_path: str,
    continuation_audio_path: str,
    continuation_new_latents: int,
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
    audio_format: str,
    session_id: str,
) -> Tuple[Any, Any, Any, Any]:
    """Generate audio continuing from an existing audio clip using blockwise generation."""

    cleanup_temp_audio(TEMP_AUDIO_DIR, session_id)
    start_time = time.time()

    device, dtype = model.device, model.dtype

    # Validate inputs
    if not continuation_audio_path:
        return (
            gr.update(),
            gr.update(value=None, visible=True),
            gr.update(value="**Error:** Please provide continuation audio to continue from.", visible=True),
            gr.update(value="", visible=False),
        )

    # Parse parameters
    num_steps_int = min(max(int(num_steps), 1), 80)
    rng_seed_int = int(rng_seed) if rng_seed is not None else 0
    cfg_scale_text_val = float(cfg_scale_text)
    cfg_scale_speaker_val = float(cfg_scale_speaker) if cfg_scale_speaker is not None else 5.0
    cfg_min_t_val = float(cfg_min_t)
    cfg_max_t_val = float(cfg_max_t)
    truncation_factor_val = float(truncation_factor)
    rescale_k_val = float(rescale_k) if rescale_k != 1.0 else None
    rescale_sigma_val = float(rescale_sigma)
    continuation_new_latents_val = int(continuation_new_latents) if continuation_new_latents else 256

    speaker_kv_enabled = bool(force_speaker)
    if speaker_kv_enabled:
        speaker_kv_scale_val = float(speaker_kv_scale) if speaker_kv_scale is not None else None
        speaker_kv_min_t_val = float(speaker_kv_min_t) if speaker_kv_min_t is not None else None
        speaker_kv_max_layers_val = int(speaker_kv_max_layers) if speaker_kv_max_layers is not None else None
    else:
        speaker_kv_scale_val = None
        speaker_kv_min_t_val = None
        speaker_kv_max_layers_val = None

    # Load speaker audio
    use_zero_speaker = not speaker_audio_path or speaker_audio_path == ""
    if use_zero_speaker:
        speaker_latent = torch.zeros((1, 4, 80), device=device, dtype=dtype)
        speaker_mask = torch.zeros((1, 4), device=device, dtype=torch.bool)
    else:
        speaker_audio = load_audio(speaker_audio_path).to(device)
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(fish_ae, pca_state, speaker_audio)

    # Load and encode continuation audio
    continuation_audio = load_audio(continuation_audio_path).to(device)
    continuation_latent, continuation_mask = get_speaker_latent_and_mask(fish_ae, pca_state, continuation_audio)
    # Trim to actual content (remove padding)
    actual_len = continuation_mask.sum().item()
    continuation_latent = continuation_latent[:, :actual_len]

    # Check that total length won't exceed 640
    max_new_latents = 640 - continuation_latent.shape[1]
    if max_new_latents <= 0:
        return (
            gr.update(),
            gr.update(value=None, visible=True),
            gr.update(value=f"**Error:** Continuation audio is too long ({continuation_latent.shape[1]} latents). Maximum is 640 latents (~30s).", visible=True),
            gr.update(value="", visible=False),
        )

    # Clamp new latents to available space
    continuation_new_latents_val = min(continuation_new_latents_val, max_new_latents)

    # Prepare text
    text_input_ids, text_mask, normalized_text = get_text_input_ids_and_mask(
        [text_prompt], max_length=None, device=device, normalize=True, return_normalized_text=True
    )

    # Generate using blockwise sampling
    latent_out = sample_blockwise_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=rng_seed_int,
        block_sizes=[continuation_new_latents_val],
        num_steps=num_steps_int,
        cfg_scale_text=cfg_scale_text_val,
        cfg_scale_speaker=cfg_scale_speaker_val,
        cfg_min_t=cfg_min_t_val,
        cfg_max_t=cfg_max_t_val,
        truncation_factor=truncation_factor_val,
        rescale_k=rescale_k_val,
        rescale_sigma=rescale_sigma_val,
        speaker_kv_scale=speaker_kv_scale_val,
        speaker_kv_max_layers=speaker_kv_max_layers_val,
        speaker_kv_min_t=speaker_kv_min_t_val,
        continuation_latent=continuation_latent,
    )

    # Decode to audio
    audio_out = ae_decode(fish_ae, pca_state, latent_out)
    audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])

    audio_to_save = audio_out[0].cpu()
    stem = make_stem("continuation", session_id)
    output_path = save_audio_with_format(audio_to_save, TEMP_AUDIO_DIR, stem, 44100, audio_format)

    generation_time = time.time() - start_time
    time_str = f"⏱️ Total generation time: {generation_time:.2f}s"
    text_display = f"**Text Prompt (normalized):**\n\n{normalized_text[0]}\n\n**Continuation Info:**\n- Continuation latents: {continuation_latent.shape[1]} (~{continuation_latent.shape[1] * 2048 / 44100:.1f}s)\n- New latents generated: {continuation_new_latents_val} (~{continuation_new_latents_val * 2048 / 44100:.1f}s)"

    return (
        gr.update(),
        gr.update(value=str(output_path), visible=True),
        gr.update(value=text_display, visible=True),
        gr.update(value=time_str, visible=True),
    )


def extract_rhythm_block_sizes(
    rhythm_audio: torch.Tensor,
    target_total_latents: int,
    min_block_size: int = 16,
    num_blocks: int = 8,
) -> list:
    """
    Extract rhythm/timing information from audio and convert to block sizes.

    This analyzes the energy envelope of the audio to determine speaking rhythm
    and maps it to block sizes for generation.
    """
    # Encode the rhythm source audio to latents
    rhythm_latent = ae_encode(fish_ae, pca_state, rhythm_audio.unsqueeze(0))
    rhythm_len = rhythm_latent.shape[1]

    # Compute energy per latent frame (simple L2 norm)
    energy = torch.norm(rhythm_latent[0], dim=-1)

    # Normalize and segment into blocks
    # We'll create blocks proportional to the rhythm audio duration
    scale_factor = target_total_latents / rhythm_len

    # Split rhythm into num_blocks segments and compute relative durations
    segment_size = rhythm_len // num_blocks
    block_sizes = []

    for i in range(num_blocks):
        start = i * segment_size
        end = start + segment_size if i < num_blocks - 1 else rhythm_len

        # Use segment length scaled to target
        seg_len = end - start
        block_size = int(seg_len * scale_factor)
        block_size = max(min_block_size, block_size)
        # Round to multiple of 4 for model compatibility
        block_size = (block_size // 4) * 4
        if block_size < 4:
            block_size = 4
        block_sizes.append(block_size)

    # Adjust to match target total
    current_total = sum(block_sizes)
    if current_total != target_total_latents:
        diff = target_total_latents - current_total
        # Distribute difference across blocks
        per_block = diff // len(block_sizes)
        remainder = diff % len(block_sizes)
        for i in range(len(block_sizes)):
            adjustment = per_block + (1 if i < abs(remainder) else 0)
            if diff < 0:
                adjustment = -abs(adjustment)
            block_sizes[i] = max(4, block_sizes[i] + adjustment)
            # Keep divisible by 4
            block_sizes[i] = (block_sizes[i] // 4) * 4

    # Final adjustment to hit exact target
    current_total = sum(block_sizes)
    if current_total != target_total_latents:
        diff = target_total_latents - current_total
        block_sizes[-1] = max(4, block_sizes[-1] + diff)

    return block_sizes


def generate_audio_rhythm_transfer(
    text_prompt: str,
    speaker_audio_path: str,
    rhythm_source_path: str,
    target_duration_seconds: float,
    num_rhythm_blocks: int,
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
    audio_format: str,
    session_id: str,
) -> Tuple[Any, Any, Any, Any]:
    """Generate audio using rhythm/timing extracted from a reference audio clip."""

    cleanup_temp_audio(TEMP_AUDIO_DIR, session_id)
    start_time = time.time()

    device, dtype = model.device, model.dtype

    # Validate inputs
    if not rhythm_source_path:
        return (
            gr.update(),
            gr.update(value=None, visible=True),
            gr.update(value="**Error:** Please provide a rhythm source audio.", visible=True),
            gr.update(value="", visible=False),
        )

    # Parse parameters
    num_steps_int = min(max(int(num_steps), 1), 80)
    rng_seed_int = int(rng_seed) if rng_seed is not None else 0
    cfg_scale_text_val = float(cfg_scale_text)
    cfg_scale_speaker_val = float(cfg_scale_speaker) if cfg_scale_speaker is not None else 5.0
    cfg_min_t_val = float(cfg_min_t)
    cfg_max_t_val = float(cfg_max_t)
    truncation_factor_val = float(truncation_factor)
    rescale_k_val = float(rescale_k) if rescale_k != 1.0 else None
    rescale_sigma_val = float(rescale_sigma)
    target_duration = float(target_duration_seconds) if target_duration_seconds else 10.0
    num_blocks = int(num_rhythm_blocks) if num_rhythm_blocks else 8

    speaker_kv_enabled = bool(force_speaker)
    if speaker_kv_enabled:
        speaker_kv_scale_val = float(speaker_kv_scale) if speaker_kv_scale is not None else None
        speaker_kv_min_t_val = float(speaker_kv_min_t) if speaker_kv_min_t is not None else None
        speaker_kv_max_layers_val = int(speaker_kv_max_layers) if speaker_kv_max_layers is not None else None
    else:
        speaker_kv_scale_val = None
        speaker_kv_min_t_val = None
        speaker_kv_max_layers_val = None

    # Load speaker audio
    use_zero_speaker = not speaker_audio_path or speaker_audio_path == ""
    if use_zero_speaker:
        speaker_latent = torch.zeros((1, 4, 80), device=device, dtype=dtype)
        speaker_mask = torch.zeros((1, 4), device=device, dtype=torch.bool)
    else:
        speaker_audio = load_audio(speaker_audio_path).to(device)
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(fish_ae, pca_state, speaker_audio)

    # Load rhythm source audio
    rhythm_audio = load_audio(rhythm_source_path).to(device)

    # Calculate target latents from duration (sample rate 44100, downsample factor 2048)
    target_latents = min(int(target_duration * 44100 / 2048), 640)
    target_latents = (target_latents // 4) * 4  # Ensure divisible by 4

    # Extract rhythm-based block sizes
    block_sizes = extract_rhythm_block_sizes(
        rhythm_audio,
        target_total_latents=target_latents,
        min_block_size=16,
        num_blocks=num_blocks,
    )

    # Prepare text
    text_input_ids, text_mask, normalized_text = get_text_input_ids_and_mask(
        [text_prompt], max_length=None, device=device, normalize=True, return_normalized_text=True
    )

    # Generate using blockwise sampling with rhythm-derived blocks
    latent_out = sample_blockwise_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=rng_seed_int,
        block_sizes=block_sizes,
        num_steps=num_steps_int,
        cfg_scale_text=cfg_scale_text_val,
        cfg_scale_speaker=cfg_scale_speaker_val,
        cfg_min_t=cfg_min_t_val,
        cfg_max_t=cfg_max_t_val,
        truncation_factor=truncation_factor_val,
        rescale_k=rescale_k_val,
        rescale_sigma=rescale_sigma_val,
        speaker_kv_scale=speaker_kv_scale_val,
        speaker_kv_max_layers=speaker_kv_max_layers_val,
        speaker_kv_min_t=speaker_kv_min_t_val,
        continuation_latent=None,
    )

    # Decode to audio
    audio_out = ae_decode(fish_ae, pca_state, latent_out)
    audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])

    audio_to_save = audio_out[0].cpu()
    stem = make_stem("rhythm_transfer", session_id)
    output_path = save_audio_with_format(audio_to_save, TEMP_AUDIO_DIR, stem, 44100, audio_format)

    generation_time = time.time() - start_time
    time_str = f"⏱️ Total generation time: {generation_time:.2f}s"

    # Format block sizes for display
    block_info = ", ".join([str(b) for b in block_sizes])
    text_display = f"**Text Prompt (normalized):**\n\n{normalized_text[0]}\n\n**Rhythm Transfer Info:**\n- Target duration: {target_duration:.1f}s ({target_latents} latents)\n- Number of blocks: {len(block_sizes)}\n- Block sizes: [{block_info}]"

    return (
        gr.update(),
        gr.update(value=str(output_path), visible=True),
        gr.update(value=text_display, visible=True),
        gr.update(value=time_str, visible=True),
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
            "rescale_sigma": "3.0"
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

    files = sorted([f.name for f in AUDIO_PROMPT_FOLDER.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS], key=str.lower)
    
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
    gr.Markdown("*Jordan Darefsky, 2025. See technical details [here](https://jordandarefsky.com/blog/2025/echo/)*")

    gr.Markdown("**License Notice:** All audio outputs are subject to non-commercial use [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).")

    gr.Markdown("**Responsible Use:** Do not use this model to impersonate real people without their explicit consent or to generate deceptive audio.")

    with gr.Accordion("📖 Quick Start Instructions", open=True):
        gr.Markdown(
            """
            **Standard Mode:**
            1. Upload or record a short reference clip (or leave blank for no speaker reference).
            2. Pick a text preset or type your own prompt.
            3. Click **Generate Audio**.

            **Continuation Mode:** Generate audio that continues from an existing clip.
            - Upload the audio to continue from in the Continuation Settings
            - Text prompt must include ALL text (original + new)

            **Rhythm Transfer Mode:** Apply the speaking rhythm from one audio to new speech.
            - Upload a rhythm source audio
            - The generated speech will follow similar pacing

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
    text_prompt = gr.Textbox(label="Text Prompt", placeholder="[S1] Enter your text prompt here...", lines=4)

    gr.HTML('<hr class="section-separator">')
    gr.Markdown("# Generation")

    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            generation_mode = gr.Radio(
                choices=["Standard", "Continuation", "Rhythm Transfer"],
                value="Standard",
                label="Generation Mode",
                info="Standard: normal generation | Continuation: continue from existing audio | Rhythm Transfer: use timing from reference",
            )
        with gr.Column(scale=1):
            pass

    # Continuation Mode Settings
    with gr.Accordion("🔗 Continuation Settings", open=True, visible=False) as continuation_settings:
        gr.Markdown("""
        **Continuation Mode** generates new audio that seamlessly continues from an existing audio clip.

        - Upload the audio you want to continue from
        - The text prompt should include ALL text (both the original and the new text to generate)
        - Use [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) for accurate transcription of the original audio
        """)
        with gr.Row():
            continuation_audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio to Continue From",
                max_length=30,
            )
            with gr.Column():
                continuation_new_latents = gr.Number(
                    label="New Latents to Generate",
                    value=256,
                    info="How many new latents to generate (~46ms each). Max depends on continuation length. 256 ≈ 12s",
                    minimum=16,
                    maximum=640,
                    step=16,
                    precision=0,
                )

    # Rhythm Transfer Mode Settings
    with gr.Accordion("🎵 Rhythm Transfer Settings", open=True, visible=False) as rhythm_settings:
        gr.Markdown("""
        **Rhythm Transfer Mode** extracts the speaking rhythm/pacing from a reference audio and applies it to new speech.

        - Upload a reference audio whose rhythm you want to mimic
        - The generated audio will have similar pacing and timing patterns
        - This works by analyzing the reference and creating block sizes that match its temporal structure
        """)
        with gr.Row():
            rhythm_source_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Rhythm Source Audio",
                max_length=60,
            )
            with gr.Column():
                rhythm_target_duration = gr.Number(
                    label="Target Duration (seconds)",
                    value=10.0,
                    info="Desired output duration in seconds (max ~30s)",
                    minimum=1.0,
                    maximum=30.0,
                    step=0.5,
                )
                rhythm_num_blocks = gr.Number(
                    label="Number of Rhythm Blocks",
                    value=8,
                    info="How many blocks to divide the generation into (more = finer rhythm control)",
                    minimum=2,
                    maximum=32,
                    step=1,
                    precision=0,
                )

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
                rng_seed = gr.Number(label="RNG Seed", value=0, info="Seed for noise", precision=0)

            with gr.Column(scale=3):
                with gr.Group():
                    gr.HTML(
                        """
                    <div class="preset-inline">
                      <span class="title">Speaker KV Attention Scaling</span>
                    </div>
                    """
                    )
                    spk_kv_preset_enable = gr.Button("", elem_id="spk_kv_enable", elem_classes=["proxy-btn"])
                    spk_kv_preset_off = gr.Button("", elem_id="spk_kv_off", elem_classes=["proxy-btn"])
                    force_speaker = gr.Checkbox(
                        label='"Force Speaker" (KV scaling)',
                        value=False,
                        info="Enable to more strongly match the reference speaker (though higher values may degrade quality)",
                    )
                    with gr.Row(visible=False) as speaker_kv_row:
                        speaker_kv_scale = gr.Number(label="KV Scale", value=1.5, info="Scale factor (>1 -> larger effect; try 1.5, 1.2, ...)", minimum=0, step=0.1)
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
                        trunc_preset_flat = gr.Button("", elem_id="trunc_flat", elem_classes=["proxy-btn"])
                        trunc_preset_sharp = gr.Button("", elem_id="trunc_sharp", elem_classes=["proxy-btn"])
                        trunc_preset_baseline = gr.Button("", elem_id="trunc_baseline", elem_classes=["proxy-btn"])
                        with gr.Row():
                            truncation_factor = gr.Number(
                                label="Truncation Factor",
                                value=0.8,
                                info="Multiply initial noise (<1 helps artifacts)",
                                minimum=0,
                                step=0.05,
                            )
                            rescale_k = gr.Number(
                                label="Rescale k", value=1.2, info="<1=sharpen, >1=flatten, 1=off", minimum=0, step=0.05
                            )
                            rescale_sigma = gr.Number(
                                label="Rescale σ", value=3.0, info="Sigma parameter", minimum=0, step=0.1
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
                        cfg_preset_higher_speaker = gr.Button("", elem_id="cfg_higher", elem_classes=["proxy-btn"])
                        cfg_preset_large_guidances = gr.Button("", elem_id="cfg_large", elem_classes=["proxy-btn"])
                        with gr.Row():
                            cfg_scale_text = gr.Number(
                                label="Text CFG Scale", value=3.0, info="Guidance strength for text", minimum=0, step=0.5
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
                                label="CFG Min t", value=0.5, info="(0-1), CFG applied when t >= val", minimum=0, maximum=1, step=0.05
                            )
                            cfg_max_t = gr.Number(
                                label="CFG Max t", value=1.0, info="(0-1), CFG applied when t <= val", minimum=0, maximum=1, step=0.05
                            )

    with gr.Row(equal_height=True):
        audio_format = gr.Radio(choices=["wav", "mp3"], value="wav", label="Format", scale=1, min_width=90)
        generate_btn = gr.Button("Generate Audio", variant="primary", size="lg", scale=10)
        with gr.Column(scale=1):
            show_original_audio = gr.Checkbox(label="Re-display Original Audio (full 5-minute cropped mono)", value=False)
            reconstruct_first_30_seconds = gr.Checkbox(
                label="Show Autoencoder Reconstruction (only first 30s of reference)", value=False
            )

    gr.HTML('<hr class="section-separator">')
    with gr.Accordion("Generated Audio", open=True, visible=True) as generated_section:
        generation_time_display = gr.Markdown("", visible=False)
        with gr.Group(elem_classes=["generated-audio-player"]):
            generated_audio = gr.Audio(label="Generated Audio", visible=True)
        text_prompt_display = gr.Markdown("", visible=False)

        gr.Markdown("---")
        reference_audio_header = gr.Markdown("#### Reference Audio", visible=False)

        with gr.Accordion("Original Audio (5 min Cropped Mono)", open=False, visible=False) as original_accordion:
            original_audio = gr.Audio(label="Original Reference Audio (5 min)", visible=True)

        with gr.Accordion("Autoencoder Reconstruction of First 30s of Reference", open=False, visible=False) as reference_accordion:
            reference_audio = gr.Audio(label="Decoded Reference Audio (30s)", visible=True)

    # Event handlers
    if AUDIO_PROMPT_FOLDER is not None and AUDIO_PROMPT_FOLDER.exists():
        audio_prompt_table.select(select_audio_prompt_file, outputs=[custom_audio_input])
        audio_prompt_search.change(filter_audio_prompts, inputs=[audio_prompt_search], outputs=[audio_prompt_table])

    text_presets_table.select(select_text_preset, outputs=text_prompt)

    def toggle_generation_mode(mode):
        """Show/hide mode-specific settings based on generation mode."""
        return (
            gr.update(visible=(mode == "Continuation")),
            gr.update(visible=(mode == "Rhythm Transfer")),
        )

    generation_mode.change(
        toggle_generation_mode,
        inputs=[generation_mode],
        outputs=[continuation_settings, rhythm_settings],
    )

    mode_selector.change(toggle_mode, inputs=[mode_selector], outputs=[advanced_mode_column])

    force_speaker.change(update_force_row, inputs=[force_speaker], outputs=[speaker_kv_row])

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
                gr.update(value=True),   # use_custom_shapes_checkbox
                gr.update(visible=True), # custom_shapes_row
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
        lambda: apply_cfg_preset("higher speaker"), outputs=[cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t, preset_dropdown]
    )
    cfg_preset_large_guidances.click(
        lambda: apply_cfg_preset("large guidances"), outputs=[cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t, preset_dropdown]
    )

    spk_kv_preset_enable.click(lambda: apply_speaker_kv_preset("enable"), outputs=[force_speaker, speaker_kv_row, preset_dropdown])
    spk_kv_preset_off.click(lambda: apply_speaker_kv_preset("off"), outputs=[force_speaker, speaker_kv_row, preset_dropdown])

    trunc_preset_flat.click(lambda: apply_truncation_preset("flat"), outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown])
    trunc_preset_sharp.click(lambda: apply_truncation_preset("sharp"), outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown])
    trunc_preset_baseline.click(
        lambda: apply_truncation_preset("baseline(sharp)"), outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown]
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

    def generate_router(
        gen_mode,
        text_prompt_val,
        speaker_audio_path,
        continuation_audio_path,
        continuation_new_latents_val,
        rhythm_source_path,
        rhythm_target_duration_val,
        rhythm_num_blocks_val,
        num_steps_val,
        rng_seed_val,
        cfg_scale_text_val,
        cfg_scale_speaker_val,
        cfg_min_t_val,
        cfg_max_t_val,
        truncation_factor_val,
        rescale_k_val,
        rescale_sigma_val,
        force_speaker_val,
        speaker_kv_scale_val,
        speaker_kv_min_t_val,
        speaker_kv_max_layers_val,
        reconstruct_first_30_seconds_val,
        use_custom_shapes_val,
        max_text_byte_length_val,
        max_speaker_latent_length_val,
        sample_latent_length_val,
        audio_format_val,
        compile_val,
        show_original_audio_val,
        session_id_val,
    ):
        """Route to the appropriate generation function based on mode."""
        if gen_mode == "Continuation":
            result = generate_audio_continuation(
                text_prompt=text_prompt_val,
                speaker_audio_path=speaker_audio_path,
                continuation_audio_path=continuation_audio_path,
                continuation_new_latents=continuation_new_latents_val,
                num_steps=num_steps_val,
                rng_seed=rng_seed_val,
                cfg_scale_text=cfg_scale_text_val,
                cfg_scale_speaker=cfg_scale_speaker_val,
                cfg_min_t=cfg_min_t_val,
                cfg_max_t=cfg_max_t_val,
                truncation_factor=truncation_factor_val,
                rescale_k=rescale_k_val,
                rescale_sigma=rescale_sigma_val,
                force_speaker=force_speaker_val,
                speaker_kv_scale=speaker_kv_scale_val,
                speaker_kv_min_t=speaker_kv_min_t_val,
                speaker_kv_max_layers=speaker_kv_max_layers_val,
                audio_format=audio_format_val,
                session_id=session_id_val,
            )
            # Return with empty values for unused outputs
            return (
                result[0],  # generated_section
                result[1],  # generated_audio
                result[2],  # text_prompt_display
                gr.update(),  # original_audio
                result[3],  # generation_time_display
                gr.update(),  # reference_audio
                gr.update(visible=False),  # original_accordion
                gr.update(visible=False),  # reference_accordion
                gr.update(visible=False),  # reference_audio_header
            )
        elif gen_mode == "Rhythm Transfer":
            result = generate_audio_rhythm_transfer(
                text_prompt=text_prompt_val,
                speaker_audio_path=speaker_audio_path,
                rhythm_source_path=rhythm_source_path,
                target_duration_seconds=rhythm_target_duration_val,
                num_rhythm_blocks=rhythm_num_blocks_val,
                num_steps=num_steps_val,
                rng_seed=rng_seed_val,
                cfg_scale_text=cfg_scale_text_val,
                cfg_scale_speaker=cfg_scale_speaker_val,
                cfg_min_t=cfg_min_t_val,
                cfg_max_t=cfg_max_t_val,
                truncation_factor=truncation_factor_val,
                rescale_k=rescale_k_val,
                rescale_sigma=rescale_sigma_val,
                force_speaker=force_speaker_val,
                speaker_kv_scale=speaker_kv_scale_val,
                speaker_kv_min_t=speaker_kv_min_t_val,
                speaker_kv_max_layers=speaker_kv_max_layers_val,
                audio_format=audio_format_val,
                session_id=session_id_val,
            )
            # Return with empty values for unused outputs
            return (
                result[0],  # generated_section
                result[1],  # generated_audio
                result[2],  # text_prompt_display
                gr.update(),  # original_audio
                result[3],  # generation_time_display
                gr.update(),  # reference_audio
                gr.update(visible=False),  # original_accordion
                gr.update(visible=False),  # reference_accordion
                gr.update(visible=False),  # reference_audio_header
            )
        else:
            # Standard mode
            return generate_audio(
                text_prompt=text_prompt_val,
                speaker_audio_path=speaker_audio_path,
                num_steps=num_steps_val,
                rng_seed=rng_seed_val,
                cfg_scale_text=cfg_scale_text_val,
                cfg_scale_speaker=cfg_scale_speaker_val,
                cfg_min_t=cfg_min_t_val,
                cfg_max_t=cfg_max_t_val,
                truncation_factor=truncation_factor_val,
                rescale_k=rescale_k_val,
                rescale_sigma=rescale_sigma_val,
                force_speaker=force_speaker_val,
                speaker_kv_scale=speaker_kv_scale_val,
                speaker_kv_min_t=speaker_kv_min_t_val,
                speaker_kv_max_layers=speaker_kv_max_layers_val,
                reconstruct_first_30_seconds=reconstruct_first_30_seconds_val,
                use_custom_shapes=use_custom_shapes_val,
                max_text_byte_length=max_text_byte_length_val,
                max_speaker_latent_length=max_speaker_latent_length_val,
                sample_latent_length=sample_latent_length_val,
                audio_format=audio_format_val,
                use_compile=compile_val,
                show_original_audio=show_original_audio_val,
                session_id=session_id_val,
            )

    generate_btn.click(
        generate_router,
        inputs=[
            generation_mode,
            text_prompt,
            custom_audio_input,
            continuation_audio_input,
            continuation_new_latents,
            rhythm_source_input,
            rhythm_target_duration,
            rhythm_num_blocks,
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
    demo.launch(
        allowed_paths=[str(AUDIO_PROMPT_FOLDER)]
    )
