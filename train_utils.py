"""
Training utilities for Echo-TTS few-shot fine-tuning.

This module provides data loading, loss computation, and training loop
utilities for fine-tuning Echo-TTS with LoRA adapters.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from model import EchoDiT
from autoencoder import DAC
from inference import (
    PCAState,
    ae_encode,
    ae_decode,
    get_speaker_latent_and_mask,
    get_text_input_ids_and_mask,
    load_audio,
)


# ============================================================================
# Data Loading
# ============================================================================

@dataclass
class TrainingSample:
    """A single training sample with audio, text, and optional speaker reference."""
    audio_path: str
    text: str
    speaker_audio_path: Optional[str] = None  # If None, uses same audio as speaker ref


def load_audio_tensor(
    path: str,
    sample_rate: int = 44100,
    max_duration: float = 30.0,
) -> torch.Tensor:
    """
    Load and preprocess audio file.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        max_duration: Maximum duration in seconds

    Returns:
        Audio tensor of shape (1, samples)
    """
    # Try using inference.load_audio first
    try:
        audio = load_audio(path, max_duration=int(max_duration))
        return audio
    except Exception:
        pass

    # Fallback to torchaudio
    audio, sr = torchaudio.load(path)

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    # Truncate to max duration
    max_samples = int(max_duration * sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    # Normalize
    audio = audio / torch.clamp(audio.abs().max(), min=1.0)

    return audio


def segment_audio(
    audio: torch.Tensor,
    segment_duration: float = 20.0,
    sample_rate: int = 44100,
    min_duration: float = 3.0,
    overlap: float = 0.0,
) -> List[torch.Tensor]:
    """
    Segment long audio into shorter chunks for training.

    Args:
        audio: Audio tensor of shape (1, samples)
        segment_duration: Target segment duration in seconds
        sample_rate: Audio sample rate
        min_duration: Minimum segment duration to keep
        overlap: Overlap between segments (0.0 to 1.0)

    Returns:
        List of audio tensors
    """
    segment_samples = int(segment_duration * sample_rate)
    min_samples = int(min_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))

    segments = []
    total_samples = audio.shape[1]

    for start in range(0, total_samples, hop_samples):
        end = min(start + segment_samples, total_samples)
        segment = audio[:, start:end]

        if segment.shape[1] >= min_samples:
            segments.append(segment)

    return segments


class EchoTTSDataset(Dataset):
    """
    Dataset for Echo-TTS fine-tuning.

    Handles loading audio files, transcriptions, and creating training samples.
    """

    def __init__(
        self,
        samples: List[TrainingSample],
        fish_ae: DAC,
        pca_state: PCAState,
        device: str = "cuda",
        max_latent_length: int = 640,
        cache_latents: bool = True,
    ):
        """
        Args:
            samples: List of TrainingSample objects
            fish_ae: Fish-S1-DAC autoencoder for encoding audio
            pca_state: PCA state for latent transformation
            device: Device for encoding
            max_latent_length: Maximum latent sequence length (640 = ~30 seconds)
            cache_latents: Whether to pre-encode and cache all latents
        """
        self.samples = samples
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.device = device
        self.max_latent_length = max_latent_length
        self.cache_latents = cache_latents

        # Pre-encode latents if caching
        self.latent_cache: Dict[str, torch.Tensor] = {}
        self.speaker_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        if cache_latents:
            self._precompute_latents()

    def _precompute_latents(self):
        """Pre-encode all audio to latents."""
        print("Pre-encoding audio to latents...")

        for i, sample in enumerate(self.samples):
            # Encode target audio
            if sample.audio_path not in self.latent_cache:
                audio = load_audio_tensor(sample.audio_path)
                audio = audio.unsqueeze(0).to(self.fish_ae.dtype).to(self.device)

                with torch.no_grad():
                    latent = ae_encode(self.fish_ae, self.pca_state, audio)
                    # Truncate to max length
                    latent = latent[:, :self.max_latent_length, :]

                self.latent_cache[sample.audio_path] = latent.cpu()

            # Encode speaker audio
            speaker_path = sample.speaker_audio_path or sample.audio_path
            if speaker_path not in self.speaker_cache:
                audio = load_audio_tensor(speaker_path)
                audio = audio.to(self.fish_ae.dtype).to(self.device)

                with torch.no_grad():
                    speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                        self.fish_ae,
                        self.pca_state,
                        audio,
                    )

                self.speaker_cache[speaker_path] = (
                    speaker_latent.cpu(),
                    speaker_mask.cpu(),
                )

            if (i + 1) % 10 == 0:
                print(f"  Encoded {i + 1}/{len(self.samples)} samples")

        print(f"Done! Cached {len(self.latent_cache)} latents, {len(self.speaker_cache)} speakers")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get target latent
        if self.cache_latents:
            latent = self.latent_cache[sample.audio_path]
        else:
            audio = load_audio_tensor(sample.audio_path)
            audio = audio.unsqueeze(0).to(self.fish_ae.dtype).to(self.device)
            with torch.no_grad():
                latent = ae_encode(self.fish_ae, self.pca_state, audio)
                latent = latent[:, :self.max_latent_length, :]
            latent = latent.cpu()

        # Get speaker latent
        speaker_path = sample.speaker_audio_path or sample.audio_path
        if self.cache_latents:
            speaker_latent, speaker_mask = self.speaker_cache[speaker_path]
        else:
            audio = load_audio_tensor(speaker_path)
            audio = audio.to(self.fish_ae.dtype).to(self.device)
            with torch.no_grad():
                speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                    self.fish_ae,
                    self.pca_state,
                    audio,
                )
            speaker_latent = speaker_latent.cpu()
            speaker_mask = speaker_mask.cpu()

        return {
            "latent": latent.squeeze(0),  # (T, 80)
            "text": sample.text,
            "speaker_latent": speaker_latent.squeeze(0),  # (S, 80)
            "speaker_mask": speaker_mask.squeeze(0),  # (S,)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader - handles variable length sequences."""
    # Find max lengths
    max_latent_len = max(item["latent"].shape[0] for item in batch)
    max_speaker_len = max(item["speaker_latent"].shape[0] for item in batch)

    # Pad latents
    latents = torch.zeros(len(batch), max_latent_len, 80)
    latent_masks = torch.zeros(len(batch), max_latent_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["latent"].shape[0]
        latents[i, :length, :] = item["latent"]
        latent_masks[i, :length] = True

    # Pad speaker latents
    speaker_latents = torch.zeros(len(batch), max_speaker_len, 80)
    speaker_masks = torch.zeros(len(batch), max_speaker_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["speaker_latent"].shape[0]
        speaker_latents[i, :length, :] = item["speaker_latent"]
        speaker_masks[i, :length] = item["speaker_mask"]

    # Collect texts
    texts = [item["text"] for item in batch]

    return {
        "latent": latents,
        "latent_mask": latent_masks,
        "speaker_latent": speaker_latents,
        "speaker_mask": speaker_masks,
        "text": texts,
    }


# ============================================================================
# Loss Function
# ============================================================================

def compute_diffusion_loss(
    model: EchoDiT,
    latent_target: torch.Tensor,
    latent_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> torch.Tensor:
    """
    Compute flow-matching diffusion loss (v-prediction).

    The model learns to predict the velocity v = noise - x_0 given x_t and t,
    where x_t = (1-t) * x_0 + t * noise.

    Args:
        model: EchoDiT model
        latent_target: Target clean latents (B, T, 80)
        latent_mask: Mask for valid latent positions (B, T)
        text_input_ids: Tokenized text (B, L)
        text_mask: Text attention mask (B, L)
        speaker_latent: Speaker reference latents (B, S, 80)
        speaker_mask: Speaker attention mask (B, S)
        min_t: Minimum timestep for sampling
        max_t: Maximum timestep for sampling

    Returns:
        Scalar loss tensor
    """
    B, T, D = latent_target.shape
    device = latent_target.device
    dtype = model.dtype

    # Sample random timesteps uniformly in [min_t, max_t]
    t = torch.rand(B, device=device) * (max_t - min_t) + min_t

    # Generate noise
    noise = torch.randn_like(latent_target)

    # Create noisy latents: x_t = (1-t) * x_0 + t * noise
    t_expanded = t[:, None, None]  # (B, 1, 1)
    x_t = (1 - t_expanded) * latent_target + t_expanded * noise

    # Target velocity: v = noise - x_0
    v_target = noise - latent_target

    # Get KV caches for conditioning (no grad needed for these)
    with torch.no_grad():
        kv_cache_text = model.get_kv_cache_text(text_input_ids, text_mask)
        kv_cache_speaker = model.get_kv_cache_speaker(speaker_latent.to(dtype))

    # Forward pass to predict velocity
    v_pred = model(
        x=x_t.to(dtype),
        t=t.to(dtype),
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        kv_cache_text=kv_cache_text,
        kv_cache_speaker=kv_cache_speaker,
    )

    # Compute MSE loss, masked to valid positions
    loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="none")

    # Apply mask and average
    loss = loss * latent_mask.unsqueeze(-1).float()

    # More numerically stable averaging
    num_valid_elements = latent_mask.sum() * D
    if num_valid_elements == 0:
        print("Warning: No valid elements in batch, returning zero loss")
        return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    loss = loss.sum() / num_valid_elements

    # Final safety check
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: NaN/Inf detected in loss computation")
        return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    return loss


# ============================================================================
# Training Loop
# ============================================================================

def training_step(
    model: EchoDiT,
    batch: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Single training step.

    Args:
        model: EchoDiT model with LoRA adapters
        batch: Batch from dataloader
        device: Device to run on

    Returns:
        Loss tensor
    """
    # Move batch to device
    latent = batch["latent"].to(device)
    latent_mask = batch["latent_mask"].to(device)
    speaker_latent = batch["speaker_latent"].to(device)
    speaker_mask = batch["speaker_mask"].to(device)

    # Encode text
    text_input_ids, text_mask = get_text_input_ids_and_mask(
        batch["text"],
        max_length=None,
        device=device,
        normalize=True,
    )

    # Compute loss
    loss = compute_diffusion_loss(
        model=model,
        latent_target=latent,
        latent_mask=latent_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
    )

    return loss


def train_epoch(
    model: EchoDiT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: EchoDiT model with LoRA
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device
        gradient_accumulation_steps: Accumulate gradients over N steps
        max_grad_norm: Maximum gradient norm for clipping
        scaler: Optional GradScaler for mixed precision

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Mixed precision forward - use new API
        with torch.amp.autocast('cuda', enabled=scaler is not None, dtype=torch.bfloat16):
            loss = training_step(model, batch, device)

            # Check for NaN before scaling
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at step {step}, skipping batch")
                continue

            loss = loss / gradient_accumulation_steps

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Optimizer step
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

                # Check gradients for NaN
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient norm detected at step {step}, skipping update")
                    optimizer.zero_grad()
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient norm detected at step {step}, skipping update")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

            optimizer.zero_grad()

            # Step scheduler after optimizer (fixed order)
            if scheduler is not None:
                scheduler.step()

    return total_loss / max(num_batches, 1)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine learning rate schedule with warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Transcription (Whisper)
# ============================================================================

def transcribe_audio_whisper(
    audio_path: str,
    model_name: str = "base",
    language: str = "en",
) -> str:
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (tiny, base, small, medium, large-v3)
        language: Language code

    Returns:
        Transcribed text
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Please install whisper: pip install openai-whisper"
        )

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)

    text = result["text"].strip()

    # Add speaker prefix if not present
    if not text.startswith("[") and "S1" not in text:
        text = "[S1] " + text

    return text


def transcribe_audio_files(
    audio_paths: List[str],
    model_name: str = "base",
    language: str = "en",
) -> Dict[str, str]:
    """
    Transcribe multiple audio files.

    Args:
        audio_paths: List of audio file paths
        model_name: Whisper model name
        language: Language code

    Returns:
        Dict mapping audio path to transcription
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Please install whisper: pip install openai-whisper"
        )

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    transcriptions = {}

    for i, path in enumerate(audio_paths):
        print(f"Transcribing {i + 1}/{len(audio_paths)}: {os.path.basename(path)}")

        result = model.transcribe(path, language=language)
        text = result["text"].strip()

        # Add speaker prefix if not present
        if not text.startswith("[") and "S1" not in text:
            text = "[S1] " + text

        transcriptions[path] = text

    return transcriptions


def transcribe_audio_files_parallel(
    audio_paths: List[str],
    model_name: str = "base",
    language: str = "en",
    num_workers: int = 4,
    batch_size: int = 8,
) -> Dict[str, str]:
    """
    Transcribe multiple audio files in parallel using multiprocessing.

    Args:
        audio_paths: List of audio file paths
        model_name: Whisper model name
        language: Language code
        num_workers: Number of parallel workers (default: 4)
        batch_size: Files per batch for progress updates (default: 8)

    Returns:
        Dict mapping audio path to transcription
    """
    try:
        from multiprocessing import Pool
        from functools import partial
    except ImportError as e:
        raise ImportError(
            f"Please install required packages: {e}"
        )

    def transcribe_single(path: str, model_name: str, language: str) -> tuple:
        """Helper function to transcribe a single file."""
        try:
            import whisper
            # Each worker loads its own model instance
            model = whisper.load_model(model_name)
            result = model.transcribe(path, language=language)
            text = result["text"].strip()

            # Add speaker prefix if not present
            if not text.startswith("[") and "S1" not in text:
                text = "[S1] " + text

            return (path, text, None)
        except Exception as e:
            return (path, None, str(e))

    print(f"Transcribing {len(audio_paths)} files with Whisper '{model_name}'...")
    print(f"Using {num_workers} parallel workers")
    print("This will be MUCH faster than sequential processing!\n")

    transcriptions = {}
    errors = []

    # Create partial function with fixed model and language
    transcribe_fn = partial(transcribe_single, model_name=model_name, language=language)

    # Process in batches for progress updates
    with Pool(processes=num_workers) as pool:
        for batch_start in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[batch_start:batch_start + batch_size]

            # Process batch in parallel
            results = pool.map(transcribe_fn, batch_paths)

            # Collect results
            for path, text, error in results:
                if error:
                    errors.append((path, error))
                    print(f"  ⚠️  Error transcribing {os.path.basename(path)}: {error}")
                else:
                    transcriptions[path] = text

            # Progress update
            completed = min(batch_start + batch_size, len(audio_paths))
            print(f"Progress: {completed}/{len(audio_paths)} files transcribed")

    print(f"\n{'='*50}")
    print(f"Transcription complete!")
    print(f"  Successful: {len(transcriptions)}")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*50}")

    if errors:
        print("\nFiles with errors:")
        for path, error in errors[:10]:  # Show first 10
            print(f"  - {os.path.basename(path)}")

    return transcriptions


# ============================================================================
# Utilities
# ============================================================================

def prepare_samples_from_directory(
    audio_dir: str,
    transcriptions: Optional[Dict[str, str]] = None,
    extensions: Tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg", ".m4a"),
    segment_duration: float = 20.0,
    min_duration: float = 3.0,
) -> List[TrainingSample]:
    """
    Prepare training samples from a directory of audio files.

    Args:
        audio_dir: Directory containing audio files
        transcriptions: Optional dict of audio_path -> text. If None, will need transcription.
        extensions: Audio file extensions to include
        segment_duration: Target segment duration for splitting long files
        min_duration: Minimum duration to keep

    Returns:
        List of TrainingSample objects
    """
    audio_dir = Path(audio_dir)
    audio_files = []

    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))

    samples = []

    for audio_path in sorted(audio_files):
        path_str = str(audio_path)

        # Get or generate transcription
        if transcriptions and path_str in transcriptions:
            text = transcriptions[path_str]
        else:
            text = None  # Will need to be transcribed

        if text is not None:
            samples.append(TrainingSample(
                audio_path=path_str,
                text=text,
                speaker_audio_path=None,  # Use same audio as speaker ref
            ))

    return samples


# Import math for cosine schedule
import math
