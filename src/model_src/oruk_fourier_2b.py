"""AudioBench model adapter for pinned Oruk Fourier 2B (native HF BF16).

Follows the AudioLLMs/AudioBench `model_src` contract:
  - `oruk_fourier_2b_model_loader(self)`
  - `oruk_fourier_2b_model_generation(self, sample)`

Downloads strictly the 7 native root checkpoint files of
`oruk/oruk-fourier-2b@8c2a8871f1304aa2f634da4fb95e24438c4945b9` via
`huggingface_hub.snapshot_download` (canonical HF Hub attribution),
verifies `model.safetensors` SHA-256 before loading, and applies
deterministic 20-28 s minimum-RMS energy chunking for recordings > 30 s.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "oruk/oruk-fourier-2b"
MODEL_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
NATIVE_ROOT_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
EXPECTED_WEIGHT_BYTES = 4_076_193_080
EXPECTED_WEIGHT_SHA256 = (
    "e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c"
)
TARGET_SR = 16_000
MIN_CHUNK_SAMPLES = 8_000


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _chunk_audio_energy(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    min_s: float = 20.0,
    max_s: float = 28.0,
    hard_max_s: float = 30.0,
) -> list[np.ndarray]:
    n = int(wav.shape[0])
    if n == 0:
        return [np.zeros(MIN_CHUNK_SAMPLES, dtype=np.float32)]
    hard_max = int(round(hard_max_s * sr))
    min_n = int(round(min_s * sr))
    max_n = int(round(max_s * sr))
    win_n = max(1, int(round(0.2 * sr)))
    hop_n = max(1, int(round(0.02 * sr)))

    chunks: list[np.ndarray] = []
    pos = 0
    while pos < n:
        if n - pos <= hard_max:
            seg = wav[pos:n]
            pos = n
        else:
            s0 = pos + min_n
            s1 = min(n, pos + max_n)
            if s1 - s0 <= win_n:
                cut = s1
            else:
                band = wav[s0:s1]
                sq = np.empty(band.shape[0] + 1, dtype=np.float64)
                sq[0] = 0.0
                np.cumsum(band.astype(np.float64) ** 2, out=sq[1:])
                starts = np.arange(0, band.shape[0] - win_n + 1, hop_n)
                energies = (sq[starts + win_n] - sq[starts]) / float(win_n)
                cut = s0 + int(starts[int(np.argmin(energies))]) + (win_n // 2)
            seg = wav[pos:cut]
            pos = cut
        if seg.shape[0] < MIN_CHUNK_SAMPLES:
            seg = np.pad(seg, (0, MIN_CHUNK_SAMPLES - seg.shape[0]), mode="constant")
        chunks.append(np.ascontiguousarray(seg, dtype=np.float32))
    return chunks


def oruk_fourier_2b_model_loader(self) -> None:
    from huggingface_hub import snapshot_download

    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Oruk Fourier 2B requires transformers>=5.24.0 with "
            "Qwen3-ASR / AutoModelForMultimodalLM support."
        ) from exc

    local_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=NATIVE_ROOT_FILES,
        )
    )
    wbytes, wsha = _sha256_file(local_dir / "model.safetensors")
    if wbytes != EXPECTED_WEIGHT_BYTES or wsha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"Pinned checkpoint mismatch: bytes={wbytes} sha256={wsha}"
        )

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    self.processor = AutoProcessor.from_pretrained(str(local_dir), local_files_only=True)
    self.processor.tokenizer.padding_side = "left"
    self.model = (
        AutoModelForMultimodalLM.from_pretrained(
            str(local_dir),
            local_files_only=True,
            dtype=dtype,
        )
        .to(self.device)
        .eval()
    )
    logger.info(
        "Loaded pinned %s@%s on %s (%s).",
        MODEL_ID,
        MODEL_REVISION[:8],
        self.device,
        dtype,
    )


def oruk_fourier_2b_model_generation(self, sample: dict) -> str:
    if sample.get("task_type") != "ASR":
        raise NotImplementedError(
            f"Oruk Fourier 2B is an English ASR model; unsupported task_type={sample.get('task_type')!r}"
        )

    audio = sample["audio"]
    wav = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[1] < wav.shape[0] else 0)
    if sr != TARGET_SR:
        try:
            import soxr

            wav = soxr.resample(wav, sr, TARGET_SR).astype(np.float32, copy=False)
        except ImportError:
            try:
                import librosa

                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR).astype(
                    np.float32, copy=False
                )
            except ImportError:
                from math import gcd
                from scipy.signal import resample_poly

                g = gcd(sr, TARGET_SR)
                wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(
                    np.float32, copy=False
                )

    chunks = _chunk_audio_energy(wav, sr=TARGET_SR)
    texts: list[str] = []
    batch_size = 16
    model_dtype = getattr(self.model, "dtype", torch.float32)
    with torch.inference_mode():
        for off in range(0, len(chunks), batch_size):
            batch = chunks[off : off + batch_size]
            inputs = self.processor.apply_transcription_request(
                audio=batch,
                language=["English"] * len(batch),
                return_tensors="pt",
                processor_kwargs={"padding": True},
            ).to(self.device, model_dtype)
            output_ids = self.model.generate(**inputs)
            gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            decoded = self.processor.decode(gen_ids, return_format="transcription_only")
            texts.extend(t.strip() for t in decoded if t and t.strip())
    return " ".join(texts).strip()
