# Standard imports
from functools import lru_cache
from pathlib import Path

# Third-party imports
import torch
import torchaudio
from torch import Tensor

from demucs.apply import apply_model
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model


@lru_cache(maxsize=1)
def _get_htdemucs_model() -> HTDemucs:
    """
    Load HTDemucs 6-stem model and move it to the appropriate device.
    This function's return is memoized to avoid re-running unnecessarily.

    Returns:
        HTDemucs 6-stem model loaded onto the appropriate device.
    """
    model = get_model('htdemucs_6s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return model.to(device)


def _isolate_guitar_waveform(waveform: Tensor) -> Tensor:
    """
    Isolate the guitar waveform from <waveform>.

    Args:
        waveform: Audio mix to be separated (2D Tensor).

    Returns:
        Isolated guitar waveform (2D Tensor).
    """
    model = _get_htdemucs_model()
    device = next(model.parameters()).device
    waveform = waveform.to(device)

    # Add null batch-dimension to waveform for Demucs model input.
    separated_waveforms = apply_model(model, waveform[None], split=True)[0]
    guitar_index = list(model.sources).index("guitar")

    # Remove batch dimension
    guitar_waveform = separated_waveforms[guitar_index].squeeze(0)

    return guitar_waveform


def _save_isolated_track(waveform: Tensor, track_name: str,
                         sample_rate: float | int, base_path: Path) -> Path:
    """
    Save <waveform> as a .WAV file under the same directory as <base_path>.

    Args:
        waveform: Isolated waveform to save (2D Tensor).
        track_name: Name of isolated track.
        sample_rate: Sample rate of isolated waveform.
        base_path: Path to original non-isolated audio file.

    Returns:
        Path to isolated track.
    """
    save_directory = base_path.parent
    new_filename = f'{base_path.stem}_{track_name}.wav'
    track_path = save_directory / new_filename

    torchaudio.save(track_path, waveform, sample_rate)

    return track_path


def isolate_guitar(audio_path: Path) -> Path:
    """
    Isolate the guitar track from the file at <audio_path>.

    Args:
        audio_path: Path to audio file.

    Returns:
        Path to new isolated guitar audio file.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    guitar_waveform = _isolate_guitar_waveform(waveform)

    guitar_path = _save_isolated_track(
        guitar_waveform,
        'guitar',
        sample_rate,
        audio_path
    )

    return guitar_path
