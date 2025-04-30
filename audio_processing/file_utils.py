# Standard imports
from pathlib import Path

# Third-party imports
import torchaudio
from torch import Tensor

from pydub import AudioSegment


def convert_to_wav(input_path: Path, keep_input_file: bool = False) -> Path:
    """
    Convert the audio file at <input_path> to .WAV format.

    Args:
        input_path: Path to input audio file.
        keep_input_file: If false, delete the input file after conversion.

    Returns:
        Path to the converted .WAV file.
    """
    if input_path.suffix.lower() == '.wav':
        return input_path

    output_path = input_path.with_suffix('.wav')
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format='wav')

    if not keep_input_file:
        input_path.unlink()

    return output_path


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
