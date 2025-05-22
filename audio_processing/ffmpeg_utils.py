# Standard imports
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

# Local imports
from .errors import (
    AudioDemuxingError,
    AudioFormatUnsupportedError,
    AudioMuxingError,
    FFmpegExecutionError,
    FFmpegNotInstalledError
)


# Constants
AUDIO_FORMATS_SAVE_PATH = Path('audio_formats.json')


def _verify_ffmpeg_installation() -> None:
    """Verify that ffmpeg is installed and available in the system PATH.

    Raises:
        FFmpegNotInstalledError: If ffmpeg is not installed or cannot be found.
    """
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception as e:
        raise FFmpegNotInstalledError(
            f'FFmpeg is not installed or could not be found in system PATH: {e}'
        )


def _build_audio_formats_dict() -> dict[str, list[bool]]:
    """Parse the output of `ffmpeg -formats` and return a dictionary mapping
    audio format names to a list indicating [demuxable, muxable] support.

    Returns:
        dict[str, list[bool]]: Dictionary where each key is an audio
        format name, and the value is a list of booleans:
            - True if format is demuxable (readable)
            - True if format is muxable (writable)

    Raises:
        FFmpegNotInstalledError: If ffmpeg is not installed or cannot be found.
        FFmpegExecutionError: If running the `ffmpeg -formats` command fails.
    """
    _verify_ffmpeg_installation()

    try:
        result = subprocess.run(
            ['ffmpeg', '-formats'],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception as e:
        raise FFmpegExecutionError(
            f'Failed to run `ffmpeg -formats` command: {e}'
        )

    formats = {}
    for line in result.stdout.splitlines():
        if match := re.match(
                r'^\s*([D\s])([E\s])\s+(\S+)\s+.*$',
                line
        ):
            demuxable = match.group(1) == 'D'
            muxable = match.group(2) == 'E'
            names = match.group(3).split(',')

            for name in names:
                formats[name.strip()] = [demuxable, muxable]

    return formats


def _save_audio_formats_dict(save_path: Path = AUDIO_FORMATS_SAVE_PATH) -> None:
    """Generate the audio formats dictionary from `ffmpeg -formats` and save it
    as a JSON file.

    Args:
        save_path (Path, optional): The file path where the JSON data should be
        saved. Defaults to `AUDIO_FORMATS_SAVE_PATH`.

    Raises:
        FFmpegNotInstalledError: If FFmpeg is not installed or cannot be found.
        FFmpegExecutionError: If running the `ffmpeg -formats` command fails.
    """
    audio_format_dict = _build_audio_formats_dict()

    with open(save_path, 'w') as outfile:
        json.dump(audio_format_dict, outfile, indent=4) # NOQA: Expected type...


@lru_cache(maxsize=1)
def _load_audio_formats_dict(save_path: Path = AUDIO_FORMATS_SAVE_PATH) \
        -> dict[str, list[bool]]:
    """Load the audio formats dictionary from a JSON file.

    Args:
        save_path (Path, optional): Path to the JSON file containing the
            saved audio format dictionary. Defaults to
            `AUDIO_FORMATS_SAVE_PATH`.

    Returns:
        dict[str, list[bool]]: Dictionary where each key is an audio
        format name, and the value is a list of booleans:
            - True if the format is demuxable (readable)
            - True if the format is muxable (writable)

    Raises:
        FileNotFoundError: If the JSON file does not exist at the specified
            path.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not save_path.exists():
        raise FileNotFoundError(f'File at {save_path} cannot be found.')

    with open(save_path, 'r') as f:
        data = json.load(f)

    return data


def _validate_audio_formats(input_format: str, input_path: Path,
                            output_format: str, output_path: Path) -> None:
    """Validate if the input and output audio formats are supported and raise
    specific errors if they are not.

    Args:
        input_format (str): Input audio format (e.g., 'mp3').
        input_path (Path): Path to the input audio file.
        output_format (str): Desired output audio format (e.g., 'wav').
        output_path (Path): Path where the output file will be saved.

    Raises:
        AudioFormatUnsupportedError: If input or output format is unsupported.
        AudioDemuxingError: If input format is not demuxable.
        AudioMuxingError: If output format is not muxable.
    """
    supported_formats = _load_audio_formats_dict()

    if input_format not in supported_formats:
        raise AudioFormatUnsupportedError(input_format, input_path)

    elif not supported_formats[input_format][0]:
        raise AudioDemuxingError(input_format, input_path)

    elif output_format not in supported_formats:
        raise AudioFormatUnsupportedError(output_format, output_path)

    elif not supported_formats[output_format][1]:
        raise AudioMuxingError(output_format, output_path)
