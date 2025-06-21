# AutoSub

AutoSub is a Python-based tool for generating subtitles from video files. It uses OpenAI's Whisper model for transcription and outputs subtitles in the SRT format.

## Features
- Extracts audio from video files using FFmpeg.
- Transcribes audio to text using Whisper.
- Generates SRT subtitle files.

## Requirements
- Python 3.12+
- FFmpeg

## Installation
1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script with a video file as input:
```bash
python autosub_cli.py -i <video_file>
```

## CLI Options

The CLI now supports the following options:

- `--input` or `-i`: Path to the input video file (required).
- `--model` or `-m`: Specify the Whisper model size (e.g., tiny, base, small, medium, large). Default is `base`.
- `--language` or `-l`: Language code for transcription (e.g., en, es, fr). Default is auto-detection.
- `--output-format` or `-f`: Subtitle output format. Currently supports `srt`. Default is `srt`.
- `--force`: Force overwrite if the subtitle file already exists. Accepts `True` or `False`. Default is `False`.
- `--default`: Mark subtitles as default. Accepts `True` or `False`. Default is `False`.
- `--forced`: Mark subtitles as forced. Accepts `True` or `False`. Default is `False`.
- `--sdh`: Include SDH (Subtitles for the Deaf and Hard of Hearing). Accepts `True` or `False`. Default is `False`.

### Example Usage

```bash
python autosub_cli.py -i "/path/to/video.mkv" --model base --language en --forced True --sdh True
```

## License
This project is licensed under the MIT License.
