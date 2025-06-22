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

- `--input`, `-i`: Path to the input video file.
- `--model`, `-m`: Whisper model size (e.g., tiny, base, small, medium, large).
- `--language`, `-l`: Language code (e.g., en, es, fr).
- `--output-format`, `-f`: Subtitle output format (default: srt).
- `--force`: Force overwrite if subtitle already exists.
- `--default`: Mark subtitles as default.
- `--forced`: Mark subtitles as forced.
- `--sdh`: Include SDH (Subtitles for the Deaf and Hard of Hearing).
- `--jobs`: Number of parallel processes to use (default: half of CPU cores).

### Example Usage

```bash
python autosub_cli.py -i "/path/to/video.mkv" --model base --language en --forced True --sdh True
```

## License
This project is licensed under the MIT License.
