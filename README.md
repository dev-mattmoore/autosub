# AutoSub

AutoSub is a Python-based tool for generating subtitles from video files. It uses OpenAI's Whisper model for transcription and outputs subtitles in the SRT format.

## Features
- Extracts audio from video files using FFmpeg.
- Transcribes audio to text using Whisper.
- Generates SRT subtitle files.
- Supports batch processing of all videos in a folder.
- Parallel processing with automatic job count based on system memory and CPU.
- Flexible subtitle naming (language, forced, SDH, default flags).

## Requirements
- Python 3.12+
- FFmpeg
- [OpenAI Whisper](https://github.com/openai/whisper)
- [psutil](https://pypi.org/project/psutil/)
- [srt](https://pypi.org/project/srt/)

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

You can process a single video file or all video files in a folder.

### Single File
```bash
python autosub_cli.py -i <video_file>
```

### Batch Folder
```bash
python autosub_cli.py -i <folder_with_videos>
```

## CLI Options

- `--input`, `-i`: Path to the input video file or folder.
- `--model`, `-m`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). Default: `base`.
- `--language`, `-l`: Language code (e.g., `en`, `es`, `fr`). Default: auto-detect.
- `--output-format`, `-f`: Subtitle output format (currently only `srt` supported).
- `--force`: Force overwrite if subtitle already exists.
- `--default`: Mark subtitles as default.
- `--forced`: Mark subtitles as forced (non-native dialogue only).
- `--sdh`: Include SDH (Subtitles for the Deaf and Hard of Hearing).
- `--jobs`, `-j`: Number of parallel processes to use (auto-calculated if not set).

## Output Naming

Output subtitle files are named using the following pattern:
```
<basename>.<language>[.forced][.sdh][.default].srt
```
For example:
```
movie.en.forced.sdh.default.srt
```

## Example Usage

Process a single video:
```bash
python autosub_cli.py -i "/path/to/video.mkv" --model base --language en --forced --sdh
```

Process all videos in a folder with automatic parallelization:
```bash
python autosub_cli.py -i "/path/to/folder"
```
