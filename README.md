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
- FFmpeg (must be in your PATH)
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

You can process a single video file or all video files in a folder. All logs are written to `autosub.log` by default, and also shown in the console.

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
- `--output-format`, `-f`: Subtitle output format (`srt` or `vtt`). Default: `srt`.
- `--force`: Force overwrite if subtitle already exists.
- `--default`: Mark subtitles as default.
- `--forced`: Mark subtitles as forced (non-native dialogue only).
- `--sdh`: Include SDH (Subtitles for the Deaf and Hard of Hearing).
- `--jobs`, `-j`: Number of parallel processes to use (auto-calculated if not set).
- `--dry-run`: Only show what would be processed, do not generate output.
- `--logfile`: Path to log file (default: `autosub.log`).
- `--no-logfile`: Disable all logging to file (only console output).
- `--no-color`: Disable colorized console output.
- `--output-dir`: Optional path to output subtitles separately from input.
- `--audio-only`: Extract audio only as .wav and skip transcription.
- `--audio-output-dir`: Optional path to save extracted audio separately from input.
- `--quiet-filenames`: Suppress per-file progress logging in batch mode.
- `--max-retries`: Max retry attempts per file on failure (default: 3).
- `--max-backoff`: Maximum wait time (in seconds) between retries (default: 30).
- `--print-config`: Print merged configuration from CLI and config file, then exit.
- `--batch-size`: Number of files to process at once in batch mode (0 = no limit).
- `--batch-delay`: Delay in seconds between batches (default: 0).
- `--postprocess-local`: Enable local grammar correction of subtitles using a lightweight language model (requires happytransformer and T5 model).

## Configuration File

You can set default options in a `~/.autosubrc` config file (TOML format). CLI arguments always override config file values. Use `--print-config` to show the effective configuration.

Example `~/.autosubrc`:

```toml
model = "base"
language = "en"
output_format = "srt"
output_dir = ""
default = false
forced = false
sdh = false
audio_only = false
audio_output_dir = ""
quiet_filenames = false
dry_run = false
max_retries = 3
max_backoff = 30
logfile = "autosub.log"
no_logfile = false
no_color = false
jobs = 0
batch_size = 1
batch_delay = 30
postprocess_local = false
```

## Output Naming

Output subtitle files are named using the following pattern:
```
<basename>.<language>[.forced][.sdh][.default].<format>
```
For example:
```
movie.en.forced.sdh.default.srt
movie.en.forced.sdh.default.vtt
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

## Dry Run Mode

To preview which files would be processed without generating output, use the `--dry-run` flag:
```bash
python autosub_cli.py -i "/path/to/folder" --dry-run
```

## Logging

By default, logs are written to a timestamped file (e.g., `autosub-YYYY-MM-DD_HH-MM-SS.log`) and to the console. A symlink `autosub-latest.log` always points to the latest log file. You can change the log file location with the `--logfile` option, or use `--no-logfile` to disable file logging and only log to the console.

Console output is colorized by default for better readability. Use `--no-color` to disable colorized output.

## License

This project is licensed under the MIT License.
