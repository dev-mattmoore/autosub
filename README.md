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
python transcribe_video.py <video_file>
```

## License
This project is licensed under the MIT License.
