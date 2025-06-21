import os
import argparse
import whisper
import subprocess
import srt
import datetime
from pathlib import Path

VIDEO_EXTS = ('.mkv', '.mp4', '.avi', '.mov')

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", audio_path
    ], check=True)

def transcribe(audio_path, model_name="base", language=None):
    model = whisper.load_model(model_name)
    return model.transcribe(audio_path, language=language)

def write_srt(result, output_path):
    subtitles = []
    for i, seg in enumerate(result['segments']):
        start = datetime.timedelta(seconds=seg['start'])
        end = datetime.timedelta(seconds=seg['end'])
        subtitles.append(srt.Subtitle(index=i+1, start=start, end=end, content=seg['text']))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))

def build_output_path(input_path, language, default, forced, sdh, fmt='srt'):
    base = os.path.splitext(input_path)[0]
    suffixes = [language or 'und']
    if forced:
        suffixes.append('forced')
    if sdh:
        suffixes.append('sdh')
    if default:
        suffixes.append('default')
    return f"{base}." + ".".join(suffixes) + f".{fmt}"

def process_file(path, args):
    output_path = build_output_path(path, args.language, args.default, args.forced, args.sdh, args.output_format)
    audio_path = os.path.splitext(path)[0] + "_audio.wav"

    if os.path.exists(output_path) and not args.force:
        print(f"⏩ Skipping (exists): {Path(path).name}")
        return

    print(f"🎞️  Processing: {Path(path).name}")
    extract_audio(path, audio_path)
    result = transcribe(audio_path, model_name=args.model, language=args.language)
    write_srt(result, output_path)
    os.remove(audio_path)
    print(f"✅ Done: {Path(output_path).name}")

def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video(s) using Whisper")
    parser.add_argument('--input', '-i', required=True, help='Path to input video file or folder')
    parser.add_argument('--model', '-m', default='base', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--language', '-l', default=None, help='Language code (e.g., en, es, fr)')
    parser.add_argument('--output-format', '-f', default='srt', choices=['srt'], help='Subtitle output format')
    parser.add_argument('--force', action='store_true', help='Force overwrite if subtitle already exists')
    parser.add_argument('--default', action='store_true', help='Mark subtitle as default')
    parser.add_argument('--forced', action='store_true', help='Mark subtitle as forced (used only for non-native dialogue)')
    parser.add_argument('--sdh', action='store_true', help='Include SDH (Subtitles for Deaf and Hard of Hearing) flag')

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        process_file(str(input_path), args)
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        if not files:
            print(f"⚠️  No video files found in {input_path}")
            return
        for file in files:
            try:
                process_file(str(file), args)
            except Exception as e:
                print(f"❌ Error processing {file.name}: {e}")
    else:
        print("❌ Input path is invalid.")

if __name__ == "__main__":
    main()
