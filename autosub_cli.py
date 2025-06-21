# autosub_cli.py
import os
import argparse
import whisper
import subprocess
import srt
import datetime

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

def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video using Open AI Whisper")
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--model', '-m', default='base', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--language', '-l', default='en', help='Language code (e.g., en, es, fr)')
    parser.add_argument('--output-format', '-f', default='srt', choices=['srt'], help='Subtitle output format')
    parser.add_argument('--force', action='store_true', help='Force overwrite if subtitle already exists')

    args = parser.parse_args()

    base = os.path.splitext(args.input)[0]
    audio_path = f"{base}_audio.wav"
    srt_path = f"{base}.{args.language}.srt"

    if os.path.exists(srt_path) and not args.force:
        print(f"⚠️  Subtitle already exists: {srt_path} (use --force to overwrite)")
        return

    print(f"🟡 Extracting audio from {args.input}...")
    extract_audio(args.input, audio_path)

    print(f"🟡 Transcribing with model '{args.model}'...")
    result = transcribe(audio_path, model_name=args.model, language=args.language)

    print(f"🟢 Writing subtitles to {srt_path}...")
    write_srt(result, srt_path)

    os.remove(audio_path)
    print("✅ Done.")

if __name__ == "__main__":
    main()
