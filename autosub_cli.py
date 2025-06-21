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

def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video using Whisper")
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--model', '-m', default='base', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--language', '-l', default=None, help='Language code (e.g., en, es, fr)')
    parser.add_argument('--output-format', '-f', default='srt', choices=['srt'], help='Subtitle output format')
    parser.add_argument('--force', action='store_true', help='Force overwrite if subtitle already exists')
    parser.add_argument('--default', action='store_true', help='Mark subtitle as default')
    parser.add_argument('--forced', action='store_true', help='Mark subtitle as forced (used only for non-native dialogue)')
    parser.add_argument('--sdh', action='store_true', help='Include SDH (Subtitles for Deaf and Hard of Hearing) flag')

    args = parser.parse_args()

    output_path = build_output_path(args.input, args.language, args.default, args.forced, args.sdh, args.output_format)
    audio_path = os.path.splitext(args.input)[0] + "_audio.wav"

    if os.path.exists(output_path) and not args.force:
        print(f"⚠️  Subtitle already exists: {output_path} (use --force to overwrite)")
        return

    print(f"🟡 Extracting audio from {args.input}...")
    extract_audio(args.input, audio_path)

    print(f"🟡 Transcribing with model '{args.model}'...")
    result = transcribe(audio_path, model_name=args.model, language=args.language)

    print(f"🟢 Writing subtitles to {output_path}...")
    write_srt(result, output_path)

    os.remove(audio_path)
    print("✅ Done.")

if __name__ == "__main__":
    main()