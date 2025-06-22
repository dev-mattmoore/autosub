import os
import argparse
import whisper
import subprocess
import srt
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import psutil

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

def process_file(path, args_dict):
    language = args_dict['language']
    model = args_dict['model']
    output_format = args_dict['output_format']
    force = args_dict['force']
    default = args_dict['default']
    forced = args_dict['forced']
    sdh = args_dict['sdh']

    output_path = build_output_path(path, language, default, forced, sdh, output_format)
    audio_path = os.path.splitext(path)[0] + "_audio.wav"

    if os.path.exists(output_path) and not force:
        print(f"⏩ Skipping (exists): {Path(path).name}")
        return

    print(f"🎞️  Processing: {Path(path).name}")
    try:
        extract_audio(path, audio_path)
        result = transcribe(audio_path, model_name=model, language=language)
        write_srt(result, output_path)
        os.remove(audio_path)
        print(f"✅ Done: {Path(output_path).name}")
    except Exception as e:
        print(f"❌ Failed to process {Path(path).name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video(s) using OpenAI Whisper")
    parser.add_argument('--input', '-i', required=True, help='Path to input video file or folder')
    parser.add_argument('--model', '-m', default='base', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--language', '-l', default=None, help='Language code (e.g., en, es, fr)')
    parser.add_argument('--output-format', '-f', default='srt', choices=['srt'], help='Subtitle output format')
    parser.add_argument('--force', action='store_true', help='Force overwrite if subtitle already exists')
    parser.add_argument('--default', action='store_true', help='Mark subtitle as default')
    parser.add_argument('--forced', action='store_true', help='Mark subtitle as forced (non-native dialogue only)')
    parser.add_argument('--sdh', action='store_true', help='Include SDH (subtitles for the deaf and hard of hearing)')
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help='Number of parallel processes to use (auto-calculated if not set)')

    args = parser.parse_args()

    # Adjust job count based on available memory
    if args.jobs is None:
        total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Rough estimate: 2GB per process (adjust as needed)
        est_jobs = max(1, min(multiprocessing.cpu_count(), int(total_mem_gb // 2)))
        args.jobs = est_jobs
        print(f"🧠 Detected ~{total_mem_gb:.1f} GB RAM -> Using {args.jobs} parallel jobs")

    input_path = Path(args.input)

    if input_path.is_file():
        simple_args = {
            'language': args.language,
            'model': args.model,
            'output_format': args.output_format,
            'force': args.force,
            'default': args.default,
            'forced': args.forced,
            'sdh': args.sdh
        }
        process_file(str(input_path), simple_args)
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        if not files:
            print(f"⚠️  No video files found in {input_path}")
            return

        print(f"🚀 Processing {len(files)} video(s) with {args.jobs} parallel jobs")

        simple_args = {
            'language': args.language,
            'model': args.model,
            'output_format': args.output_format,
            'force': args.force,
            'default': args.default,
            'forced': args.forced,
            'sdh': args.sdh
        }

        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(process_file, str(file), simple_args) for file in files]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Error during batch processing: {e}")
    else:
        print("❌ Input path is invalid.")

if __name__ == "__main__":
    main()
