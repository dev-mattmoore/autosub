# transcribe_video.py
import os
import sys
import whisper
import subprocess
import srt
import datetime

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", audio_path
    ], check=True)

def transcribe(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result

def write_srt(result, output_path):
    segments = result['segments']
    subtitles = []
    for i, seg in enumerate(segments):
        start = datetime.timedelta(seconds=seg['start'])
        end = datetime.timedelta(seconds=seg['end'])
        subtitles.append(srt.Subtitle(index=i+1, start=start, end=end, content=seg['text']))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))

def main(video_path):
    base = os.path.splitext(video_path)[0]
    audio_path = f"{base}_audio.wav"
    srt_path = f"{base}.srt"

    print(f"🟡 Extracting audio from {video_path}...")
    extract_audio(video_path, audio_path)

    print("🟡 Transcribing audio...")
    result = transcribe(audio_path)

    print(f"🟢 Writing subtitles to {srt_path}")
    write_srt(result, srt_path)

    os.remove(audio_path)
    print("✅ Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_video.py <video_file>")
        sys.exit(1)
    main(sys.argv[1])
