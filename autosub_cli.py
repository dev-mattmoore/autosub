import os
import argparse
import whisper
import subprocess
import srt
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import psutil
import logging
import sys
from colorama import init, Fore
import shutil
import time
import concurrent.futures
import toml
from happytransformer import HappyTextToText

MKVMERGE_AVAILABLE = shutil.which("mkvmerge") is not None

# Global variable to control colorized output
USE_COLOR = True

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")


def extract_audio(video_path, audio_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            audio_path,
        ],
        check=True,
    )


def transcribe(audio_path, model_name="base", language=None):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)
    detected_lang = language
    if language is None:
        detected_lang = result.get("language", "und")
        print(f"🧠 Detected language: {detected_lang}")
        logging.getLogger().info(f"Detected language for {audio_path}: {detected_lang}")
    return result, detected_lang


def write_subtitle(result, output_path, fmt):
    subtitles = []
    for i, seg in enumerate(result["segments"]):
        start = timedelta(seconds=seg["start"])
        end = timedelta(seconds=seg["end"])
        subtitles.append(
            srt.Subtitle(index=i + 1, start=start, end=end, content=seg["text"])
        )
    subtitle_text = srt.compose(subtitles)

    if fmt == "vtt":
        subtitle_text = "WEBVTT\n\n" + subtitle_text.replace(",", ".")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(subtitle_text)


def set_mkv_subtitle_default(video_path, forced=False, sdh=False):
    """
    Sets the default subtitle track flag on the most recently added subtitle track in an MKV file,
    with optional forced and SDH flags.

    This function uses `mkvmerge` to list subtitle tracks in the specified MKV file and assumes
    the last subtitle track is the one just added. It then uses `mkvpropedit` to set the
    "default" flag on that track. Additionally, it can set the "forced" flag and name the track as "SDH".

    Args:
        video_path (str): The path to the MKV video file.
        forced (bool): If True, sets the forced flag on the subtitle track.
        sdh (bool): If True, sets the name of the subtitle track to "SDH".

    Logs:
        - Warning if no subtitle tracks are found or if an error occurs.
        - Info when the default subtitle track is successfully set.

    Raises:
        None. All exceptions are caught and logged as warnings.
    """
    if not MKVMERGE_AVAILABLE:
        logging.getLogger().info(
            "mkvmerge not found; skipping MKV subtitle flag update.\n"
            "To enable automatic marking of default subtitle tracks in MKV files, install mkvtoolnix:\n"
            "  Linux:   sudo apt install mkvtoolnix\n"
            "  macOS:   brew install mkvtoolnix\n"
            "  Windows: Download from https://mkvtoolnix.download/"
        )
        return

    try:
        result = subprocess.run(
            ["mkvmerge", "-i", video_path],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        subtitle_tracks = [
            line for line in lines if "subtitles" in line.lower()
        ]
        if not subtitle_tracks:
            logging.getLogger().warning(f"No subtitle tracks found in {video_path}")
            return

        # Assume the last subtitle track is the one just added
        last_track_line = subtitle_tracks[-1]
        track_id = last_track_line.split(":")[0].strip().split()[-1]

        edit_cmd = [
            "mkvpropedit",
            video_path,
            "--edit", f"track:{track_id}",
            "--set", "flag-default=1"
        ]
        if forced:
            edit_cmd.extend(["--set", "flag-forced=1"])
        if sdh:
            edit_cmd.extend(["--set", 'name=SDH'])

        subprocess.run(edit_cmd, check=True)
        logging.getLogger().info(f"Set default subtitle track {track_id} in {video_path}")
    except Exception as e:
        logging.getLogger().warning(f"Could not set default subtitle flag: {e}")


def build_output_path(input_path, language, default, forced, sdh, fmt="srt", output_dir=None):
    base_name = Path(input_path).stem
    suffixes = [language or "und"]
    if forced:
        suffixes.append("forced")
    if sdh:
        suffixes.append("sdh")
    if default:
        suffixes.append("default")
    filename = f"{base_name}." + ".".join(suffixes) + f".{fmt}"
    return str(Path(output_dir) / filename) if output_dir else f"{os.path.splitext(input_path)[0]}." + ".".join(suffixes) + f".{fmt}"


def setup_logging(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []

    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def console_print(msg, level="info"):
    if not USE_COLOR:
        print(msg)
        return
    if level == "warning":
        print(Fore.YELLOW + msg)
    elif level == "success":
        print(Fore.GREEN + msg)
    elif level == "error":
        print(Fore.RED + msg)
    elif level == "info-cyan":
        print(Fore.CYAN + msg)
    else:
        print(msg)


def process_file(path, args_dict):
    language = args_dict["language"]
    model = args_dict["model"]
    output_format = args_dict["output_format"]
    force = args_dict["force"]
    default = args_dict["default"]
    forced = args_dict["forced"]
    sdh = args_dict["sdh"]
    dry_run = args_dict.get("dry_run", False)
    output_dir = args_dict.get("output_dir")
    audio_only = args_dict.get("audio_only", False)
    audio_output_dir = args_dict.get("audio_output_dir")
    logger = logging.getLogger()

    max_retries = args_dict.get("max_retries", 3)
    max_backoff = args_dict.get("max_backoff", 30)
    attempts = 0

    # Ensure output directories exist
    for dir_path in (output_dir, audio_output_dir):
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    base_audio_name = Path(path).stem + "_audio.wav"
    audio_path = str(Path(audio_output_dir) / base_audio_name) if audio_output_dir else os.path.splitext(path)[0] + "_audio.wav"

    if dry_run:
        output_path = build_output_path(path, language, default, forced, sdh, output_format, output_dir)
        msg = f"📝 Would process: {Path(path).name} -> {Path(output_path).name}"
        console_print(msg, "warning")
        logger.info(msg)
        return

    if os.path.exists(build_output_path(path, language, default, forced, sdh, output_format, output_dir)) and not force:
        msg = f"⏩ Skipping (exists): {Path(path).name}"
        console_print(msg, "warning")
        logger.info(msg)
        return

    msg = f"🎞️  Processing: {Path(path).name}"
    console_print(msg, "info-cyan")
    logger.info(msg)

    import traceback
    while attempts < max_retries:
        try:
            start_time = time.time()
            extract_audio(path, audio_path)
            if audio_only:
                msg = f"🎧 Audio extracted: {Path(audio_path).name}"
                console_print(msg, "success")
                logger.info(msg)
                return
            result, detected_lang = transcribe(audio_path, model_name=model, language=language)
            language = detected_lang
            output_path = build_output_path(path, language, default, forced, sdh, output_format, output_dir)
            write_subtitle(result, output_path, output_format)
            # Local grammar correction post-processing
            if args_dict.get("postprocess_local"):
                try:
                    msg = "✏️  Post-processing subtitles for grammar correction..."
                    console_print(msg, "info-cyan")
                    logger.info(msg)
                    corrector = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
                    with open(output_path, "r", encoding="utf-8") as f:
                        original = f.read()

                    import re
                    def correct_srt(text):
                        blocks = re.split(r"\n\s*\n", text.strip())
                        corrected_blocks = []
                        for block in blocks:
                            lines = block.split("\n")
                            if len(lines) >= 3:
                                text_line = " ".join(lines[2:])
                                corrected = corrector.generate_text(text_line, args={"max_new_tokens": 50}).text
                                corrected_block = "\n".join(lines[:2] + [corrected])
                                corrected_blocks.append(corrected_block)
                            else:
                                corrected_blocks.append(block)
                        return "\n\n".join(corrected_blocks)

                    corrected_text = correct_srt(original)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(corrected_text)

                    msg = "✅ Grammar correction complete"
                    console_print(msg, "success")
                    logger.info(msg)
                except Exception as e:
                    msg = f"⚠️  Failed to post-process subtitles: {e}"
                    console_print(msg, "warning")
                    logger.warning(msg)
            if default and Path(path).suffix.lower() == ".mkv":
                set_mkv_subtitle_default(path, forced=forced, sdh=sdh)
            os.remove(audio_path)
            end_time = time.time()
            duration_sec = end_time - start_time
            video_duration = result.get("duration", None)
            speed_ratio = round(video_duration / duration_sec, 2) if video_duration else "N/A"
            timing_msg = f"⏱️  {Path(output_path).name} took {duration_sec:.2f}s (Speed: {speed_ratio}x)"
            console_print(timing_msg, "info-cyan")
            logger.info(timing_msg)
            msg = f"✅ Done: {Path(output_path).name}"
            console_print(msg, "success")
            logger.info(msg)
            return
        except subprocess.CalledProcessError as e:
            attempts += 1
            msg = f"🔁 Attempt {attempts}/{max_retries} failed (FFmpeg or mkvpropedit error) for {Path(path).name}: {e}"
        except MemoryError as e:
            attempts += 1
            msg = f"💥 Memory error on attempt {attempts}/{max_retries} for {Path(path).name}: {e}"
        except Exception as e:
            attempts += 1
            msg = f"⚠️ Unknown error on attempt {attempts}/{max_retries} for {Path(path).name}: {e}"

        tb = traceback.format_exc()
        full_msg = f"{msg}\n{tb}"
        console_print(full_msg, "warning")
        logger.warning(full_msg)

        if attempts == max_retries:
            msg = f"❌ Failed permanently after {max_retries} attempts: {Path(path).name}"
            console_print(msg, "error")
            logger.error(msg)
        else:
            # Exponential backoff, but capped at max_backoff seconds
            backoff_time = min(2 ** attempts, max_backoff)
            time.sleep(backoff_time)


def main():
    init(autoreset=True)
    parser = argparse.ArgumentParser(
        description="Generate subtitles from video(s) using OpenAI Whisper (configurable via ~/.autosubrc)"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged configuration from CLI and config file, then exit"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file or folder",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="base",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Language code (e.g., en, es, fr)",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        default="srt",
        choices=["srt", "vtt"],
        help="Subtitle output format (srt or vtt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if subtitle already exists",
    )
    parser.add_argument(
        "--default", action="store_true", help="Mark subtitle as default"
    )
    parser.add_argument(
        "--forced",
        action="store_true",
        help="Mark subtitle as forced (non-native dialogue only)",
    )
    parser.add_argument(
        "--sdh",
        action="store_true",
        help="Include SDH (subtitles for the deaf and hard of hearing)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel processes to use (auto-calculated if not set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be processed, do not generate output",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="autosub.log",
        help="Path to log file (default: autosub.log)",
    )
    parser.add_argument(
        "--no-logfile",
        action="store_true",
        help="Disable all logging to file (only console output)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colorized console output"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional path to output subtitles separately from input"
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Extract audio only as .wav and skip transcription"
    )
    parser.add_argument(
        "--audio-output-dir",
        type=str,
        default=None,
        help="Optional path to save extracted audio separately from input"
    )
    parser.add_argument(
        "--quiet-filenames",
        action="store_true",
        help="Suppress per-file progress logging in batch mode"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts per file on failure (default: 3)"
    )
    parser.add_argument(
        "--max-backoff",
        type=int,
        default=30,
        help="Maximum wait time (in seconds) between retries (default: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Number of files to process at once (0 = no limit)"
    )
    parser.add_argument(
        "--batch-delay",
        type=int,
        default=0,
        help="Delay in seconds between batches (default: 0)"
    )
    parser.add_argument(
        "--postprocess-local",
        action="store_true",
        help="Enable local grammar correction of subtitles using a lightweight language model"
    )

    args = parser.parse_args()

    # Optional config file at ~/.autosubrc in TOML format, e.g.:
    config_path = Path.home() / ".autosubrc"
    config_defaults = {}
    if config_path.exists():
        try:
            config_defaults = toml.load(config_path)
            console_print(f"📁 Loaded config defaults from {config_path}", "info-cyan")
        except Exception as e:
            console_print(f"⚠️  Failed to parse config file: {e}", "warning")

    global USE_COLOR
    USE_COLOR = not args.no_color

    if args.no_logfile:
        setup_logging(None)
    else:
        logdir = "."
        if args.logfile == "autosub.log":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.logfile = f"autosub-{timestamp}.log"
        Path(logdir).mkdir(parents=True, exist_ok=True)
        setup_logging(os.path.join(logdir, args.logfile))

        # Create or update symlink to latest log file
        latest_symlink = Path(logdir) / "autosub-latest.log"
        try:
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()
            latest_symlink.symlink_to(Path(args.logfile).resolve())
        except Exception as e:
            console_print(f"⚠️  Could not update symlink: {e}", "warning")

    logger = logging.getLogger()

    # Adjust job count based on available memory
    if args.jobs is None:
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        # Rough estimate: 2GB per process (adjust as needed)
        est_jobs = max(1, min(multiprocessing.cpu_count(), int(total_mem_gb // 2)))
        args.jobs = est_jobs
        msg = (
            f"🧠 Detected ~{total_mem_gb:.1f} GB RAM -> Using {args.jobs} parallel jobs"
        )
        console_print(msg, "info-cyan")
        logger.info(msg)

    # Warn if available memory is low
    avail_mem_gb = psutil.virtual_memory().available / (1024**3)
    if avail_mem_gb < 2 * args.jobs:
        warn_msg = f"⚠️  Low available memory ({avail_mem_gb:.1f} GB) for {args.jobs} jobs. Consider reducing --jobs or using a smaller model."
        console_print(warn_msg, "warning")
        logger.warning(warn_msg)

    input_path = Path(args.input)

    if input_path.is_file():
        simple_args = {
            "language": args.language or config_defaults.get("language"),
            "model": args.model or config_defaults.get("model", "base"),
            "output_format": args.output_format or config_defaults.get("output_format", "srt"),
            "force": args.force or config_defaults.get("force", False),
            "default": args.default or config_defaults.get("default", False),
            "forced": args.forced or config_defaults.get("forced", False),
            "sdh": args.sdh or config_defaults.get("sdh", False),
            "dry_run": args.dry_run or config_defaults.get("dry_run", False),
            "output_dir": args.output_dir or config_defaults.get("output_dir"),
            "audio_only": args.audio_only or config_defaults.get("audio_only", False),
            "audio_output_dir": args.audio_output_dir or config_defaults.get("audio_output_dir"),
            "quiet_filenames": args.quiet_filenames or config_defaults.get("quiet_filenames", False),
            "max_retries": args.max_retries or config_defaults.get("max_retries", 3),
            "max_backoff": args.max_backoff or config_defaults.get("max_backoff", 30),
            "postprocess_local": args.postprocess_local or config_defaults.get("postprocess_local", False),
        }
        if args.print_config:
            print("🔧 Effective Configuration:")
            for k, v in simple_args.items():
                print(f"{k}: {v}")
            return
        process_file(str(input_path), simple_args)
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        if not files:
            msg = f"⚠️  No video files found in {input_path}"
            console_print(msg, "warning")
            logger.warning(msg)
            return

        msg = f"🚀 Processing {len(files)} video(s) with {args.jobs} parallel jobs"
        console_print(msg, "info-cyan")
        logger.info(msg)

        simple_args = {
            "language": args.language or config_defaults.get("language"),
            "model": args.model or config_defaults.get("model", "base"),
            "output_format": args.output_format or config_defaults.get("output_format", "srt"),
            "force": args.force or config_defaults.get("force", False),
            "default": args.default or config_defaults.get("default", False),
            "forced": args.forced or config_defaults.get("forced", False),
            "sdh": args.sdh or config_defaults.get("sdh", False),
            "dry_run": args.dry_run or config_defaults.get("dry_run", False),
            "output_dir": args.output_dir or config_defaults.get("output_dir"),
            "audio_only": args.audio_only or config_defaults.get("audio_only", False),
            "audio_output_dir": args.audio_output_dir or config_defaults.get("audio_output_dir"),
            "quiet_filenames": args.quiet_filenames or config_defaults.get("quiet_filenames", False),
            "max_retries": args.max_retries or config_defaults.get("max_retries", 3),
            "max_backoff": args.max_backoff or config_defaults.get("max_backoff", 30),
            "postprocess_local": args.postprocess_local or config_defaults.get("postprocess_local", False),
        }
        if args.print_config:
            print("🔧 Effective Configuration:")
            for k, v in simple_args.items():
                print(f"{k}: {v}")
            return

        # Batch summary stats
        summary_stats = {
            "total": len(files),
            "success": 0,
            "fail": 0,
            "timings": [],
        }

        logger.info("Batch processing started.")
        batches = [files[i:i + args.batch_size] for i in range(0, len(files), args.batch_size)] if args.batch_size > 0 else [files]
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(files), desc="Batch progress", unit="file", position=1, leave=True, dynamic_ncols=True)
            for batch_index, batch in enumerate(batches):
                with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                    futures = [executor.submit(process_file, str(file), simple_args) for file in batch]
                    for i, future in enumerate(futures):
                        file_name = batch[i].name
                        if not args.quiet_filenames:
                            tqdm.write(f"⏳ Processing: {file_name}", file=sys.stdout)
                        try:
                            future.result()
                            summary_stats["success"] += 1
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            msg = f"❌ Error during batch processing: {e}\n{tb}"
                            console_print(msg, "error")
                            logger.error(msg)
                            summary_stats["fail"] += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(f"{progress_bar.n}/{progress_bar.total}")
                if args.batch_delay and batch_index < len(batches) - 1:
                    time.sleep(args.batch_delay)
            progress_bar.close()
        except concurrent.futures.process.BrokenProcessPool as e:
            err_msg = ("❌ Batch processing failed: A process in the pool was terminated abruptly. "
                       "This may be due to running out of memory or a crash in a worker process.\n"
                       "Try reducing --jobs or using a smaller model.")
            console_print(err_msg, "error")
            logger.error(err_msg)
        # Print and log batch summary
        total = summary_stats["total"]
        success = summary_stats["success"]
        fail = summary_stats["fail"]
        summary_msg = f"\n📊 Summary: {success}/{total} succeeded, {fail} failed"
        console_print(summary_msg, "info-cyan")
        logger.info(summary_msg)
        logger.info("Batch processing finished.")
    else:
        msg = "❌ Input path is invalid."
        console_print(msg, "error")
        logger.error(msg)


if __name__ == "__main__":
    main()
