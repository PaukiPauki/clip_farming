#!/usr/bin/env python3
"""
clip_farming — Découpe automatique de vidéos YouTube en clips courts
avec post-traitements optionnels : recadrage 9:16, sous-titres automatiques,
normalisation audio, filigrane texte, vignettes JPEG, split screen.

Usage :
    python main.py <URL>
    python main.py <URL> --all
    python main.py <URL> --splitscreen "URL_FOND"
    python main.py <URL> --all --splitscreen "URL_FOND" --watermark "Ma chaîne"
    python main.py <URL> --clip-duration 60 --overlap 5 --subtitles --lang en
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# ── Répertoires ───────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.resolve()
DOWNLOADS   = ROOT / "downloads"
EXPORTS     = ROOT / "exports"
CONFIGS     = ROOT / "configs"
SRTS        = EXPORTS / "subtitles"
BACKGROUNDS = ROOT / "backgrounds"
FACE_MODEL   = ROOT / "models" / "face_detection_yunet.onnx"

for _d in (DOWNLOADS, EXPORTS, CONFIGS, SRTS, BACKGROUNDS):
    _d.mkdir(parents=True, exist_ok=True)

# Vidéo de fond par défaut (split screen)
DEFAULT_BG_URL = "https://www.youtube.com/watch?v=ZtLrNBdXT7M"
SMART_CROP_INTERVAL = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 1. TÉLÉCHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_url(url: str) -> str:
    """Supprime les backslashes parasites introduits par certains shells (zsh, bash)."""
    return url.replace("\\", "")


def download_video(url: str) -> Path:
    """Télécharge la vidéo YouTube et retourne son chemin local."""
    url = sanitize_url(url)
    print(f"\n📥  Téléchargement : {url}")
    before = set(DOWNLOADS.glob("*.mp4"))

    result = subprocess.run([
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(DOWNLOADS / "%(title)s.%(ext)s"),
        url,
    ], text=True)

    if result.returncode != 0:
        print("❌  Erreur yt-dlp (voir sortie ci-dessus).", file=sys.stderr)
        sys.exit(1)

    after = set(DOWNLOADS.glob("*.mp4"))
    new_files = after - before
    if new_files:
        video_path = max(new_files, key=os.path.getmtime)
    else:
        candidates = sorted(DOWNLOADS.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("❌  Impossible de localiser la vidéo téléchargée.", file=sys.stderr)
            sys.exit(1)
        video_path = candidates[0]

    print(f"✅  Vidéo sauvegardée : {video_path.name}")
    return video_path


# ══════════════════════════════════════════════════════════════════════════════
# 2. INFOS VIDÉO (durée + résolution)
# ══════════════════════════════════════════════════════════════════════════════

def get_video_info(video_path: Path) -> dict:
    """Retourne durée, largeur et hauteur via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌  Erreur ffprobe :\n", result.stderr, file=sys.stderr)
        sys.exit(1)

    data   = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    fmt    = data.get("format", {})
    info   = {
        "duration": float(fmt.get("duration", 0)),
        "width":    int(stream.get("width",  1920)),
        "height":   int(stream.get("height", 1080)),
    }
    print(f"⏱   Durée : {info['duration']:.1f} s ({info['duration']/60:.1f} min) "
          f"— {info['width']}×{info['height']}")
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 3. CALCUL DES TRANCHES
# ══════════════════════════════════════════════════════════════════════════════

def compute_segments(
    total_duration: float,
    clip_duration: int = 65,
    overlap: int = 10,
) -> list[dict]:
    """Calcule les segments {index, start, end} avec chevauchement."""
    step = clip_duration - overlap
    segments, start, idx = [], 0.0, 1
    while start < total_duration:
        end = min(start + clip_duration, total_duration)
        segments.append({"index": idx, "start": round(start, 3), "end": round(end, 3)})
        if end >= total_duration:
            break
        start += step
        idx += 1
    print(f"🔢  {len(segments)} clip(s) — {clip_duration}s / {overlap}s chevauchement (pas={step}s)")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIGS JSON
# ══════════════════════════════════════════════════════════════════════════════

def generate_configs(
    video_path: Path,
    segments: list[dict],
    clip_duration: int,
    overlap: int,
) -> list[Path]:
    config_paths = []
    for seg in segments:
        cfg = {
            "version": "1.0",
            "source": str(video_path),
            "clip": {
                "start":    seg["start"],
                "end":      seg["end"],
                "duration": round(seg["end"] - seg["start"], 3),
            },
            "metadata": {"part": seg["index"], "clip_duration": clip_duration, "overlap": overlap},
            "output":   {"filename": f"Partie_{seg['index']:02d}.mp4",
                         "directory": str(EXPORTS), "format": "mp4"},
        }
        path = CONFIGS / f"segment_{seg['index']:02d}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        config_paths.append(path)
    print(f"📄  {len(config_paths)} config(s) JSON → {CONFIGS}/")
    return config_paths


# ══════════════════════════════════════════════════════════════════════════════
# 5. DÉCOUPAGE BRUT (sans réencodage)
# ══════════════════════════════════════════════════════════════════════════════

def cut_clips(video_path: Path, segments: list[dict]) -> None:
    """Découpe rapide par copie de flux (-c copy)."""
    print("\n✂️   Découpage des clips …")
    for seg in tqdm(segments, desc="Découpage", unit="clip"):
        out = EXPORTS / f"Partie_{seg['index']:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            "-i",  str(video_path),
            "-c",  "copy",
            str(out),
        ], capture_output=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SPLIT SCREEN & GESTION DES CLIPS DE FOND
# ══════════════════════════════════════════════════════════════════════════════

def download_background(url: str) -> Path:
    """Télécharge la vidéo de fond pour le split screen si absente."""
    url = sanitize_url(url)
    before = set(DOWNLOADS.glob("bg_*.mp4"))
    if before:
        # Déjà présente en local
        candidates = sorted(before, key=os.path.getmtime, reverse=True)
        return candidates[0]

    print(f"\n🎮  Téléchargement vidéo de fond source : {url}")
    result = subprocess.run([
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(DOWNLOADS / "bg_%(title)s.%(ext)s"),
        url,
    ], text=True)
    if result.returncode != 0:
        print("❌  Erreur téléchargement vidéo de fond.", file=sys.stderr)
        sys.exit(1)
    after = set(DOWNLOADS.glob("bg_*.mp4"))
    new_files = after - before
    if new_files:
        bg_path = max(new_files, key=os.path.getmtime)
    else:
        candidates = sorted(DOWNLOADS.glob("bg_*.mp4"), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("❌  Impossible de localiser la vidéo de fond.", file=sys.stderr)
            sys.exit(1)
        bg_path = candidates[0]
    print(f"✅  Vidéo de fond source disponible : {bg_path.name}")
    return bg_path


def extract_background_clips(bg_video_path: Path, nb_clips: int = 20, clip_duration: float = 65.0) -> list[Path]:
    """Découpe 20 segments de 1min5 (65s) dans la vidéo de fond."""
    print(f"\n✂️   Préparation du réservoir de {nb_clips} clips de fond (1min5) dans {BACKGROUNDS}/ ...")
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(bg_video_path),
    ], capture_output=True, text=True)

    if r.returncode != 0:
        print(f"❌ Erreur ffprobe sur le fond : {r.stderr}", file=sys.stderr)
        sys.exit(1)

    total_dur = float(json.loads(r.stdout)["format"]["duration"])
    max_start = max(0.0, total_dur - clip_duration)
    step = max_start / (nb_clips - 1) if nb_clips > 1 else 0

    generated = []
    for i in range(nb_clips):
        start = i * step
        out_clip = BACKGROUNDS / f"gta_clip_{i+1:02d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(bg_video_path),
            "-t", str(clip_duration),
            "-c", "copy",
            "-an",
            str(out_clip),
        ]
        subprocess.run(cmd, capture_output=True)
        generated.append(out_clip)

    print(f"✅  {len(generated)} clips de fond créés dans {BACKGROUNDS}/")
    return generated


def get_or_create_background_clips(bg_url: str = DEFAULT_BG_URL) -> list[Path]:
    """Retourne la liste des clips du dossier backgrounds/, en les créant si besoin."""
    existing = sorted(BACKGROUNDS.glob("*.mp4"))
    if existing:
        return existing

    bg_source = download_background(bg_url)
    return extract_background_clips(bg_source, nb_clips=20, clip_duration=65.0)


# ══════════════════════════════════════════════════════════════════════════════
# 7. POST-TRAITEMENTS (filtres vidéo/audio)
# ══════════════════════════════════════════════════════════════════════════════

# ── 6a. Sous-titres automatiques (faster-whisper) ────────────────────────────

def _srt_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _whisper_device() -> tuple[str, str]:
    """Détecte automatiquement si CUDA est disponible. Retourne (device, compute_type)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def transcribe_clip(clip_path: Path, seg: dict, lang: str = "fr") -> Path | None:
    """Transcrit le clip avec faster-whisper et génère un fichier .srt."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  ⚠️  faster-whisper non installé (pip install faster-whisper). Sous-titres ignorés.")
        return None

    srt_path = SRTS / f"Partie_{seg['index']:02d}.srt"
    device, compute_type = _whisper_device()
    model = WhisperModel("small", device=device, compute_type=compute_type)
    segments_iter, _ = model.transcribe(str(clip_path), language=lang)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments_iter, 1):
            f.write(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text.strip()}\n\n")

    return srt_path


# ── 6b. Orchestrateur de post-traitement par clip ────────────────────────────

def get_video_encoder() -> tuple[list[str], str]:
    """Détecte si h264_nvenc (NVIDIA) est disponible pour accélérer l'encodage."""
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_nvenc" in r.stdout:
            return ["-c:v", "h264_nvenc", "-b:v", "4M"], "NVIDIA GPU (NVENC)"
    except Exception:
        pass
    return ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast"], "CPU (libx264 veryfast)"


def build_smart_crop(
    clip_path: Path,
    video_info: dict,
    clip_duration: float,
    command_path: Path,
) -> tuple[int, int, int] | None:
    """Détecte un visage périodiquement et écrit ses positions pour ffmpeg sendcmd."""
    width, height = video_info["width"], video_info["height"]
    crop_w = min(width, int(height * 9 / 16) // 2 * 2)
    crop_h = min(height, int(width * 16 / 9) // 2 * 2)

    if crop_w >= width and crop_h >= height:
        return None

    if not FACE_MODEL.is_file():
        print(f"  ⚠️ Modèle de détection absent ({FACE_MODEL.name}) : recadrage centré.")
        return None

    try:
        import cv2
    except ImportError:
        print("  ⚠️ OpenCV non installé : recadrage centré. Installez les dépendances du projet.")
        return None

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        print(f"  ⚠️ Impossible d'analyser {clip_path.name} : recadrage centré.")
        return None

    try:
        detector = cv2.FaceDetectorYN_create(
            str(FACE_MODEL), "", (320, 320), 0.6, 0.3, 5000
        )
        positions: list[tuple[float, int]] = []
        smoothed_center = width / 2
        last_x = (width - crop_w) // 2

        sample_count = max(1, int(clip_duration / SMART_CROP_INTERVAL) + 1)
        for sample_index in range(sample_count):
            timestamp = min(sample_index * SMART_CROP_INTERVAL, clip_duration)
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                continue

            frame_h, frame_w = frame.shape[:2]
            scale = min(1.0, 640 / max(frame_w, frame_h))
            detection_w = max(1, round(frame_w * scale))
            detection_h = max(1, round(frame_h * scale))
            detection_frame = cv2.resize(frame, (detection_w, detection_h))
            detector.setInputSize((detection_w, detection_h))
            _, faces = detector.detect(detection_frame)
            if faces is not None and len(faces):
                if positions:
                    previous_center = smoothed_center
                    face = min(
                        faces,
                        key=lambda candidate: abs(
                            candidate[0] + candidate[2] / 2 - previous_center
                        ),
                    )
                else:
                    face = max(faces, key=lambda candidate: candidate[2] * candidate[3])

                face_center = (face[0] + face[2] / 2) * width / detection_w
                smoothed_center = 0.65 * smoothed_center + 0.35 * face_center
                last_x = round(max(0, min(width - crop_w, smoothed_center - crop_w / 2)))
            positions.append((timestamp, last_x))

        if not any(x != (width - crop_w) // 2 for _, x in positions):
            print(f"  ℹ️ Aucun visage détecté dans {clip_path.name} : recadrage centré.")
            return None

        command_path.write_text(
            "".join(f"{time:.3f} smart_crop x {x};\n" for time, x in positions),
            encoding="utf-8",
        )
        return crop_w, crop_h, positions[0][1]
    except cv2.error as error:
        print(f"  ⚠️ Détection des visages impossible ({error}) : recadrage centré.")
        return None
    finally:
        capture.release()


def postprocess_clip(
    clip_path: Path,
    seg: dict,
    args: argparse.Namespace,
    video_info: dict,
    bg_clip_path: Path | None = None,
    total_clips: int = 1,
) -> None:
    """Applique toutes les transformations en UNE SEULE passe ffmpeg optimisée."""
    clip_idx = seg["index"]
    clip_dur = seg["end"] - seg["start"]
    enc_opts, enc_name = get_video_encoder()

    af = ["loudnorm"] if args.normalize else []
    filters_top = []

    # Sous-titres
    if args.subtitles:
        srt_path = transcribe_clip(clip_path, seg, args.lang)
        if srt_path:
            srt_escaped = str(srt_path).replace(":", "\\:")
            filters_top.append(
                f"subtitles={srt_escaped}:"
                "force_style='FontSize=18,PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=1,Alignment=2'"
            )

    # Filigrane texte
    if args.watermark:
        text = args.watermark.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        filters_top.append(
            f"drawtext=text='{text}':fontcolor=white:fontsize=28:"
            "box=1:boxcolor=black@0.5:boxborderw=6:"
            "x=(w-text_w)/2:y=h-th-30"
        )

    tmp = clip_path.with_suffix(".proc.tmp.mp4")

    if args.splitscreen and bg_clip_path and bg_clip_path.exists():
        print(f"\n  🎬 Clip {clip_idx:02d}/{total_clips:02d} : Split screen avec {bg_clip_path.name} [{enc_name}] ...")
        
        # Filtres pour la vidéo du haut
        top_filter_chain = ""
        if filters_top:
            top_filter_chain = "," + ",".join(filters_top)

        # Badge "Partie X" au centre (entre les deux vidéos) — style sticker rouge TikTok
        part_badge = ""
        if not getattr(args, "no_part_badge", False):
            font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font_opt = f":fontfile='{font_file}'" if os.path.exists(font_file) else ""
            part_badge = (
                f",drawtext={font_opt}:text='Partie {clip_idx}':"
                f"fontcolor=white:fontsize=54:"
                f"box=1:boxcolor=0xEE3344@1.0:boxborderw=20:"
                f"x=(w-text_w)/2:y=(h-th)/2"
            )
            
        filter_complex = (
            f"[0:v]fps=30,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960{top_filter_chain}[top];"
            f"[1:v]fps=30,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[bot];"
            f"[top][bot]vstack{part_badge}[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-i", str(bg_clip_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a",
        ]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += enc_opts + [
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-t", str(clip_dur),
            str(tmp),
        ]
        r = subprocess.run(cmd, text=True)
        if r.returncode == 0:
            try:
                if clip_path.exists():
                    os.chmod(clip_path, 0o666)
                tmp.replace(clip_path)
                os.chmod(clip_path, 0o666)
            except PermissionError:
                print(f"  ⚠️  {clip_path.name} est ouvert dans Windows ! Fermez votre lecteur vidéo pour mettre à jour ce fichier.")
                tmp.unlink(missing_ok=True)
        else:
            print(f"  ⚠️ Erreur split screen sur {clip_path.name}")
            tmp.unlink(missing_ok=True)

    else:
        # Pas de split-screen : filtres standards
        vf = []
        crop_command_path = clip_path.with_suffix(".crop-commands.txt")
        if args.vertical:
            w, h = video_info["width"], video_info["height"]
            smart_crop = build_smart_crop(clip_path, video_info, clip_dur, crop_command_path)
            if smart_crop:
                target_w, target_h, initial_x = smart_crop
                command_file = str(crop_command_path).replace("\\", "\\\\").replace(":", "\\:")
                vf.extend([
                    f"sendcmd=f='{command_file}'",
                    f"crop@smart_crop={target_w}:{target_h}:{initial_x}:(ih-{target_h})/2",
                ])
            else:
                target_w = int(h * 9 / 16)
                if target_w <= w:
                    vf.append(f"crop={target_w}:{h}:(iw-{target_w})/2:0")
                else:
                    target_h = int(w * 16 / 9)
                    vf.append(f"crop={w}:{target_h}:0:(ih-{target_h})/2")
        vf.extend(filters_top)

        if vf or af:
            print(f"\n  🎬 Clip {clip_idx:02d}/{total_clips:02d} : Encodage [{enc_name}] ...")
            cmd = ["ffmpeg", "-y", "-i", str(clip_path)]
            if vf:
                cmd += ["-vf", ",".join(vf)]
            if af:
                cmd += ["-af", ",".join(af)]
            cmd += enc_opts + [
                "-c:a", "aac", "-b:a", "128k",
                str(tmp),
            ]
            r = subprocess.run(cmd, text=True)
            if r.returncode == 0:
                tmp.replace(clip_path)
            else:
                print(f"  ⚠️ Erreur encodage sur {clip_path.name}")
                tmp.unlink(missing_ok=True)
            crop_command_path.unlink(missing_ok=True)


def run_postprocessing(
    segments: list[dict],
    args: argparse.Namespace,
    video_info: dict,
) -> None:
    """Applique les post-traitements demandés à chaque clip généré."""
    active = [
        args.vertical, args.normalize, args.subtitles,
        bool(args.watermark), bool(args.splitscreen),
    ]
    if not any(active):
        return

    # Récupération ou génération du réservoir de clips de fond
    bg_clips: list[Path] = []
    if args.splitscreen:
        bg_clips = get_or_create_background_clips(args.splitscreen)

    labels = []
    if args.vertical and not args.splitscreen: labels.append("9:16")
    if args.normalize:   labels.append("loudnorm")
    if args.subtitles:   labels.append(f"sous-titres ({args.lang})")
    if args.watermark:   labels.append(f"watermark « {args.watermark} »")
    if args.splitscreen: labels.append(f"split screen ({len(bg_clips)} fonds)")

    print(f"\n🛠   Post-traitement : {' | '.join(labels)}")

    total_clips = len(segments)
    for seg in tqdm(segments, desc="Post-traitement", unit="clip"):
        clip_path = EXPORTS / f"Partie_{seg['index']:02d}.mp4"
        if clip_path.exists():
            chosen_bg = random.choice(bg_clips) if bg_clips else None
            postprocess_clip(clip_path, seg, args, video_info, chosen_bg, total_clips)


# ══════════════════════════════════════════════════════════════════════════════
# 7. RENDU FILMCRAFT-CLI (optionnel)
# ══════════════════════════════════════════════════════════════════════════════

def try_filmcraft(config_paths: list[Path]) -> bool:
    """Tente filmcraft-cli ; retourne True si succès, False si absent."""
    try:
        fc = subprocess.run(["filmcraft-cli", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    if fc.returncode != 0:
        return False

    print("\n🎬  Rendu via filmcraft-cli …")
    for cfg in tqdm(config_paths, desc="Clips", unit="clip"):
        r = subprocess.run(["filmcraft-cli", "render", str(cfg)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ⚠️  {cfg.name} : {r.stderr.strip()}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 8. RAPPORT FINAL
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(segments: list[dict], args: argparse.Namespace) -> None:
    print("\n✅  Clips générés dans ./exports/")
    print("─" * 55)
    for seg in segments:
        out  = EXPORTS / f"Partie_{seg['index']:02d}.mp4"
        size = f"{out.stat().st_size / 1_048_576:.1f} Mo" if out.exists() else "❌ manquant"
        dur  = seg["end"] - seg["start"]
        icons = ""
        if args.subtitles and (SRTS / f"Partie_{seg['index']:02d}.srt").exists():
            icons += " 💬"
        print(f"  Partie_{seg['index']:02d}.mp4  "
              f"{seg['start']:.0f}s→{seg['end']:.0f}s  ({dur:.0f}s)  {size}{icons}")
    print("─" * 55)
    if args.subtitles:
        print(f"  💬  Sous-titres .srt → exports/subtitles/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Découpe une vidéo YouTube en clips courts avec post-traitements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemples :
  python main.py URL
  python main.py URL --all --watermark "Ma chaîne"
  python main.py URL --vertical --clip-duration 58 --overlap 8
""",
    )
    parser.add_argument("url", help="URL YouTube de la vidéo source")
    parser.add_argument("--clip-duration", type=int, default=65, metavar="SEC",
                        help="Durée de chaque clip en secondes (défaut : 65)")
    parser.add_argument("--overlap", type=int, default=10, metavar="SEC",
                        help="Chevauchement entre clips en secondes (défaut : 10)")

    grp = parser.add_argument_group("post-traitements")
    grp.add_argument("--all", action="store_true",
                     help="Active vertical + normalize (subtitles exclus car lent)")
    grp.add_argument("--vertical", action="store_true",
                     help="Recadrer en 9:16 pour Shorts / TikTok / Reels")
    grp.add_argument("--splitscreen", metavar="URL_FOND",
                     default=DEFAULT_BG_URL,
                     help=f"Split screen : clip en haut, extrait aléatoire de URL_FOND en bas (9:16). "
                          f"Défaut : vidéo GTA ({DEFAULT_BG_URL})")
    grp.add_argument("--no-splitscreen", action="store_true",
                     help="Désactiver le split screen")
    grp.add_argument("--subtitles", action="store_true",
                     help="Sous-titres automatiques incrustés (nécessite faster-whisper)")
    grp.add_argument("--lang", default="fr", metavar="CODE",
                     help="Langue de transcription ISO-639-1 (défaut : fr)")
    grp.add_argument("--normalize", action="store_true",
                     help="Normaliser le volume audio (loudnorm EBU R128)")
    grp.add_argument("--watermark", metavar="TEXTE",
                     help="Filigrane texte centré en bas de l'image")
    grp.add_argument("--no-part-badge", action="store_true",
                     help="Désactiver le badge 'Partie X' au milieu de l'écran")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def clean_temp_files() -> None:
    """Nettoie d'éventuels fichiers temporaires résiduels."""
    for pattern in ("*.tmp.mp4", "*.split.*", "*proc.tmp*"):
        for f in EXPORTS.glob(pattern):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> None:
    clean_temp_files()
    args = parse_args()

    if args.overlap >= args.clip_duration:
        print("❌  Le chevauchement doit être inférieur à la durée du clip.", file=sys.stderr)
        sys.exit(1)

    # --all active vertical + normalize (pas subtitles : trop lent sans GPU)
    if args.all:
        args.vertical  = True
        args.normalize = True

    # Split screen GTA activé par défaut, sauf si --no-splitscreen
    if args.no_splitscreen:
        args.splitscreen = None

    video_path   = download_video(args.url)
    video_info   = get_video_info(video_path)
    segments     = compute_segments(video_info["duration"], args.clip_duration, args.overlap)
    config_paths = generate_configs(video_path, segments, args.clip_duration, args.overlap)

    if not try_filmcraft(config_paths):
        print("ℹ️   filmcraft-cli introuvable → utilisation de ffmpeg.")
        cut_clips(video_path, segments)

    run_postprocessing(segments, args, video_info)
    print_summary(segments, args)


if __name__ == "__main__":
    main()
