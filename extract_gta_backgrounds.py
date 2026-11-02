#!/usr/bin/env python3
"""
Extrait 20 vidéos de 1min5 (65 secondes) depuis la vidéo de fond GTA 5
et les enregistre dans le dossier backgrounds/.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DOWNLOADS = ROOT / "downloads"
BACKGROUNDS = ROOT / "backgrounds"
BACKGROUNDS.mkdir(parents=True, exist_ok=True)

# Recherche de la vidéo GTA dans downloads/
gta_candidates = list(DOWNLOADS.glob("bg_*.mp4"))
if not gta_candidates:
    print("❌ Aucune vidéo de fond bg_*.mp4 trouvée dans downloads/.", file=sys.stderr)
    sys.exit(1)

gta_path = gta_candidates[0]
print(f"🎬 Vidéo source : {gta_path.name}")

# Récupération de la durée totale
r = subprocess.run([
    "ffprobe", "-v", "error",
    "-show_entries", "format=duration",
    "-of", "json",
    str(gta_path),
], capture_output=True, text=True)

if r.returncode != 0:
    print(f"❌ Erreur ffprobe: {r.stderr}", file=sys.stderr)
    sys.exit(1)

total_duration = float(json.loads(r.stdout)["format"]["duration"])
clip_duration = 65.0  # 1min5
nb_clips = 20

print(f"⏱  Durée totale : {total_duration:.1f} s")
print(f"✂️  Génération de {nb_clips} clips de {clip_duration:.0f}s (1min5)...")

max_start = max(0.0, total_duration - clip_duration)
step = max_start / (nb_clips - 1) if nb_clips > 1 else 0

for i in range(nb_clips):
    start = i * step
    out_file = BACKGROUNDS / f"gta_clip_{i+1:02d}.mp4"
    print(f"  [{i+1:02d}/{nb_clips}] start={start:.1f}s -> {out_file.name}")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(gta_path),
        "-t", str(clip_duration),
        "-c", "copy",
        "-an",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    ⚠️ Erreur sur {out_file.name}: {res.stderr}")

print(f"\n✅ Terminé ! 20 clips générés dans {BACKGROUNDS}/")

