# Clip Farming

Découpe automatique de vidéos YouTube en clips courts avec chevauchement, prêts pour l'export FilmCraft.

## Structure du projet

```
clip_farming/
├── main.py            # Script principal
├── requirements.txt   # Dépendances Python
├── downloads/         # Vidéos source téléchargées (yt-dlp)
├── configs/           # Configurations JSON FilmCraft par clip
└── exports/           # Clips rendus (Partie_01.mp4, Partie_02.mp4, …)
```

## Installation

```bash
# 1. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 2. Installer les dépendances Python
pip install -r requirements.txt

# 3. Installer ffmpeg (requis pour le découpage)
sudo apt install ffmpeg        # Debian / Ubuntu
# brew install ffmpeg          # macOS
```

## Utilisation

```bash
# Lancement standard (65 s / 10 s de chevauchement)
python main.py "https://www.youtube.com/watch?v=XXXXXXXXX"

# Paramètres personnalisés
python main.py "https://www.youtube.com/watch?v=XXXXXXXXX" \
    --clip-duration 60 \
    --overlap 5
```

## Fonctionnement

| Étape | Description |
|-------|-------------|
| 1 | `yt-dlp` télécharge la vidéo en MP4 dans `downloads/` |
| 2 | `ffprobe` détecte la durée totale |
| 3 | Calcul des tranches : **65 s** par clip, **10 s** de chevauchement (pas effectif = 55 s) |
| 4 | Génération des fichiers JSON de configuration FilmCraft dans `configs/` |
| 5 | Rendu batch via `filmcraft-cli` (si installé) ou `ffmpeg` en fallback |
| 6 | Rapport final dans `exports/` |

## Options CLI

| Option | Défaut | Description |
|--------|--------|-------------|
| `url` | *(requis)* | URL YouTube de la vidéo à découper |
| `--all` | `False` | Active le combo : vertical + normalisation sonore loudnorm |
| `--vertical` | `False` | Recadrage adaptatif 9:16 (utiliser `--no-splitscreen` pour l'activer seul) |
| `--splitscreen` | *(Vidéo GTA)* | Split screen 9:16 avec pioche aléatoire dans `backgrounds/` |
| `--no-splitscreen` | `False` | Désactive le split screen |
| `--no-part-badge` | `False` | Désactive le sticker rouge « Partie X » au centre |
| `--subtitles` | `False` | Sous-titres incrustés automatiques (`faster-whisper`) |
| `--lang` | `fr` | Langue pour la transcription des sous-titres |
| `--watermark` | `None` | Texte en filigrane centré en bas de la vidéo |
| `--clip-duration` | `65` | Durée de chaque clip en secondes (1min5) |
| `--overlap` | `10` | Chevauchement entre deux clips en secondes |

En mode vertical sans split screen (`--vertical --no-splitscreen`), le visage
principal est détecté périodiquement afin de déplacer le recadrage et de le
garder au centre. Si aucun visage n'est détecté, le recadrage reste centré sur
l'image.
