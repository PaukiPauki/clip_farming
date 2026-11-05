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

## Clips intelligents (100% local)

Le script autonome `smart_clips.py` transcrit toute la vidéo avec
`faster-whisper`, puis demande à Ollama de proposer des passages horodatés dans
des blocs qui se chevauchent. Il supprime les doublons, calcule localement un
score vocal à partir des variations de hauteur, d'intensité et du débit de
parole, puis demande à Ollama un classement global du texte. Le score final
combine par défaut 50% texte, 40% voix et 10% durée (un clip plus court reçoit
un léger bonus). La plage par défaut est de 15 à 35 secondes. FFmpeg rend les clips et les assemble
en compilation verticale 9:16, avec un recadrage qui suit le visage et un fondu
au noir de 0,5 seconde au début et à la fin de chaque clip. Les sous-titres
apparaissent mot par mot; police, taille, couleur, position et marge sont
configurables. Si aucun visage n'est détecté, le recadrage 9:16 reste centré.
Le fondu vidéo dure 0,5 seconde au début et à la fin de chaque clip. L'audio
original est conservé; si `assets/transition.wav` existe, son bruitage est
automatiquement mixé au début de chaque clip, premier inclus, à 35% du volume.
Il n'y a pas de
musique, de transition décorative ni de titre incrusté: titres, descriptions et
justifications restent dans `smart_clips.json`. Le score vocal compare les passages entre eux
dans la même vidéo; c'est un indicateur de dynamisme, pas une détection certaine
d'émotion. Aucune API de LLM hébergée n'est utilisée.

Installer Ollama, démarrer son service puis télécharger un modèle une seule fois :

```bash
ollama pull qwen3:8b
```

Lancer ensuite le script depuis `clip_farming` :

```bash
python smart_clips.py "https://www.youtube.com/watch?v=XXXXXXXXX" --lang fr --count 5
```

Pour ajouter ou remplacer le bruitage, place un fichier WAV nommé
`transition.wav` dans `clip_farming/assets/`. Aucun paramètre supplémentaire
n'est nécessaire dans la commande. Le fichier fourni doit être un effet sonore
auquel tu as le droit d'utilisation. S'il est absent, le script affiche un
avertissement et poursuit la compilation sans bruitage.

Par défaut, la compilation commence par le clip ayant le meilleur score et
continue par score décroissant. Utilisez `--order chronological` pour assembler
les clips dans l'ordre de la vidéo source. `--text-weight 0.6` pondère le
texte à 60%; `--duration-weight 0.1` réserve 10% au bonus de brièveté, et la
voix reçoit le poids restant,
`--min-duration` et `--max-duration` pour régler les durées, et `--block-duration`
avec `--block-overlap` pour régler l'analyse. Les sous-titres acceptent
`--subtitle-font`, `--subtitle-size`, `--subtitle-color` (`#RRGGBB`) et
`--subtitle-position` (`top`, `middle`, `bottom`); utilisez `--no-subtitles` pour
les désactiver. Chaque exécution crée un nouveau
dossier sous `exports/smart_clips/`, contenant les clips, `compilation.mp4` et
`smart_clips.json`. Le téléchargement YouTube nécessite Internet; la transcription,
l'analyse vocale et le LLM sont locaux.
