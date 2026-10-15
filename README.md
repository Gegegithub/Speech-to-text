# Speech-to-Text avec Whisper

Transcription locale de fichiers audio et vidéo en texte avec Python et OpenAI Whisper.

## Prérequis

- Python avec `pip`
- FFmpeg, accessible depuis le `PATH`
- GPU NVIDIA compatible CUDA (optionnel, pour accélérer la transcription)

Installer FFmpeg sur Windows :

```powershell
winget install --id Gyan.FFmpeg -e --source winget
```

Vérifier :

```powershell
ffmpeg -version
```

## Installation

Depuis le dossier du projet :

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Option GPU :** installer une version de PyTorch compatible CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/).

## Utilisation

1. Modifier `audio_path` dans `transcribe.py` pour indiquer le fichier audio ou vidéo.
2. Choisir un modèle Whisper (`base`, `small`, `medium`, etc.).
3. Exécuter :

```powershell
.\venv\Scripts\python.exe transcribe.py
```

La transcription est enregistrée dans un fichier `.txt`.

## Notes

- Le traitement est effectué localement, sans clé API OpenAI.
- Une connexion Internet est nécessaire au premier téléchargement de chaque modèle.
- Les modèles volumineux, comme `medium`, peuvent nécessiter davantage de RAM ou de VRAM.
- Whisper transcrit la parole, mais n'extrait pas le texte affiché à l'écran d'une vidéo.
