
from pathlib import Path
import time

import torch
import whisper

# 1. Chemin du fichier audio / vidéo
audio_path = Path(
    r"C:\chemin\vers\fichier\audio\ou\video.mp4"
)

if not audio_path.is_file():
    raise FileNotFoundError(
        f"Fichier introuvable : {audio_path}"
    )

# 2. Configuration du modèle
MODEL_NAME = "medium"

# Medium utilise ici le CPU pour éviter
# de dépasser les 4 Go de VRAM du GPU
device = "cpu"

print(f"Modèle sélectionné : {MODEL_NAME}")
print(f"Périphérique utilisé : {device}")

# 3. Charger Whisper Medium
print("\nChargement du modèle...")

model = whisper.load_model(
    MODEL_NAME,
    device=device
)

print(f"Modèle chargé sur : {model.device}")

# 4. Lancer la transcription
print("\nTranscription en cours...")

start_time = time.perf_counter()

result = model.transcribe(
    str(audio_path),
    language="fr",
    fp16=False,
    verbose=False
)

elapsed_time = time.perf_counter() - start_time

# 5. Récupérer le texte
transcription = result["text"].strip()

# 6. Afficher la transcription
print("\nTRANSCRIPTION :")
print(transcription)

# 7. Formatage sur plusieurs lignes
segments = result["segments"]

transcription_formatee = "\n".join(
    segment["text"].strip()
    for segment in segments
    if segment["text"].strip()
)

# 8. Enregistrer dans un fichier distinct
# pour conserver les résultats Base et Small
output_path = Path("transcription_medium.txt")

output_path.write_text(
    transcription_formatee,
    encoding="utf-8"
)

# 9. Afficher le bilan
print("\n--- RÉSULTATS ---")
print(f"Fichier : {output_path.resolve()}")
print(f"Modèle : {MODEL_NAME}")
print(f"Périphérique : {device}")
print(f"Temps de transcription : {elapsed_time:.2f} secondes")
