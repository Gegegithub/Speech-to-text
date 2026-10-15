
from pathlib import Path
import whisper

# 1. Chemin vers le fichier audio
audio_path = Path("C:\\chemin\\vers\\le\\fichier\\audio.mp4")

# 2. Charger le modèle Whisper
model = whisper.load_model("base")

# 3. Transcrire le fichier audio
result = model.transcribe(
    str(audio_path),
    language="fr",
    fp16=False
)

# 4. Récupérer le texte
transcription = result["text"]

# 5. Afficher la transcription
print("TRANSCRIPTION :")
print(transcription)

# 6. Enregistrer dans un fichier texte
# Récupérer les segments générés par Whisper
segments = result["segments"]

# Ajouter un retour à la ligne après chaque segment
transcription_formatee = "\n".join(
    segment["text"].strip()
    for segment in segments
)

# Enregistrer la transcription formatée
output_path = Path("transcription.txt")
output_path.write_text(transcription_formatee, encoding="utf-8")

print(f"\nTranscription enregistrée : {output_path}")
