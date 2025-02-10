import os
from pydub import AudioSegment, silence
import whisper
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition

# Configuration du modèle Whisper
model = whisper.load_model("large")

# Charger le modèle de reconnaissance des locuteurs
spk_recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_speechbrain"
)

def transcribe_segment(audio_segment, segment_path="temp_segment.wav"):
    """Transcrit un segment audio en texte avec Whisper"""
    audio_segment.export(segment_path, format="wav")
    result = model.transcribe(segment_path)
    return result["segments"]

def extract_speaker_embedding(audio_segment, segment_path="temp_segment.wav"):
    """Extrait une empreinte vocale (embedding) pour un segment audio"""
    audio_segment.export(segment_path, format="wav")
    signal, sr = torchaudio.load(segment_path)
    embedding = spk_recognizer.encode_batch(signal)
    return embedding

def diarize_audio(file_path):
    """Effectue une diarisation simple en détectant les pauses et en comparant les locuteurs"""
    audio = AudioSegment.from_wav(file_path)

    # Détecter les silences pour segmenter l'audio
    segments = silence.split_on_silence(audio, min_silence_len=300, silence_thresh=-50)
    
    speaker_segments = []
    speaker_profiles = {}  # Dictionnaire des profils vocaux
    transcriptions = []

    current_time = 0.0  # Suivi du temps écoulé

    for i, segment in enumerate(segments):
        embedding = extract_speaker_embedding(segment)
        speaker = "Speaker_1"
        
        # Comparer avec les locuteurs connus
        min_score = float("inf")
        for spk, ref_embedding in speaker_profiles.items():
            score = spk_recognizer.similarity(embedding, ref_embedding)
            print("*"*100)
            print(f"score de similutude par rapport à speaker {spk} = {score}")
            if score < min_score:
                min_score = score
                speaker = spk

        # Si le score est trop faible, c'est un nouveau locuteur
        if min_score > 0.65:  # Seuil arbitraire à affiner
            speaker = f"Speaker_{len(speaker_profiles) + 1}"
            speaker_profiles[speaker] = embedding

        # Ajouter le segment à la liste
        segment_start = current_time
        segment_end = current_time + (len(segment) / 1000.0)  # Convertir en secondes
        current_time = segment_end

        transcript = transcribe_segment(segment)
        transcriptions.append({
            "start": segment_start,
            "end": segment_end,
            "speaker": speaker,
            "text": " ".join([s["text"] for s in transcript])       
        })
        print(transcriptions[-1])
    return transcriptions

def process_audio(file_path, output_txt):
    """Pipeline complète de transcription + diarisation"""
    transcriptions = diarize_audio(file_path)
    
    with open(output_txt, "w", encoding="utf-8") as f:
        for entry in transcriptions:
            f.write(f"{entry['speaker']}: {entry['text']}\n")
    
    return transcriptions

def convert_audio_to_wav(input_file, output_file="output.wav"):
    """Convertit un fichier audio en format WAV"""
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
    return output_file

def process_meeting_summary(input_file, output_file):
    """Pipeline complète de traitement de réunion"""
    # Convertir l'audio en WAV
    wav_file = convert_audio_to_wav(input_file)
    
    # Transcrire l'audio par segments et générer la transcription
    process_audio(wav_file, output_file)

# Fichier d'entrée
input_file = "Deuil complexe.m4a"
output_file = "Deuil complexe.txt"

process_meeting_summary(input_file, output_file)
