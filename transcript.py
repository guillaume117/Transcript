import os
from pydub import AudioSegment, silence
import whisper
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition

# Load Whisper model for speech-to-text transcription
model = whisper.load_model("large")

# Load Speaker Recognition model
spk_recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_speechbrain"
)

def transcribe_segment(audio_segment: AudioSegment, segment_path: str = "temp_segment.wav") -> list:
    """
    Transcribes an audio segment using Whisper.
    
    :param audio_segment: The audio segment to be transcribed.
    :param segment_path: Temporary file path for segment export.
    :return: List of transcription segments with timestamps.
    """
    audio_segment.export(segment_path, format="wav")
    result = model.transcribe(segment_path)
    return result["segments"]

def extract_speaker_embedding(audio_segment: AudioSegment, segment_path: str = "temp_segment.wav") -> torch.Tensor:
    """
    Extracts a speaker embedding from an audio segment.
    
    :param audio_segment: The audio segment for speaker identification.
    :param segment_path: Temporary file path for segment export.
    :return: Speaker embedding tensor.
    """
    audio_segment.export(segment_path, format="wav")
    signal, sr = torchaudio.load(segment_path)
    embedding = spk_recognizer.encode_batch(signal)
    return embedding

def diarize_audio(file_path: str) -> list:
    """
    Performs basic speaker diarization by detecting pauses and identifying speakers.
    
    :param file_path: Path to the audio file to be processed.
    :return: List of transcribed segments with associated speaker information.
    """
    audio = AudioSegment.from_wav(file_path)

    # Split audio into segments based on silence detection
    segments = silence.split_on_silence(audio, min_silence_len=300, silence_thresh=-50)
    
    speaker_segments = []
    speaker_profiles = {}  # Dictionary to store speaker embeddings
    transcriptions = []
    current_time = 0.0  # Track elapsed time

    for i, segment in enumerate(segments):
        embedding = extract_speaker_embedding(segment)
        speaker = "Speaker_1"
        
        # Compare with existing speaker profiles
        min_score = float("inf")
        for spk, ref_embedding in speaker_profiles.items():
            score = spk_recognizer.similarity(embedding, ref_embedding)
            print("*" * 100)
            print(f"Similarity score with {spk}: {score}")
            if score < min_score:
                min_score = score
                speaker = spk

        # Assign a new speaker if similarity score is too low
        if min_score > 0.65:  # Adjustable threshold
            speaker = f"Speaker_{len(speaker_profiles) + 1}"
            speaker_profiles[speaker] = embedding

        # Define segment start and end times
        segment_start = current_time
        segment_end = current_time + (len(segment) / 1000.0)  # Convert milliseconds to seconds
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

def process_audio(file_path: str, output_txt: str) -> list:
    """
    Full pipeline for transcription and speaker diarization.
    
    :param file_path: Path to the input audio file.
    :param output_txt: Path to save the final transcription output.
    :return: List of processed transcription segments.
    """
    transcriptions = diarize_audio(file_path)
    
    with open(output_txt, "w", encoding="utf-8") as f:
        for entry in transcriptions:
            f.write(f"{entry['speaker']}: {entry['text']}\n")
    
    return transcriptions

def convert_audio_to_wav(input_file: str, output_file: str = "output.wav") -> str:
    """
    Converts an audio file to WAV format.
    
    :param input_file: Path to the input audio file.
    :param output_file: Path to save the converted WAV file.
    :return: Path to the converted WAV file.
    """
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
    return output_file

def process_meeting_summary(input_file: str, output_file: str) -> None:
    """
    Complete pipeline for processing a meeting recording, including transcription and diarization.
    
    :param input_file: Path to the input meeting audio file.
    :param output_file: Path to save the meeting transcription.
    """
    # Convert audio to WAV format
    wav_file = convert_audio_to_wav(input_file)
    
    # Process audio and generate transcription
    process_audio(wav_file, output_file)

# Input file
input_file = "Deuil complexe.m4a"
output_file = "Deuil complexe.txt"

process_meeting_summary(input_file, output_file)
