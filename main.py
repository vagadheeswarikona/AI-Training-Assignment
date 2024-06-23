import os
import numpy as np
import torch
from transformers import pipeline
import whisper

# Check NumPy availability
try:
    np_version = np.__version__
    print(f"NumPy version: {np_version}")
except Exception as e:
    raise RuntimeError(f"Error with NumPy: {e}")

# Check PyTorch availability
try:
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
except Exception as e:
    raise RuntimeError(f"Error with PyTorch: {e}")

# Check Transformers availability
try:
    nlp_pipeline = pipeline("sentiment-analysis")
    print("Transformers pipeline loaded successfully")
except Exception as e:
    raise RuntimeError(f"Error with Transformers: {e}")

# Load Whisper model with error handling
try:
    whisper_model = whisper.load_model("large-v2")
    print("Whisper model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Error loading Whisper model: {e}")

def process_text_files(text_folder):
    for filename in os.listdir(text_folder):
        file_path = os.path.join(text_folder, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                text = file.read()
                result = nlp_pipeline(text)
                print(f"Text file: {filename}, Analysis: {result}")

def transcribe_audio(hello_audio):
    if hello_audio.endswith(('.wav', '.mp3')):
        try:
            result = whisper_model.transcribe(hello_audio)
            print(f"Audio file: {os.path.basename(hello_audio)}, Transcription: {result['text']}")
        except Exception as e:
            print(f"Error transcribing audio file {hello_audio}: {e}")

def process_video_file(video_file):
    # Placeholder for video processing logic
    # Implement video processing based on specific requirements
    print(f"Processing video file: {video_file}")

def process_hello_audio(hello_audio):
    if hello_audio.endswith(('.wav', '.mp3')):
        try:
            result = whisper_model.transcribe(hello_audio)
            print(f"Hello Audio file: {os.path.basename(hello_audio)}, Transcription: {result['text']}")
        except Exception as e:
            print(f"Error transcribing hello audio file {hello_audio}: {e}")

def process_hello_video(video_file):
    # Placeholder for video processing logic
    # Implement video processing based on specific requirements
    print(f"Processing hello video file: {video_file}")

if __name__ == "__main__":
    base_folder = "AI_training/datasets"
    text_folder = os.path.join(base_folder, "text_file")
    hello_audio = os.path.join(base_folder, "audio")
    hello_video = os.path.join(base_folder, "video")
    hello_audio_file = os.path.join(base_folder, "hello_audio")
    hello_video_file = os.path.join(base_folder, "hello_video")

    if os.path.exists(text_folder):
        process_text_files(text_folder)
    else:
        print(f"Text folder not found: {text_folder}")

    if os.path.exists(hello_audio):
        transcribe_audio(hello_audio)
    else:
        print(f"Audio file not found: {hello_audio}")

    if os.path.exists(hello_video):
        process_video_file(hello_video)
    else:
        print(f"Video file not found: {hello_video}")

    if os.path.exists(hello_audio):
        process_hello_audio(hello_audio)
    else:
        print(f"Hello audio file not found: {hello_audio}")

    if os.path.exists(hello_video):
        process_hello_video(hello_video)
    else:
        print(f"Hello video file not found: {hello_video}")
