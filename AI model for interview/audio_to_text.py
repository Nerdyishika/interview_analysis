import os
import speech_recognition as sr
from pydub import AudioSegment

def convert_audio_to_wav(input_audio_path: str, output_wav_path: str):
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_wav_path, format="wav")
    print(f"Audio converted to WAV: {output_wav_path}")

def recognize_speech_from_audio(audio_path: str, output_text_path: str):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        print("Recognizing...")
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            print("Transcription complete.")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Transcription saved to: {output_text_path}")
        except sr.UnknownValueError:
            print("Speech was unintelligible.")
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")

if __name__ == "__main__":
    input_audio = r"C:\Users\HP\OneDrive\Desktop\AI model for interview\AI_model_for_interview.mp3"
    temp_wav = r"temp_audio.wav"
    output_text = r"transcription.txt"

    convert_audio_to_wav(input_audio, temp_wav)
    recognize_speech_from_audio(temp_wav, output_text)

    os.remove(temp_wav)
