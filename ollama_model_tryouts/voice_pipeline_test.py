import ollama
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Audio Recording Settings
FS = 16000  # 16kHz sample rate standard for speech recognition
DURATION = 10  # Seconds to listen for your command

print("Preparation complete. Get ready to speak your command...")
print(f"Recording for {DURATION} seconds... Speak NOW.")

# Record raw audio array from laptop microphone
audio_data = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
sd.wait()  # Wait until the recording finishes
print("Recording stopped. Processing audio...")

# Save the local file
wav.write("voice_command.wav", FS, audio_data)

# NOTE: In production, you would insert faster-whisper here to turn the .wav into text.
# For this local laptop test, we will simulate the text extraction step:
mock_transcribed_text = "Translate the English alphabet to roman numerals."

print(f"\n[STT Output Transcription]: '{mock_transcribed_text}'")

# Forward the text command to our ultra-fast voice brain
print("\nSending command to Ollama...")
response = ollama.generate(
    model="tinyllama:1.1b",
    prompt=f"You are a local automated system voice assistant. Respond to this user command concisely in one short sentence: {mock_transcribed_text}"
)

print(f"\n[Assistant Voice Response]: {response['response']}")