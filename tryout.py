from dotenv import load_dotenv
import simpleaudio as sa
import concurrent.futures
import threading
import speech_recognition as sr
import openai
import contextlib
import os
import sys
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
import re
from collections import defaultdict
from string import *

# Create a lock for thread synchronization
lock = threading.Lock()

def play_wav_files(files):
    files = sorted(files, key=lambda x: int(x[:-4]))  # Sort files based on sentence IDs
    for i, file in enumerate(files):
        wave_obj = sa.WaveObject.from_wave_file(file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        
        if i < len(files) - 1:
            next_file = files[i + 1]
            while not os.path.exists(next_file):
                time.sleep(0.2)


def generate_and_save_audio(sentence, output_file):
    # Tokenize, clean, and phonemize input text
    x = prepare_text(sentence).to('cpu')

    with torch.no_grad():
        # Generate generic TTS-output
        tts_output = glados.generate_jit(x)

        # Use HiFiGAN as vocoder to make output sound like GLaDOS
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)

        # Normalize audio to fit in WAV file
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')

        # Write audio file to disk
        # 22.05 kHz sample rate
        write(output_file, 22050, audio)

def process_sentence(sentence, sentence_id):
    # Generate unique output file name for each sentence
    output_file = f'{sentence_id:03d}.wav'

    # Generate and save audio for the sentence
    generate_and_save_audio(sentence, output_file)

    return output_file

def run_glaidos():
    shutdown_soon = 0
    # Remove the generated audio files
    remove_audio_files()
    print("<< launching glAIdos and testing noise levels >>")
    with suppress_stdout():
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        # Detect noise levels.
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=5)

    # Prepare messages, first: "priming" the system role.
    messages = [
        {
            "role": "system",
            "content": "You are glados from the portal game series. Behave exactly like glados would. Stay in your role no matter what. Do not remember me that you are an artificial intelligence.",
        }
    ]

    while True:
        # Remove the generated audio files
        remove_audio_files()
        if(shutdown_soon == 1):
            break
        print("<< glAIdos is waiting for input >>")

        # Get input from the microphone (this can be done with callbacks, etc.).
        with suppress_stdout():
            with microphone as source:
                audio = recognizer.listen(source)

        # Transcribe the audio using whisper. SpeechRecognition supports a lot of different ways (Google, Whisper API, ...).
        text = recognizer.recognize_whisper(
            audio_data=audio, model="base", language="en"
        ).strip()
        
        if text == "you" or text == "":
            continue
        
        print(f"user: {text}")

        # Check for "Shut down!" command.
        if ("shut" in text.lower()) and ("down" in text.lower()):
            print("Shutting down now after next reply.")
            shutdown_soon = 1

        # Add the user command.
        messages.append({"role": "user", "content": text})

        # Ask for a proper completion.
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        response = completion.choices[0].message.content
        response = response.replace("GLaDOS", "glados")
        # Split the message into sentences using regular expressions
        sentences = re.split(r'(?<=[.?!])\s+|\n', response)

        # Remove any empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        print(sentences)

        # Create a dictionary to store the sentence IDs and corresponding sentences
        sentence_map = defaultdict(str)

        # Generate and save audio for each sentence
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Create a list to hold the future objects
            future_tasks = []

            # Create a set to keep track of the completed sentences
            completed_sentences = set()

            # Iterate over the sentences and submit audio generation tasks
            for sentence_id, sentence in enumerate(sentences):
                # Submit the audio generation task to the executor
                future = executor.submit(process_sentence, sentence, sentence_id)
                future_tasks.append((future, sentence_id))

            # Process the audio generation tasks and play the available files
            while future_tasks:
                # Check if the next task is available
                next_task = future_tasks[0]
                if next_task[0].done():
                    # Get the output file from the completed task
                    output_file = next_task[0].result()

                    # Play the audio file if it has not been played before
                    if next_task[1] not in completed_sentences:
                        play_wav_files([output_file])
                        completed_sentences.add(next_task[1])

                    # Remove the completed task from the future tasks list
                    future_tasks.pop(0)
                else:
                    # Sleep for a short duration if the next task is not available yet
                    time.sleep(0.2)

        # Make sure to append the answer from the assistant's role to keep up the conversation
        messages.append({"role": "assistant", "content": response})

def remove_audio_files():
    files = [f'{i:03d}.wav' for i in range(len(os.listdir()))]
    for file in files:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    load_dotenv()

    @contextlib.contextmanager
    def suppress_stdout():
        """Used to suppress random error messages by PyAudio."""
        with open(os.devnull, "w") as devnull:
            orig_stdout_fno = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), 1)
            orig_stderr_fno = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(orig_stdout_fno, 1)
                os.dup2(orig_stderr_fno, 2)

    # Select the device
    if torch.is_vulkan_available():
        device = 'vulkan'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load models
    glados = torch.jit.load('models/glados.pt')
    vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

    run_glaidos()
