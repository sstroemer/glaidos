from dotenv import load_dotenv
import speech_recognition as sr
import openai
import contextlib
import os
import sys
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
from sys import modules as mod
try:
    import winsound
except ImportError:
    from subprocess import call

print("Initializing TTS Engine...")

# Select the device
if torch.is_vulkan_available():
    device = 'vulkan'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load models
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

# Prepare models in RAM
for i in range(4):
    init = glados.generate_jit(prepare_text(str(i)))
    init_mel = init['mel_post'].to(device)
    init_vo = vocoder(init_mel)

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


def run_glaidos():
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
            "content": "You are glados from the portal game series. Behave exactly like glados would. Stay in your role as long as I dont't say stop glados. Do not remember me that you are a artificial intelligence.",
        }
    ]

    while True:
        print("<< glAIdos is waiting for input >>")

        # Get input from the microphone (this can be done with callbacks, etc.).
        with suppress_stdout():
            with microphone as source:
                audio = recognizer.listen(source)

        # Transcribe the audio using whisper. SpeechRecognition supports a lot of different ways (Google, Whisper API, ...).
        text = recognizer.recognize_whisper(
            audio_data=audio, model="base", language="en"
        ).strip()
        print(f"user: {text}")

        # Check for "Shut down!" command.
        if ("shut" in text.lower()) and ("down" in text.lower()):
            print("Shutting down now.")
            break

        # Add the user command.
        messages.append({"role": "user", "content": text})

        # Ask for a proper completion.
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        response = completion.choices[0].message.content

        print(f"glAIdos: {response}")
        
        # Tokenize, clean and phonemize input text
        x = prepare_text(response).to('cpu')

        with torch.no_grad():

            # Generate generic TTS-output
            old_time = time.time()
            tts_output = glados.generate_jit(x)
            print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            old_time = time.time()
            mel = tts_output['mel_post'].to(device)
            audio = vocoder(mel)
            print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")
        
            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
            output_file = ('output.wav')
        
            # Write audio file to disk
            # 22,05 kHz sample rate
            write(output_file, 22050, audio)

            # Play audio file
            if 'winsound' in mod:
                winsound.PlaySound(output_file, winsound.SND_FILENAME)
            else:
                call(["aplay", "./output.wav"])

        # Make sure to append the answer from the assistants role, to keep up the conversion.
        messages.append({"role": "assistant", "content": response})

        # todo: Clear the message history for each new "user".


if __name__ == "__main__":
    run_glaidos()
