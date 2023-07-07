from dotenv import load_dotenv
import speech_recognition as sr
import openai
import contextlib
import os
import sys

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
            "content": "You are GLaDOS, the artificial intelligence created by Aperture Science",
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

        # Make sure to append the answer from the assistants role, to keep up the conversion.
        messages.append({"role": "assistant", "content": response})

        # todo: Clear the message history for each new "user".


if __name__ == "__main__":
    run_glaidos()
