from dotenv import load_dotenv
import speech_recognition as sr
import openai
import contextlib
import os
import sys

load_dotenv()


@contextlib.contextmanager
def suppress_stdout():
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

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=5)

    messages = [
        {
            "role": "system",
            "content": "You are GLaDOS, the artificial intelligence created by Aperture Science",
        }
    ]

    while True:
        print("<< glAIdos is waiting for input >>")
        with suppress_stdout():
            with microphone as source:
                audio = recognizer.listen(source)

        text = recognizer.recognize_whisper(
            audio_data=audio, model="base", language="en"
        ).strip()
        print(f"user: {text}")

        if ("shut" in text.lower()) and ("down" in text.lower()):
            print("Shutting down now.")
            break

        messages.append({"role": "user", "content": text})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        response = completion.choices[0].message.content

        print(f"glAIdos: {response}")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run_glaidos()
