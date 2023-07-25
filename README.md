# GLAIDOS

Currently under heavy development and in a pre-alpha state! - Use at your own risk!

## Includes code from the following projects

glados-tts (MIT License) https://github.com/R2D2FISH/glados-tts - awesome TTS speech synthesis to sound like GLaDOS

(no single line of code used yet) glados-voice-assistant (License inside glados_voice_assistant_LICENSE.md file)  https://github.com/nerdaxic/glados-voice-assistant - big voice assistant project

## Some notes

Installing required linux packages (tested on Ubuntu):

```
sudo apt update
sudo apt install portaudio19-dev python3-pyaudio ffmpeg python3-pip espeak
```

Activating a conda environment for glaidos (you can use the python instance of your OS if you want instead):

```
conda create -n glaidos python=3.8
conda activate glaidos
```

Installing required python3 modules:

```
pip install -r requirements.txt
```

## Some help with common installation issues

https://github.com/Uberi/speech_recognition#on-ubuntudebian-i-get-annoying-output-in-the-terminal-saying-things-like-bt_audio_service_open--connection-refused-and-various-others

see: https://stackoverflow.com/questions/69711732/pyaudio-install-error-failed-building-wheel

## .env file

looks like
```yaml
OPENAI_API_KEY="sk-jkolahOJFHupoUHS990kjhojaSjASFJb89871jknWBs810ß1"
```
