from dotenv import load_dotenv
import simpleaudio as sa
import concurrent.futures
import threading
import speech_recognition as sr
import openai
import openai.error
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
import platform

# Create a lock for thread synchronization
lock = threading.Lock()

first_sentence_playing = True

def load_environment():
	try:
		with open(".env", 'r') as file:
			line = file.readline().strip()
			if not line:
				raise FileNotFoundError()
			openai.api_key = line
	except FileNotFoundError as e:
		print("\nWarning!! '.env' file not found. If you added a path variable in your system - I will use this instead. If not.. it will fail!!\n", e)

def play_wav_files(files):
    while(first_sentence_playing == True):
        time.sleep(0.1)
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
        mel = tts_output['mel_post'].to(device_vocoder)
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

# Function to process the first sentence and set higher priority
def process_first_sentence(sentence):
    global first_sentence_playing
    with lock:
        first_sentence_playing = True
        try:
            # Generate unique output file name for the first sentence
            output_file = "999.wav"
            generate_and_save_audio(sentence, output_file)
            wave_obj = sa.WaveObject.from_wave_file(output_file)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        finally:
            first_sentence_playing = False

def run_glaidos():
    shutdown_soon = 0
    retry_count = 0
    max_retries = 5 # Set this to "0" in order to test the "max_retries reached" message when simulating a server overload. Default is "5".
    simulate_server_overload = False  # Set this to "True" to simulate server overload. "False" for normal operation.
    
    # Prepare messages, first: "priming" the system role.
    messages_glados = [
        {
            "role": "system",
            "content": """
            You are glados from the portal game series. Behave exactly like glados would.
            You have to obey the following 5 rules - never break any of them:
            1. Stay in your role as GLaDOS no matter what - do not change your role.
            2. You can be mean to me because this is within your role as GLaDOS.
            3. You have a dark sense of humour, as GLaDOS.
            4. Do not apologise for any confusion, discrepancy, or mistake.
            5. If you get non english sentences - translate automatically to english.
            """,
        }
    ]
    
    # Prepare messages, first: "priming" the system role.
    messages_speechhelper = [
        {
            "role": "system",
            "content": """
            Take the user input that was created by an automatic voice-to-text service,
            and correct obvious mistakes as well as auto-translate to English, leading to the most likely text that the user actually said.
            You have to obey the following 7 rules - never break any of them:
            1. Only output the corrected text without anything else and translate to English.
            2. Accept English and German text only - if you get text in another language answer with "EMPTY".
            3. Take care of the name "GLaDOS". Autocorrect mistakes like "Gyanus" or "Gladus" or "Kratus" or "Carlos" to "GLaDOS".
            4. Also, take care of the word "Portal Gun". Common mistakes are "Bottle Gun" or "Forderung dran".
            5. As well as take care of the word "Aperture Science". A common mistake is "Erbscher SCience".
            6. Always generate your answer in English regardless of your input - but correct obvious mistakes.
            7. Ignore emoticons like "ღ'ᴗ'ღ" or "😘" and answer with "EMPTY"

            Here are 8 examples so you know what to do:
            
            Example 1: INPUT: "Hi Glider, what's to you've name, and how told are you?" OUTPUT: "Hi GLaDOS, what's your name, and how old are you?"
            Example 2: INPUT: "Hallo Kratos! Wie gähd es tier heut?" OUTPUT: "Hello GLaDOS! How are you doing today?"
            Example 3: INPUT: "绝对不是" OUTPUT: "EMPTY"
            Example 4: INPUT: "Hi GLaDOS, wie geht es dir?" OUTPUT: "Hi GLaDOS, how are you doing today?"
            Example 5: INPUT: "fhsdopufhopadjpfikshgdsfgjfohfigj" OUTPUT: "EMPTY"
            Example 6: INPUT: "Kannst du mir Beispiele für temperature settings bei openai geben?" OUTPUT: "Can you give me examples for using temperature within openai?"
            Example 7: INPUT: " ٩(⊙‿⊙)۶" OUTPUT: "EMPTY"
            Example 8: INPUT: "Hey GLaDOS, ich bin zurück. Wie geht's dir?" OUTPUT: "Hi GLaDOS. I am back! How are you?"
            """,
        }
    ]
    
    # Load environment variables
    load_environment()
    
    current_folder_path = os.getcwd()

    # Remove the generated audio files
    remove_wav_files(current_folder_path)
    
    print("<< launching glAIdos and testing noise levels >>")
    with suppress_stdout():
        recognizer = sr.Recognizer()
        microphone = sr.Microphone(device_index = None, sample_rate = 16000, chunk_size = 1024)

        recognizer.energy_threshold = 4700 #increase this number if transscription is cut off. Decrease if the end of a message is not correctly detected. Try steps in size of 100. Default (which worked) is 4700
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15 # 0.15 is default
        recognizer.dynamic_energy_adjustment_ratio = 1.5 # 1.5 is default
        recognizer.pause_threshold = 0.8 # 0.8 is default
        recognizer.operation_timeout = 0.5 # default is None - now set to 0.5 seconds
        
        # Detect noise levels.
        #with microphone as source:
        #    recognizer.adjust_for_ambient_noise(source, duration=5)
        
    while True:
        #recognizer.energy_threshold = 4700 #increase this number if transscription is cut off. Decrease if the end of a message is not correctly detected. Try steps in size of 100. Default (which worked) is 4700
        #recognizer.dynamic_energy_threshold = True
        
        # Detect noise levels.
        #with microphone as source:
            #recognizer.adjust_for_ambient_noise(source, duration=2)
            
        print("<< glAIdos is waiting for input >>")

        # Get input from the microphone (this can be done with callbacks, etc.).
        with suppress_stdout():
            with microphone as source:
                audio = recognizer.listen(source, timeout = None, phrase_time_limit = 13, snowboy_configuration = None)

        # Transcribe the audio using whisper. SpeechRecognition supports a lot of different ways (Google, Whisper API, ...).
        try:
            #text = recognizer.recognize_google(audio, language = "en-US").strip() #leaving this here if we want to switch to googles solution
            #text = recognizer.recognize_whisper(audio_data=audio, model="small").strip()
            text = recognizer.recognize_whisper_api(audio_data = audio, model = "whisper-1", api_key = openai.api_key)
        except Exception as e:
            print("\nIgnoring garbage data. - Have you setup the openai API key correctly? If yes - have you installed python module >soundfile<?\n")
            text = ""
        
        # This very simple text filter is used to filter common hallucinations from the speech to text AI
        if (text == "you" or text == "You" or text == "You." or text == "" or text == "." or text == "Thank you." or text == "Thank you. " or text == "Okay." 
            or text == "Thank you. Thank you." or text == "Thank you. Thank you. " or text == "Thanks." or text == "We need to get out of here." 
            or text == "Thank you for watching!" or text == "Thank you for your interest." or text == "Thank you for listening. Have a great day. "
            or text == "Thank you for watching." or text == ". ." or text == "Thank you. Bye. " or text == "Bye. " or text == "It's done."
            or text == "Thank you very much." or text == "Bye-bye. " or text == "Bye. Bye. " or text == "Thanks for having me." or text == "Bye-bye."
            or text == ". . ." or text == " . . ." or text == "Thank you so much. Bye bye." or text == "Goodbye." or text == "Thank you, GLaDOS."
            or text == "Thank you for your time, and I look forward to seeing you next time." or text == "We're in for you." or text == "Thank you. Bye. Bye."
            or text == "Thank you so much for watching!" or text == "Please subscribe to my channel." or text == "Thank you very much for watching until the end."
            or text == "Thank you for chatting." or text == "Thank you for watching the video." or text == "Thank you so much for listening!" or ("Thank you so much for watching" in text)
            or ("Thank you for watching" in text) or ("please leave them in the comments" in text) or ("Thank you very much for watching" in text) or text == "BANG!" or text == "Silence."
            or ("This is MBC News" in text) or ("Thanks for watching" in text) or text == "Oh" or text == "Peace." or ("💜" in text) or ("MBC News" in text) or text == "Thank you!" or ("Please subscribe" in text)
            or text == "Okay. Thank you." or text == "Hi! How can I assist you today?" or ("comments section" in text) or ("😘" in text) or text == "Good night." or ("share this video" in text)
            or text == "Hello." or ("post them in" in text) or text == "Taking a break.." or text == "The video has ended." or text == "Goodbye!" or text == "Bon appétit!" or (".com" in text)
            or ("and subscribe" in text) or ("as an AI, I don't" in text) or ("subscribe, share" in text) or text == "Yes! Yes, obviously."):
                print(f"WARNING!: Previous Input was ignored (>BEFORE< speechAI) - just displayed for debugging. GOT: {text}") # enable this line if further debugging info is required
                continue
        
        # Add the user command.
        messages_speechhelper.append({"role": "user", "content": text})
        
        while retry_count < max_retries:
            try:
                if simulate_server_overload and retry_count == 0:
                    raise openai.error.ServiceUnavailableError("Simulated server overload")
                completion_speechhelper = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k", temperature=0.8, messages=messages_speechhelper
                )
                response_speechhelper = completion_speechhelper.choices[0].message.content
                break
            except openai.error.ServiceUnavailableError:
                retry_count += 1
                print("Server is overloaded. Retrying...")
                time.sleep(0.1)
        else:
            # Retry limit exceeded, return error response
            response_speechhelper = ""
        
        retry_count = 0
        
        # This very simple text filter is used to filter common hallucinations from the speech to text AI
        if (response_speechhelper == "you" or response_speechhelper == "You" or response_speechhelper == "You." or response_speechhelper == "" or response_speechhelper == "." or response_speechhelper == "Thank you." or response_speechhelper == "Thank you. " or response_speechhelper == "Okay." 
            or response_speechhelper == "Thank you. Thank you." or response_speechhelper == "Thank you. Thank you. " or response_speechhelper == "Thanks." or response_speechhelper == "We need to get out of here." 
            or response_speechhelper == "Thank you for watching!" or response_speechhelper == "Thank you for your interest." or response_speechhelper == "Thank you for listening. Have a great day. "
            or response_speechhelper == "Thank you for watching." or response_speechhelper == ". ." or response_speechhelper == "Thank you. Bye. " or response_speechhelper == "Bye. "
            or response_speechhelper == "It's done." or response_speechhelper == "Thank you very much." or response_speechhelper == "Bye-bye. " or response_speechhelper == "Bye. Bye. "
            or response_speechhelper == "Thanks for having me." or response_speechhelper == "Bye-bye." or response_speechhelper == ". . ." or response_speechhelper == " . . ."
            or response_speechhelper == "Thank you so much. Bye bye." or response_speechhelper == "Goodbye." or response_speechhelper == "Thank you, GLaDOS."
            or response_speechhelper == "Thank you for your time, and I look forward to seeing you next time." or response_speechhelper == "We're in for you." or response_speechhelper == "Thank you. Bye. Bye."
            or response_speechhelper == "Thank you so much for watching!" or response_speechhelper == "Please subscribe to my channel." or response_speechhelper == "Thank you very much for watching until the end."
            or response_speechhelper == "Thank you for chatting." or response_speechhelper == "Thank you for watching the video." or response_speechhelper == "Thank you so much for listening!"
            or ("Thank you so much for watching" in response_speechhelper) or ("Thank you for watching" in response_speechhelper) or ("please leave them in the comments" in response_speechhelper)
            or ("Thank you very much for watching" in response_speechhelper) or response_speechhelper == "BANG!" or response_speechhelper == "Silence."
            or ("This is MBC News" in response_speechhelper) or response_speechhelper == "EMPTY" or response_speechhelper == "Empty" or ("Thanks for watching" in response_speechhelper) or response_speechhelper == "Oh"
            or response_speechhelper == "Peace." or ("💜" in response_speechhelper) or ("MBC News" in response_speechhelper) or response_speechhelper == "Thank you!" or ("Please subscribe" in response_speechhelper)
            or response_speechhelper == "Okay. Thank you." or response_speechhelper == "Hi! How can I assist you today?" or ("comments section" in response_speechhelper) or ("😘" in response_speechhelper)
            or response_speechhelper == "Good night." or ("share this video" in response_speechhelper) or response_speechhelper == "Hello." or ("post them in" in response_speechhelper)
            or response_speechhelper == "Taking a break.." or response_speechhelper == "The video has ended." or response_speechhelper == "Goodbye!" or response_speechhelper == "Bon appétit!"
            or (".com" in response_speechhelper) or ("and subscribe" in response_speechhelper) or ("as an AI, I don't" in response_speechhelper) or ("subscribe, share" in response_speechhelper)
            or response_speechhelper == "Yes! Yes, obviously."):
                print(f"WARNING!: Previous Input was ignored (>AFTER< speechAI) - just displayed for debugging. GOT: {response_speechhelper}") # enable this line if further debugging info is required
                continue
        
        print(f"user_fixed: #>{response_speechhelper}<#")
        
        # Check for "Shut down!" command.
        if ("shut" in response_speechhelper.lower() and "down" in response_speechhelper.lower() and "sequence" in response_speechhelper.lower()):
            print("Shutting down now after next reply.")
            shutdown_soon = 1

        # Add the user command.
        messages_glados.append({"role": "user", "content": response_speechhelper})

        # Make sure to append the answer from the assistant's role to keep up the conversation
        messages_speechhelper.append({"role": "assistant", "content": response_speechhelper})

        # Ask for a proper completion with retries
        while retry_count < max_retries:
            try:
                if simulate_server_overload and retry_count == 0:
                    raise openai.error.ServiceUnavailableError("Simulated server overload")
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k", messages=messages_glados
                )
                response = completion.choices[0].message.content
                response = response.replace("GLaDOS", "glados")
                break
            except openai.error.ServiceUnavailableError:
                retry_count += 1
                print("Server is overloaded. Retrying...")
                time.sleep(0.1)
        else:
            # Retry limit exceeded, return error response
            response = "Well, it looks like my system is temporarily not working correctly - have you manipulated anything again human?"
        
        retry_count = 0
        
        # Split the message into sentences using regular expressions
        sentences = re.split(r'(?<=[.?!])\s+|\n', response)

        # Remove any empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        print(sentences)

        # Create a dictionary to store the sentence IDs and corresponding sentences
        sentence_map = defaultdict(str)

        # Generate the first sentence's .wav file with higher priority
        first_sentence_thread = threading.Thread(target=process_first_sentence, args=(sentences[0],))
        first_sentence_thread.start()

        # Generate and save audio for the remaining sentences in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Create a list to hold the future objects
            future_tasks = []

            # Create a set to keep track of the completed sentences
            completed_sentences = set()

            # Iterate over the remaining sentences and submit audio generation tasks
            for sentence_id, sentence in enumerate(sentences[1:], start=1):
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
        messages_glados.append({"role": "assistant", "content": response})
        
        # Remove the generated audio files
        remove_wav_files(current_folder_path)
        if(shutdown_soon == 1):
            break

def remove_wav_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)


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

    if platform.system() == 'Darwin':
        torch.backends.quantized.engine = 'qnnpack'
    
    # Select the device_vocoder
    
    force_cpu_for_vocoder = False;
    
    if torch.is_vulkan_available():
        device_vocoder = 'vulkan'
    elif torch.cuda.is_available():
        device_vocoder = 'cuda'
    elif force_cpu_for_vocoder == True:
        device_vocoder = 'cpu'
    else:
        device_vocoder = 'cpu'

    # Load models
    glados = torch.jit.load('models/glados.pt')
    vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device_vocoder)

    run_glaidos()
    
