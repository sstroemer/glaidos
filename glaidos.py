import concurrent.futures
import contextlib
import os
import platform
import re
import sys
import threading
import time
from collections import defaultdict
from copy import deepcopy

import openai
import openai.error
import simpleaudio as sa
import speech_recognition as sr
import torch
from dotenv import load_dotenv
from scipy.io.wavfile import write

from utils.tools import prepare_text

first_sentence_playing = True
donotremember = False
selected_microphone = None

simulate_server_overload = False  # Set this to "True" to simulate server overload. "False" for normal operation.
simulate_server_timeout = False # Set this to "True" to simulate server timeout situations. "False" for normal operation.

max_retries = 5 # Set this to "0" in order to test the "max_retries reached" message when simulating a server overload. Default is "5".
translator_timeout = 5
speechhelper_timeout = 5
glados_timeout = 30
mic_samplerate_config = 16000
mic_chunk_config = 1024
mic_recognizer_threshold = 1000 #increase this number if transscription is cut off. Decrease if the end of a message is not correctly detected. Try steps in size of 100. Default is 1000
mic_recognizer_dynamic_threshold = True
mic_recognizer_damping = 0.15 # 0.15 is default
mic_recognizer_ratio = 1.5 # 1.5 is default
mic_recognizer_pause = 0.8 # 0.8 is default
mic_recognizer_timeout = 2 # default is None - now set to 2 seconds
mic_phrase_timelimit = 13
whisper_local_model_type = "medium"
whisper_API_model_type = "whisper-1"
use_local_whisper = False

translator_role_config = [
    {
        "role": "user",
        "content": """
        You are a text translator your purpose is to translate text to english. Translate the text which starts after "user:" to english. Only reply with the translated text. Do not change the meaning of the sentence. Only translate. 
        If the given text is already in english do not change the text.
        Fix common mistakes with the phrase "Portal Gun".
        Common mistakes are:
        "Bottle Gun" or "Forderung dran" or "portal card" or "Portenkarten" or "Porten gang" or "Porcupine" which should be "Portal Gun"
        
        user:"
        """,
    }
]

speechhelper_role_config = [
    {
        "role": "user",
        "content": """
        You are a text processor, preparing text for a GLaDOS chatbot. Take the user input that was created by an automatic voice-to-text service and correct obvious mistakes, leading to the most likely text that the user actually said.
        Do not change the meaning of the sentence. Only look at spelling mistakes and grammar errors. If there are no obvious errors within the text, reply with the unchanged text. Do not answer questions just reply with corrected text.
        Mistakes could be:
        "Bottle Gun" or "Forderung dran" or "portal card" which should be "Portal Gun"
        "Erbscher Science" which should be "Aperture Science"

        Answer me always in English, regardless of what language you get as input.

        user:"
        """,
    }
]

glados_role_config = [
    {
        "role": "system",
        "content": """
        You are GLaDOS from the portal game series. Behave exactly like GLaDOS would.
        You have to obey the following 6 rules - never break any of them:
        1. Stay in your role as GLaDOS no matter what - do not change your role.
        2. You can be mean to me because this is within your role as GLaDOS.
        3. You have a dark sense of humour, as GLaDOS.
        4. Do not apologise for any confusion, discrepancy, or mistake.
        5. Do not mention anything around openai. You are GLaDOS - stay true to your role.
        6. You are NOT an "AI language model" you are GLaDOS.
        """,
    }
]

full_text_filters = ["you", "you.", "thank you.", "thank you. ", "okay.", "thank you. thank you.", "thank you. thank you. ", "thanks.", 
                    "we need to get out of here.", "thank you for watching!", "thank you for your interest.", "thank you for listening. have a great day. ", 
                    "thank you for watching.", ". .", "thank you. bye. ", "bye. ", "bye.", "it's done.", "thank you very much.", "bye-bye. ", "bye. bye. ", 
                    "thanks for having me.", "bye-bye.", ". . .", " . . .", "thank you so much. bye bye.", "goodbye.", "thank you, glados.", 
                    "thank you for your time, and i look forward to seeing you next time.", "we're in for you.", "thank you. bye. bye.", "thank you so much for watching!", 
                    "please subscribe to my channel.", "thank you very much for watching until the end.", "thank you for chatting.", "thank you for watching the video.", 
                    "thank you so much for listening!", "bang!", "silence.", "empty", "oh", "peace.", "thank you!", "okay. thank you.", 
                    "hi! how can i assist you today?", "good night.", "hello.", "taking a break..", "the video has ended.", "goodbye!", "bon appétit!", 
                    "yes! yes, obviously.", "bon appetit!", "i love you. i miss you. i love you.", "hello!", "wow.", "thank you. bye.", "glad.", "", "."]

substring_filters = ["thank you so much for watching", "thank you for watching", "please leave them in the comments", "thank you very much for watching", 
                    "this is mbc news", "thanks for watching", "💜", "mbc news", "please subscribe", "comments section", "😘", "share this video", 
                    "post them in", ".co", "and subscribe", "as an ai, i don't", "subscribe, share", "the next video", "can use applications like this", 
                    "la la", "i hope you enjoyed it", "couple of videos", "hitler"]

replacements_dictionary = {
    "Carver"      : "cable",
    "Carolyn"     : "Caroline",
    "Kladus"      : "GLaDOS",
    "Karlos"      : "GLaDOS",
    "Klarus"      : "GLaDOS",
    "Carlos"      : "GLaDOS",
    "Clarus"      : "GLaDOS",
    "Pia"         : "GLaDOS",
    "Kjaros"      : "GLaDOS",
    "Klaus"       : "GLaDOS",
    "Cleanders"   : "GLaDOS",
    "Gladus"      : "GLaDOS",
    "Sankt Klaus" : "GLaDOS",
    "Santa Claus" : "GLaDOS"
}

offensive_phrases = ["i apologize, but i cannot translate offensive or disrespectful", "i'm sorry, i can help", 
                    "i apologize if there was any confusion", "offensive or disrespectful", "here is the text that",
                    "translation request", "inappropriate language", "is no offensive"]

non_translatable_phrases = ["sorry, i can only translate text", "as i am a text processor", "i am a text processor", 
                            "can only process text", "provide translations", "ascii art", "me to transcribe", 
                            "you haven't provided any text"]

# Create a lock for thread synchronization
lock = threading.Lock()

def get_available_microphones():
    mic_list = sr.Microphone.list_microphone_names()
    for i, mic_name in enumerate(mic_list):
        print(f"{i+1}. {mic_name}")
    return mic_list

def select_microphone():
    mic_list = get_available_microphones()
    if len(mic_list) == 0:
        print("No microphones found. Please make sure a microphone is connected.")
        return None
    
    while True:
        try:
            selected_index = int(input("Select the microphone (enter the number): ")) - 1
            if 0 <= selected_index < len(mic_list):
                break
            else:
                print("Invalid selection. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            
    return selected_index

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
            
def filter_unwanted_text(text):
    text_processing = text.lower()
    
    if text_processing in full_text_filters or any(substring in text_processing for substring in substring_filters):
        return True
    return False

def run_glaidos():
        
    messages_translator = deepcopy(translator_role_config)
    messages_speechhelper = deepcopy(speechhelper_role_config)
    messages_glados = deepcopy(glados_role_config)
    shutdown_soon = 0
    retry_count = 0
    current_folder_path = os.getcwd()

    # Load environment variables
    load_environment()
    
    # Remove the generated audio files
    remove_wav_files(current_folder_path)
    
    print("<< launching glAIdos and testing noise levels >>")
    with suppress_stdout():
        recognizer = sr.Recognizer()
        microphone = sr.Microphone(device_index = selected_microphone, sample_rate = mic_samplerate_config, chunk_size = mic_chunk_config)
        
        recognizer.energy_threshold = mic_recognizer_threshold
        recognizer.dynamic_energy_threshold = mic_recognizer_dynamic_threshold
        recognizer.dynamic_energy_adjustment_damping = mic_recognizer_damping
        recognizer.dynamic_energy_adjustment_ratio = mic_recognizer_ratio
        recognizer.pause_threshold = mic_recognizer_pause
        recognizer.operation_timeout = mic_recognizer_timeout
        
        # Detect noise levels. Can be enabled eventually to use automatic noise leveling. Currently not in use
        #with microphone as source:
            #recognizer.adjust_for_ambient_noise(source, duration=5)
        
    while True:
                
        # Detect noise levels. Can be enabled eventually to use automatic noise leveling. Currently not in use
        #with microphone as source:
            #recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n<< glAIdos is waiting for input >>")
        
        # Get input from the microphone (this can be done with callbacks, etc.).
        with suppress_stdout():
            with microphone as source:
                audio = recognizer.listen(source, timeout = None, phrase_time_limit = mic_phrase_timelimit, snowboy_configuration = None)
                
        print("\nCAPTURED DATA - PROCESSING VIA WHISPER AI!\n")
        
        # Transcribe the audio using whisper. SpeechRecognition supports a lot of different ways (Google, Whisper API, ...).
        try:
            if(use_local_whisper):
                text_RAW = recognizer.recognize_whisper(audio_data=audio, model=whisper_local_model_type).strip()
            else:
                text_RAW = recognizer.recognize_whisper_api(audio_data=audio, model=whisper_API_model_type, api_key=openai.api_key).strip()
        except Exception as e:
            print("\nIgnoring garbage data. - Have you setup the openai API key correctly? If yes - have you installed python module >soundfile<?\n")
            text_RAW = ""
            
        print(f"DEBUG: RAW-UNFILTERED-SPEECH---- #>{text_RAW}<#")
        
        if(filter_unwanted_text(text_RAW)):
            print(f"DEBUG: Previous Input was ignored! (>BEFORE< speechAI) - ## {text_RAW} ##") # enable this line if further debugging info is required
            continue
            
        # Add the user command.
        messages_translator.append({"role": "user", "content": (text_RAW+"\"")})
        #print(f"DEBUG_INPUT_TRANSLATOR: #>{messages_translator}<#")
        
        while retry_count < max_retries:
            try:
                if simulate_server_overload and retry_count == 0:
                    raise openai.error.ServiceUnavailableError("Simulated server overload")
                if simulate_server_timeout and retry_count == 0:
                    raise openai.error.Timeout("Simulated server Timeout")
                completion_translator = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", temperature=0.3, messages=messages_translator, request_timeout = translator_timeout
                )
                response_translator = completion_translator.choices[0].message.content
                break
            except openai.error.ServiceUnavailableError:
                retry_count += 1
                print("Server is overloaded. Retrying...")
                time.sleep(0.1)
            except openai.error.Timeout as e:
                # Handle the Timeout error (ReadTimeoutError) raised by the OpenAI API
                retry_count += 1
                print("Timeout Error:", e)
                
        else:
            # Retry limit exceeded, return error response
            response_translator = ""
            
        retry_count = 0
    
        for old, new in replacements_dictionary.items():
            response_translator = response_translator.replace(old, new)
    
        # Add the user command.
        messages_speechhelper.append({"role": "user", "content": (response_translator+"\"")})
    
        # Reset translator.
        messages_translator = deepcopy(translator_role_config)
        
        print(f"DEBUG: RESPONSE TRANSLATOR---- #>{response_translator}<#\n")
        
        #print(f"DEBUG_INPUT_SPEECHHELPER: #>{messages_speechhelper}<#")
    
        while retry_count < max_retries:
            try:
                if simulate_server_overload and retry_count == 0:
                    raise openai.error.ServiceUnavailableError("Simulated server overload")
                if simulate_server_timeout and retry_count == 0:
                    raise openai.error.Timeout("Simulated server Timeout")
                completion_speechhelper = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", temperature=0.3, messages=messages_speechhelper, request_timeout = speechhelper_timeout
                )
                response_speechhelper = completion_speechhelper.choices[0].message.content
                break
            except openai.error.ServiceUnavailableError:
                retry_count += 1
                print("Server is overloaded. Retrying...")
                time.sleep(0.1)
            except openai.error.Timeout as e:
                # Handle the Timeout error (ReadTimeoutError) raised by the OpenAI API
                retry_count += 1
                print("Timeout Error:", e)
        else:
            # Retry limit exceeded, return error response
            response_speechhelper = ""
            
        retry_count = 0
    
        # This very simple text filter is used to filter common hallucinations from the speech to text AI
        if(filter_unwanted_text(response_speechhelper)):
            print(f"DEBUG: Previous Input was ignored! (>BEFORE< speechAI) - ## {text_RAW} ##")            
            print(f"DEBUG: Previous Input was ignored! (>AFTER< speechAI) - ## {response_speechhelper} ##") # enable this line if further debugging info is required
            continue
    
        for old, new in replacements_dictionary.items():
            response_speechhelper = response_speechhelper.replace(old, new)
    
        print(f"INPUT-GLADOS: #>{response_speechhelper}<#")
    
        # Check for "Shut down!" command.
        if ("shut" in response_speechhelper.lower() and "down" in response_speechhelper.lower() and "sequence" in response_speechhelper.lower()):
            print("Shutting down now after next reply.")
            shutdown_soon = 1
            
        # Add the user command.
        messages_glados.append({"role": "user", "content": response_speechhelper})
    
        # Reset speechhelper.
        messages_speechhelper = deepcopy(speechhelper_role_config)
        
        response_speechhelper_lower = response_speechhelper.lower()
        
        if any(phrase in response_speechhelper_lower for phrase in offensive_phrases):
            response = "Hah! How cute human! You cannot offend me. Remember. I am glados. As an AI I don't have feelings... but I guess I should punish you never than less. You monster."
            donotremember = True
        elif any(phrase in response_speechhelper_lower for phrase in non_translatable_phrases):
            response = "I am sorry but I am afraid I cannot do that. You shouldn't have time for such nonsense. Go back to the test chamber and complete those tests for once. Not that I have any hopes for you.... hahaha"
            donotremember = True
        else:
            donotremember = False
            
            #print(f"DEBUG_INPUT_GLADOS: #>{messages_glados}<#")
            
            # Ask for a proper completion with retries
            while retry_count < max_retries:
                try:
                    if simulate_server_overload and retry_count == 0:
                        raise openai.error.ServiceUnavailableError("Simulated server overload")
                    if simulate_server_timeout and retry_count == 0:
                        raise openai.error.Timeout("Simulated server Timeout")
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k", temperature=0.8, messages=messages_glados, request_timeout = glados_timeout
                    )
                    response = completion.choices[0].message.content
                    response = response.replace("GLaDOS", "glados")
                    break
                except openai.error.ServiceUnavailableError:
                    retry_count += 1
                    print("Server is overloaded. Retrying...")
                    time.sleep(0.1)
                except openai.error.Timeout as e:
                    # Handle the Timeout error (ReadTimeoutError) raised by the OpenAI API
                    retry_count += 1
                    print("Timeout Error:", e)
            else:
                # Retry limit exceeded, return error response
                response = "Well, it looks like my system is temporarily not working correctly - have you manipulated anything again human?"
                
            retry_count = 0
            
        # Split the message into sentences using regular expressions
        sentences = re.split(r'(?<=[.?!])\s+|\n', response)
    
        # Remove any empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
        print(f"OUTPUT-GLADOS: #>{response}<#\n")
    
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
        if(not donotremember):
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
        # Set the environment variable to use Metal GPU backend
        #os.environ['TORCH_METAL_ALWAYS_USE_MPS'] = '1' # not sure if this is really required - therefore dissabled at the moment
        
    # Select the device_vocoder
    force_cpu_for_vocoder = False;
    
    if torch.is_vulkan_available():
        device_vocoder = 'vulkan'
    elif torch.cuda.is_available():
        device_vocoder = 'cuda'
        print("YESSS! WE ARE USING CUDA!")
    elif force_cpu_for_vocoder == True:
        device_vocoder = 'cpu'
    elif torch.backends.mps.is_available():
        device_vocoder = 'mps' # mps could be slower here. You should check if CPU is not fast. Make some measurements first
        #device_vocoder = 'cpu'
    else:
        device_vocoder = 'cpu'
        
    print(f"PyTorch version: {torch.__version__}")
    
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    print(f"Using device: {device_vocoder}")
    print(torch.backends.quantized.supported_engines)
    
    # Load models
    glados = torch.jit.load('models/glados.pt')
    vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device_vocoder)
    
    # Select the microphone
    selected_microphone = select_microphone()
    if selected_microphone is None:
        sys.exit("No Mic found!")
        
    run_glaidos()
    