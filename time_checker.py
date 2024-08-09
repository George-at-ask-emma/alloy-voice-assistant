import base64
import time
import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
import speech_recognition as sr

load_dotenv()

print("Initializing Voice Agent...")

class Assistant:
    def __init__(self, model):
        print("Creating inference chain...")
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt):
        if not prompt:
            return
        print(f"Processing prompt: {prompt}")
        try:
            llm_start = time.time()
            response = self.chain.invoke(
                {"prompt": prompt},
                config={"configurable": {"session_id": "unused"}},
            )
            llm_end = time.time()
            response = response.strip()
            print(f"AI Response: {response}")
            print(f"LLM latency: {llm_end - llm_start:.2f} seconds")
            if response:
                self._tts(response)
        except Exception as e:
            print(f"Error in answer method: {e}")

    def _tts(self, response):
        print("Converting text to speech...")
        try:
            tts_start = time.time()
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            first_chunk = True
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    if first_chunk:
                        tts_first_audio = time.time()
                        print(f"TTS latency to first audio: {tts_first_audio - tts_start:.2f} seconds")
                        first_chunk = False
                    player.write(chunk)
            tts_end = time.time()
            print(f"Total TTS and playback time: {tts_end - tts_start:.2f} seconds")
            print("Finished playing audio response.")
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history to answer the user's questions.
        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.
        Be friendly and helpful. Show some personality. Do not be too formal.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{prompt}"),
        ])
        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

print("Initializing OpenAI model...")
model = ChatOpenAI(model="gpt-4o-mini", verbose=False)

print("Creating Assistant...")
assistant = Assistant(model)

def audio_callback(recognizer, audio):
    print("Processing audio input...")
    try:
        stt_start = time.time()
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        stt_end = time.time()
        print(f"Transcribed: {prompt}")
        print(f"STT latency: {stt_end - stt_start:.2f} seconds")
        assistant.answer(prompt)
    except sr.UnknownValueError:
        print("Error: Unable to recognize speech.")
    except Exception as e:
        print(f"Error in audio callback: {e}")

print("Setting up audio recognition...")
recognizer = sr.Recognizer()
microphone = sr.Microphone()

try:
    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)

    print("Starting background listening...")
    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    print("Voice Agent is ready. Say something!")

    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping Voice Agent...")
    stop_listening(wait_for_stop=False)
    print("Voice Agent stopped.")
except Exception as e:
    print(f"Unexpected error: {e}")