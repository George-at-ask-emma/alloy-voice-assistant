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
        start_time = time.time()
        print(f"Processing prompt: {prompt}")
        try:
            response = self.chain.invoke(
                {"prompt": prompt},
                config={"configurable": {"session_id": "unused"}},
            )
            response = response.strip()
            print(f"AI Response: {response}")
            print(f"Time to generate response: {time.time() - start_time:.2f} seconds")
            if response:
                self._tts(response)
        except Exception as e:
            print(f"Error in answer method: {e}")

    def _tts(self, response):
        start_time = time.time()
        print("Converting text to speech...")
        try:
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=4096):  # Increased chunk size
                    player.write(chunk)
            print("Finished playing audio response.")
            print(f"Time for text-to-speech: {time.time() - start_time:.2f} seconds")
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
    start_time = time.time()
    print("Processing audio input...")
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        print(f"Transcribed: {prompt}")
        print(f"Time to transcribe: {time.time() - start_time:.2f} seconds")
        assistant.answer(prompt)
    except sr.UnknownValueError:
        print("Error: Unable to recognize speech.")
    except Exception as e:
        print(f"Error in audio callback: {e}")
    finally:
        print(f"Total time for audio processing and response: {time.time() - start_time:.2f} seconds")

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

    # Keep the program running
    while True:
        pass  # This keeps the main thread alive without using sleep

except KeyboardInterrupt:
    print("Stopping Voice Agent...")
    stop_listening(wait_for_stop=False)
    print("Voice Agent stopped.")
except Exception as e:
    print(f"Unexpected error: {e}")