import base64
import threading
import time
import asyncio

import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
import speech_recognition as sr

import langchain
langchain.verbose = False

load_dotenv()

print("Initializing Voice Agent...")

def custom_exception_handler(loop, context):
    exception = context.get('exception')
    if isinstance(exception, ValueError) and "RootListenersTracer" in str(context.get('message', '')):
        return
    loop.default_exception_handler(context)

class Assistant:
    def __init__(self, model):
        print("Creating inference chain...")
        self.chain = self._create_inference_chain(model)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.set_exception_handler(custom_exception_handler)
        self.loop.run_forever()

    def answer(self, prompt):
        if not prompt:
            return
        future = asyncio.run_coroutine_threadsafe(self._async_answer(prompt), self.loop)
        future.result()

    async def _async_answer(self, prompt):
        print(f"Processing prompt: {prompt}")
        try:
            response = await self.chain.ainvoke(
                {"prompt": prompt},
                config={"configurable": {"session_id": "unused"}},
            )
            response = response.strip()
            print(f"AI Response: {response}")
            if response:
                await self._tts(response)
        except Exception as e:
            print(f"Error in answer method: {e}")

    async def _tts(self, response):
        print("Converting text to speech...")
        try:
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    player.write(chunk)
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
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

print("Creating Assistant...")
assistant = Assistant(model)

def audio_callback(recognizer, audio):
    print("Processing audio input...")
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        print(f"Transcribed: {prompt}")
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