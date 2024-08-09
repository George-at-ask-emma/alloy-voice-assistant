import speech_recognition as sr

def listen_and_transcribe():
    recognizer = sr.Recognizer()
    
    print("Listening... Speak into your microphone.")
    
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print("You said:", text)
            except sr.WaitTimeoutError:
                print("No speech detected. Listening again...")
            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except KeyboardInterrupt:
                print("Stopping...")
                break

if __name__ == "__main__":
    listen_and_transcribe()
