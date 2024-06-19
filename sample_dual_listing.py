import threading
import speech_recognition as sr
import time

def listen_for_audio():
    """Listens for audio input using speech recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=2) as source:
        print("Listening for audio...")
        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=None)
            text = recognizer.recognize_google(audio)
            print(f"Audio input: {text}")
            # Signal that audio input has been received
            global audio_received
            audio_received = text
        except sr.UnknownValueError:
            print("Audio not understood.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition; {e}")

def listen_for_text():
    """Listens for text input from the user."""
    while True:
        text = input("Enter text or speak: ")
        if text.lower() == "exit":
            break
        print(f"Text input: {text}")
        # Signal that text input has been received
        global text_received
        text_received = text

def listen_for_audio_and_text():
    """Listens for audio and text simultaneously."""
    audio_thread = threading.Thread(target=listen_for_audio)
    text_thread = threading.Thread(target=listen_for_text)

    audio_thread.start()
    text_thread.start()

    # Wait for either audio or text input to be received
    while not audio_received and not text_received:
        time.sleep(0.1)

    # Stop both threads
    audio_thread.join()
    text_thread.join()


def main():
    """Starts threads for listening to audio and text simultaneously."""
    global audio_received, text_received
    audio_received = ''
    text_received = ''

    listen_for_audio_and_text()

    # Print the received audio and text
    print(f"Received audio: {audio_received}")
    print(f"Received text: {text_received}")

    print("Exiting...")

if __name__ == "__main__":
    main()
