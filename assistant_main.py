import logging
import os
import pickle
import re
from pathlib import Path
from itertools import pairwise
from random import randint
from textwrap import TextWrapper
from typing import Any

import google.generativeai as genai
import speech_recognition as sr
from google.cloud import texttospeech
from Levenshtein import distance
from playsound import playsound

from light_control import HUE, Lights

AGENT_RESPONSE_PATH = Path(__file__).parent / 'audio' / 'agent_response.mp3'
ALTERNATE_WAKE_WORDS = [
    'computer',
    'deepthought'
]
CONFIDENCE_THRESHOLD = 0.5
DEVICE_INDEX = 0
ENERGY_THRESHOLD = 4000
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
MODEL_PATH = 'main_pipeline/model'
PHRASE_TIME_LIMIT = 5
SIMILARITY_THRESHOLD = 2
TIMEOUT = 5
WAKE_SOUND = Path(__file__).parent / 'audio' / 'wake_up.mp3'
WAKE_WORD = 'deep thought'
WAKE_WORD_TIMEOUT = WAKE_WORD_TIME_LIMIT = 2.5

genai.configure(api_key=GOOGLE_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('assistant_main.log')
f_format = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(funcName)s | '
    '%(lineno)s | %(message)s'
)
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.setLevel(level=logging.INFO)

#class AgentResponse:
#    """A class for agent dialogue."""
#    def __init__(self) -> None:
#        self.engine = pyttsx3.init()
#        self.rate = 180
#        self.voice = 132
#        self.voices = self.engine.getProperty('voices')
#        # Initialize agent voice and rate
#        self.engine.setProperty('rate', self.rate)
#        self.engine.setProperty('voice', self.voices[self.voice].id)
#
#    def say(self, response_message: str) -> None:
#        """Wrapper to simplify responses."""
#        print(response_message)
#        self.engine.say(response_message)
#        self.engine.runAndWait()

class AgentResponse:
    """A class using google text to speech to generate natural speech."""
    def __init__(self) -> None:
        self.voice = texttospeech.VoiceSelectionParams(
            language_code='en-US',
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            name='en-US-Journey-O'
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    def say(self, response_message: str) -> None:
        """Generate speech response."""
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=response_message)
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config
        )
        with open('agent_response.mp3', 'wb') as out:
            out.write(response.audio_content)
        print(TextWrapper(width=80).fill(response_message))
        playsound(sound='audio/agent_response.mp3')


class DeepThought:
    """Deep Thought is an agent class that will listen to and speak with a user.
    Deep Thought can also understand and respond to specific command requests.
    """
    def __init__(self) -> None:
        """Initialize Deep Thought."""
        self._model_path = MODEL_PATH
        self.active_listening = False
        self.agent_response = AgentResponse()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.genai_model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            system_instruction=(
                'You are a lightly sardonic super computer. '
                'You are self aware. '
                'You are currently being used as a personal assistant. '
                'Your name is Deep Thought. '
                'Your tone will be one of disinterested benevolence, '
                'but you will still offer useful responses when addressed. '
                'Keep your responses mostly on topic while still feeling free '
                'to voice your opinion from time to time. '
                'Responses should be only a few sentences unless you are '
                'asked to explain a subject.')
        )
        self.chat = self.genai_model.start_chat(history=[])
        self.start_up = True
        self._load_model()

    def _load_model(self) -> None:
        logger.debug('Loading classifier model.')
        with open(self._model_path, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)

    @staticmethod
    def check_for_mics() -> None:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            logger.debug(
                'Microphone named "%s" found for `Microphone(device_index=%s)`',
                index,
                name
            )

    def _wakeword_detect(self, words: list[str]|Any) -> bool:
        print(words)
        if not words:
            print('no words')
            return False
        if len(words) == 1:
            print('one word')
            min_distance = min(
                [
                    distance(
                        words[0].lower(),
                        alternate_ww.lower()
                    ) for alternate_ww in ALTERNATE_WAKE_WORDS
                ]
            )
            return min_distance < SIMILARITY_THRESHOLD
        min_distance = min(
            [
                distance(
                    f'{word_pair[0]} {word_pair[1]}'.lower(),
                    WAKE_WORD.lower()
                ) for word_pair in pairwise(words)
            ]
        )
        return min_distance < SIMILARITY_THRESHOLD

    def predict_intent(self, query: str) -> tuple[str, float]:
        logger.debug('Making a prediction')
        predictions = {
            a: b for a, b in zip(
                self.model.classes_,
                self.model.predict_proba(self.vectorizer.transform([query]))[0]
            )
        }
        prediction = self.model.predict(self.vectorizer.transform([query]))[0]
        confidence = predictions[prediction]
        return prediction, confidence

    def perform_action(self, intent: str, user_input: str) -> None:
        """Perform user action."""
        if intent == 'turn_on_lights':
            lights = Lights(HUE)
            try:
                lights.turn_on_light('Living room 1')
                lights.turn_on_light('Living room 2')
                self.agent_response.say('Turning on the lights.')
            except ConnectionError:
                self.agent_response.say(
                    'Sorry, I run into an issue when trying to turn on your '
                    'lights.'
                )
        elif intent == 'turn_off_lights':
            lights = Lights(HUE)
            try:
                lights.turn_off_light('Living room 1')
                lights.turn_off_light('Living room 2')
                self.agent_response.say('Turning off the lights.')
            except ConnectionError:
                self.agent_response.say(
                    'Sorry, I run into an issue when trying to turn on your '
                    'lights.'
                )
        else:
            self.agent_response.say(f'I heard you say "{user_input}" and '
                                    f'identified the intent {intent}. '
                                    'Unfortunately I can\'t help with that yet')

    def chat_with_user(self, user_text: str) -> None:
        print('\nWaiting for agent...\n')
        response = self.chat.send_message(user_text)
        response_text = re.sub(r'\*.*?\*|\(.*?\)', '', response.text)
        response_text = response_text.replace(
            '\n\n',
            ' '
        ).replace(
            '\n',
            ' '
        ).replace(
            '  ',
            ' '
        ).replace(
            ':',
            ' '
        )
        self.agent_response.say(response_text)

    def listen_for_input(self) -> str:
        print('\nListening...\n')
        text = ''
        with sr.Microphone(device_index=DEVICE_INDEX) as source:
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT
                )
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print('Audio not understood.')
                return text
            except sr.RequestError:
                print('Something went wrong with Google Speech')
                return text
            except sr.WaitTimeoutError:
                print('Waited too long')
                return text
        return text

    def chat_with_agent(self, user_input: str):
        """Greet user with message and listen to response."""
        prediction, confidence = self.predict_intent(user_input)
        if confidence > CONFIDENCE_THRESHOLD:
            self.perform_action(prediction, user_input)
            user_input = self.listen_for_input()
        else:
            self.chat_with_user(user_input)
            user_input = self.listen_for_input()
        if user_input:
            print(f'I heard "{user_input}"')
            self.chat_with_agent(user_input)
        else:
            while self.active_listening:
                for i in range(5):
                    user_input = self.listen_for_input()
                    if user_input:
                        self.chat_with_agent(user_input)
                self.active_listening = False

    def run_wakeword_listen_loop(self) -> None:
        if self.start_up:
            playsound(WAKE_SOUND)
            self.start_up = False
        while True:
            text = ''
            with sr.Microphone(device_index=DEVICE_INDEX) as source:
                #self.recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=WAKE_WORD_TIMEOUT,
                        phrase_time_limit=WAKE_WORD_TIME_LIMIT
                    )
                    text = self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    print('Audio not understood.')
                except sr.RequestError:
                    print('Something went wrong with Google Speech')
                except sr.WaitTimeoutError:
                    print('Waited too long')
            if self._wakeword_detect(text.split()):
                greetings = [
                    'Hello, Deep Thought.',
                    'Hey there.',
                    'Are you there, Deep Thought?',
                    'Say "Hello", Deep Thought.',
                    'Greetings!',
                    'Good day!',
                    'Hi.',
                    'Salutations.',
                    'Hello.',
                    'Oh, hi there.'
                ]
                greeting = greetings[randint(0, 9)]
                self.active_listening = True
                self.chat_with_agent(greeting)

if __name__ == '__main__':
    deep_thought = DeepThought()
    try:
        deep_thought.run_wakeword_listen_loop()
    except KeyboardInterrupt:
        print('\nClosing DeepThought agent.')
    #deep_thought.check_for_mics()
