import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from itertools import pairwise
from random import randint
from textwrap import TextWrapper
from typing import Any

import google.generativeai as genai
import spacy
import speech_recognition as sr
from google.api_core.exceptions import InternalServerError
from google.cloud import texttospeech
from google.generativeai.types.generation_types import StopCandidateException
from Levenshtein import distance
from playsound import playsound

from agent_actions.light_control import HUE, Lights

AGENT_RESPONSE_PATH = Path(__file__).parent / 'audio' / 'agent_response.mp3'
ALTERNATE_WAKE_WORDS = [
    'computer',
    'deepthought'
]
CONFIDENCE_THRESHOLD = 0.7
DEVICE_INDEX = 0
ENERGY_THRESHOLD = 4000
ENTITY_MODEL_PATH = Path(__file__).parent / 'main_pipeline' / 'entity_model'
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
MODEL_PATH = Path(__file__).parent / 'main_pipeline' / 'model'
MODEL_TRAINING_PATH = Path(__file__).parent / 'model_training'
PHRASE_TIME_LIMIT = None
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
        audio_file = Path(__file__).parent / 'audio' / 'agent_response.mp3'
        with audio_file.open(mode='wb') as out:
            out.write(response.audio_content)
        print(TextWrapper(width=80).fill(response_message))
        playsound(sound='audio/agent_response.mp3')

    def save_message(self, response_message: str, file_name: str) -> None:
        """Generate speech response and save to audio folder."""
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=response_message)
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config
        )
        audio_file = Path(__file__).parent / 'audio' / f'{file_name}.mp3'
        with audio_file.open(mode='wb') as out:
            out.write(response.audio_content)
        logger.info('Message written to file.')

    def play_message(self, file_name: str) -> None:
        """Play audio file from audio folder."""
        playsound(sound=f'audio/{file_name}.mp3')

class DeepThought:
    """Deep Thought is an agent class that will listen to and speak with a user.
    Deep Thought can also understand and respond to specific command requests.
    """
    def __init__(self) -> None:
        """Initialize Deep Thought."""
        self._model_path = MODEL_PATH
        self._entity_model_path = ENTITY_MODEL_PATH
        self.active_listening = False
        self.agent_response = AgentResponse()
        self.device_index = DEVICE_INDEX
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
        self._load_entity_model()
        self.check_for_mics()

    def _load_model(self) -> None:
        """Load model."""
        logger.debug('Loading classifier model.')
        with self._model_path.open('rb') as f:
            self.vectorizer, self.model = pickle.load(f)

    def _load_entity_model(self) -> None:
        """Load entity phrase matcher model."""
        logger.debug('Loading phrase matcher model and nlp.')
        self.nlp = spacy.load((ENTITY_MODEL_PATH / 'nlp'))
        self.phrase_matcher = pickle.load(
            (self._entity_model_path / 'matcher.pkl').open('rb')
        )

    def check_for_mics(self) -> None:
        """Check for available microphones."""
        mics = list(enumerate(sr.Microphone.list_microphone_names()))
        if not mics:
            sys.exit('No mics detected. Please connect a microphone.')
        mics_string = ''
        for index, name in mics:
            mics_string += f'{index} - {name}\n'
        self.device_index = int(input(
            'Starting up.\n\nPlease let me know which microphone I can use:\n\n'
            f'{mics_string}\n'
        ))

    def _wakeword_detect(self, words: list[str]|Any) -> bool:
        """Detect wake word."""
        if not words:
            return False
        if len(words) == 1:
            logger.debug('One wake word detected. %s', str(words))
            min_distance = min(
                [
                    distance(
                        words[0].lower(),
                        alternate_ww.lower()
                    ) for alternate_ww in ALTERNATE_WAKE_WORDS
                ]
            )
            return min_distance < SIMILARITY_THRESHOLD
        logger.debug('Wake words detected. %s', str(words))
        min_distance = min(
            [
                distance(
                    f'{word_pair[0]} {word_pair[1]}'.lower(),
                    WAKE_WORD.lower()
                ) for word_pair in pairwise(words)
            ]
        )
        return min_distance < SIMILARITY_THRESHOLD

    def predict_intent(self, query: str) -> str:
        """Predict intent.

        If confidence is below threshold, return 'no_match'.

        Returns:
            str: intent
        """
        logger.debug('Making a prediction')
        predictions = dict(
            zip(
                self.model.classes_,
                self.model.predict_proba(self.vectorizer.transform([query]))[0]
            )
        )
        prediction = self.model.predict(self.vectorizer.transform([query]))[0]
        confidence = predictions[prediction]
        if confidence < CONFIDENCE_THRESHOLD:
            prediction = 'no_match'
        return prediction

    def get_entity_values(self, query: str, prediction: str) -> list:
        """Get entity values for the predicted intent.

        Returns:
            list: entity values
        """
        doc = self.nlp(query)
        matches = self.phrase_matcher(doc)
        match_list = []
        for match_id, start, end in matches:
            span = doc[start:end]
            match_list.append(
                (span.text, self.nlp.vocab.strings[match_id], start, end)
            )
        return_matches = []
        if prediction:
            intent_path = MODEL_TRAINING_PATH / f'{prediction}.json'
            with intent_path.open('r', encoding='utf-8') as f:
                intent_dict = json.load(f)
            intent_entities = intent_dict['entities']
            if match_list:
                return_matches = [
                    match for match in match_list
                        if match[1].lower() in intent_entities
                ]
        return return_matches

    def perform_action(
            self,
            intent: str,
            entities: list,
            user_input: str) -> None:
        """Perform user action."""
        print(entities)
        if intent == 'turn_on_lights':
            lights = Lights(HUE)
            try:
                lights.turn_on_light('Living room 1')
                lights.turn_on_light('Living room 2')
                self.agent_response.say('Turning on the lights.')
            except ConnectionError:
                self.agent_response.play_message(
                    file_name='hue_connection_error'
                )
        elif intent == 'turn_off_lights':
            lights = Lights(HUE)
            try:
                lights.turn_off_light('Living room 1')
                lights.turn_off_light('Living room 2')
                self.agent_response.say('Turning off the lights.')
            except ConnectionError:
                self.agent_response.play_message(
                    file_name='hue_connection_error'
                )
        else:
            self.agent_response.say(f'I heard you say "{user_input}" and '
                                    f'identified the intent {intent}. '
                                    'Unfortunately I can\'t help with that yet')

    def chat_with_user(self, user_text: str) -> None:
        """Chat with user."""
        print('\nWaiting for agent...\n')
        try:
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
        except InternalServerError:
            self.agent_response.play_message(
                file_name='gemini_server_error'
            )
            return
        except StopCandidateException:
            self.agent_response.play_message(
                file_name='gemini_stop_candidate_error'
            )
            return
        self.agent_response.say(response_text)

    def listen_for_input(self) -> str:
        """Listen for input."""
        print('\nListening...\n')
        text = ''
        with sr.Microphone(device_index=self.device_index) as source:
            try:
                audio = self.recognizer.listen(
                    source=source,
                    timeout=TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT
                )
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                logger.debug('Audio not understood.')
                return text
            except sr.RequestError:
                logger.debug('Something went wrong with Google Speech')
                return text
            except sr.WaitTimeoutError:
                logger.debug('Waited too long')
                return text
        return text

    def chat_with_agent(self, user_input: str, greeting: bool = False) -> str:
        """Greet user with message and listen to response."""
        if user_input:
            logger.info('Agent heard "%s"', user_input)
            prediction = self.predict_intent(user_input)
            logger.info('Intent: %s', prediction)
            if (prediction != 'no_match') and not greeting:
                logger.info('Taking an action')
                entities = self.get_entity_values(user_input, prediction)
                self.perform_action(prediction, entities, user_input)
            else:
                logger.info('Having a chat with user')
                self.chat_with_user(user_input)
            return 'active_user'
        else:
            return 'no_input'

    def loop_conversation(self) -> str:
        """Loop conversation."""
        user_status = 'user_inactive'
        for _ in range(5):
            user_status = self.chat_with_agent(self.listen_for_input())
            if user_status == 'active_user':
                break
        if user_status == 'no_input':
            user_status = 'user_inactive'
        return user_status

    def run_conversation(self, greeting: bool = False) -> None:
        """Run conversation."""
        if greeting:
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
            greeting_message = greetings[randint(0, 9)]
        user_status = self.chat_with_agent(greeting_message, greeting=greeting)
        while self.active_listening:
            while user_status == 'active_user':
                user_status = self.chat_with_agent(self.listen_for_input())
            user_status = self.loop_conversation()
            if user_status == 'user_inactive':
                self.active_listening = False

    def run_wakeword_listen_loop(self) -> None:
        """Run wakeword listen loop."""
        if self.start_up:
            playsound(WAKE_SOUND)
            self.start_up = False
        while True:
            text = ''
            with sr.Microphone(device_index=self.device_index) as source:
                #self.recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = self.recognizer.listen(
                        source=source,
                        timeout=WAKE_WORD_TIMEOUT,
                        phrase_time_limit=WAKE_WORD_TIME_LIMIT
                    )
                    text = self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    logger.debug('Audio not understood.')
                except sr.RequestError:
                    logger.debug('Something went wrong with Google Speech')
                except sr.WaitTimeoutError:
                    logger.debug('Waited too long')
            if self._wakeword_detect(text.split()):
                self.active_listening = True
                self.run_conversation(greeting=True)

if __name__ == '__main__':
    deep_thought = DeepThought()
    try:
        deep_thought.run_wakeword_listen_loop()
    except KeyboardInterrupt:
        logger.info('Closing DeepThought agent.')
    #deep_thought.check_for_mics()
