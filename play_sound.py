# import musicplayer
#
# from gtts import gTTS
# from io import BytesIO
#
# mp3_fp = BytesIO()
# tts = gTTS('hello', 'en')
# tts.write_to_fp(mp3_fp)

from google_speech import Speech

# say "Hello World"
text = "Hi. My name is Assistant. What can I help you with today?"
lang = "en"
speech = Speech(text, lang)
speech.save("response/output.mp3")

# speech2 = Speech("This is a test of the text to speech", "en")
# speech2.save("output2.mp3")

from playsound import playsound
playsound('response/output.mp3')