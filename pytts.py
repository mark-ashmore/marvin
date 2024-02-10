import pyttsx3
engine = pyttsx3.init()
rate = engine.getProperty('rate')
print(rate)
engine.setProperty('rate', 180)
voices = engine.getProperty('voices')
#for i in range(141, len(voices)):
#    engine.setProperty('voice', voices[i].id)
#    print(i)
#    engine.say('Okay, turning on the living room lights.')
engine.setProperty('voice', voices[132].id)
engine.say('Oh, it\'s you again. Here I am, brain the size of a planet and you want me to turn lights on and off instead of doing anything useful.')
engine.runAndWait()
