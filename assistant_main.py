import pickle
import speech_recognition as sr
import response as resp

r1 = sr.Recognizer()
r1.energy_threshold = 50
r2 = sr.Recognizer()
r2.energy_threshold = 50
r3 = sr.Recognizer()
r3.energy_threshold = 50

# Keep in case you need to identify a mic index.
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

def predict_intent(query):
    print('Loading model')
    with open('model', 'rb') as f:
        vectorizer, model = pickle.load(f)

    print('Making a prediction')
    prediction = model.predict(vectorizer.transform([query]))
    return prediction


with sr.Microphone(device_index=1) as source:
    print('\nOh, its you... I suppose you need my help with something again')
    resp.say('Oh, its you... I suppose you need my help with something again')
    print('Listening')
    audio = r3.listen(source, timeout=6, phrase_time_limit=6)
    get = r2.recognize_google(audio)
    print(get)
    parse = predict_intent(get)
    print('\nOkay, you said', '"' + get + '".')
    resp.say('Okay you said' + get)
    resp.Fulfillment(parse[0])
