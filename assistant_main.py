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
# for index, name in enumerate(sr.Microphone.list_microphone_names()):
#     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))


def predict_intent(query):
    with open('PycharmProjects/personal_assistant/model', 'rb') as f:
        vectorizer, model = pickle.load(f)

    prediction = model.predict(vectorizer.transform([query]))
    return prediction


with sr.Microphone() as source:
    print('\nOh, its you... I suppose you need my help with something again')
    resp.say('Oh, its you... I suppose you need my help with something again')
    audio = r3.listen(source, timeout=6)
    get = r2.recognize_google(audio)
    parse = predict_intent(get)
    print('\nOkay, you said', '"' + get + '".')
    resp.say('Okay you said' + get)
    resp.Fulfillment(parse[0])
