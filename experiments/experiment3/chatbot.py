import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

f = open('encyclopedia.txt', 'r', encoding='ISO-8859-1')

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def cleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word.lower())
                     for word in sentenceWords]
    return sentenceWords

def bagOfWords(sentence):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0] * len(words)
    for word in sentenceWords:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)

def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return returnList


def getResponse(intentsList, intentsFile):
    tag = intentsList[0]['intent']
    with open(intentsFile, 'r') as f:
        intentsJson = json.load(f)
    for i in intentsJson['intents']:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running!")

while True:
    message = input("")
    if message == "quit":
        break
    ints = predictClass(message)
    res = getResponse(ints, intents)
    print(res)
