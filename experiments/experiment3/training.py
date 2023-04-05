import random
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer


# Load the encyclopedia data from plain text file
with open('encyclopedia.txt', 'r', encoding='ISO-8859-1') as f:
    data = f.read()


# Tokenize the words
words = nltk.word_tokenize(data)

# Create a bag-of-words representation for each sentence
sentences = nltk.sent_tokenize(data)
document_words = []
for sentence in sentences:
    word_list = nltk.word_tokenize(sentence)
    document_words.append((word_list))

# Label each sentence with a corresponding category
categories = ['history', 'geography', 'science', 'art']
document_categories = []
for i, sentence in enumerate(sentences):
    if i < len(sentences) // 4:
        document_categories.append(categories[0])
    elif i < len(sentences) // 2:
        document_categories.append(categories[1])
    elif i < 3 * len(sentences) // 4:
        document_categories.append(categories[2])
    else:
        document_categories.append(categories[3])

# Preprocess the data
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words]
words = sorted(set(words))

document_data = []
output_empty = [0] * len(categories)

for i, word_list in enumerate(document_words):
    bag = []
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    for word in words:
        bag.append(1) if word in word_list else bag.append(0)

    output_row = list(output_empty)
    output_row[categories.index(document_categories[i])] = 1
    document_data.append((bag, output_row))

# Shuffle and split the data into training and testing sets
random.shuffle(document_data)
train_data = document_data[:int(len(document_data)*0.8)]
test_data = document_data[int(len(document_data)*0.8):]

train_X = np.array([x[0] for x in train_data])
train_Y = np.array([y[1] for y in train_data])
test_X = np.array([x[0] for x in test_data])
test_Y = np.array([y[1] for y in test_data])

# Define and train a neural network model on the training data
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    128, input_shape=(len(words),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(categories), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=100, batch_size=5, verbose=1)

# Evaluate the performance of the trained model on the testing data
loss, accuracy = model.evaluate(test_X, test_Y)
print(f"Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}")
model.save('chatbot_model.h5')
print('Done')
