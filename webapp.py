from io import StringIO
import streamlit as st
import tkinter
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import nltk
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
nltk.download('stopwords')

print("test")
st.title("test")
#input corpus
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     #st.write(bytes_data)

     # To convert to a string based IO:
     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
     #st.write(stringio)

     # To read file as string:
     string_data = stringio.read()
     st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     #st.write(dataframe)
else:
    string_data="test data"
st.markdown("""---""")
#train model
rawt=string_data
def tokenize_words(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)


processed_inputs = tokenize_words(rawt)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

seq_length = 100
x_data = []
y_data = []

for i in range(0, input_len - seq_length, 1):
    in_seq = processed_inputs[i:i + seq_length]

    out_seq = processed_inputs[i + seq_length]

    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])



n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

X = np.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


filename ="model_weights_saved.hdf5"
model.load_weights(filename)

filepath ="model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam')

#output new txt on button press mebe idk
print("test")
if st.button("generate"):
    num_to_char = dict((i, c) for i, c in enumerate(chars))
    res=""
    start = np.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]
    print("Random Seed:")
    st.write("Random Seed:")
    print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
    st.write("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

    for i in range(50):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_len)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = num_to_char[index]
        seq_in = [num_to_char[value] for value in pattern]

        sys.stdout.write(result)
        #st.markdown("""---""")
        #st.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        res+=result
    st.markdown("""---""")
    data=res
    st.write(data)
    st.markdown("""---""")

if(st.button("train")):
    st.write("training")
    model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)
    st.write("training done")
    st.write(model.loss)
else:
    st.write("train by pressing the button")


