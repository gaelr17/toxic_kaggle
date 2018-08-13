import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.python.keras.layers import GlobalMaxPool1D
from tensorflow.python.keras.models import Model, load_model

############################################################
########## Parameters
############################################################

NUMBER_OF_UNIQUE_WORDS_CONSIDERED = 25000
LENGTH_OF_SENTENCES = 250

DROPOUT_FACTOR = 0.1

BATCH_SIZE = 32
EPOCHS = 2

############################################################
########## Creating and training the Neural Network
############################################################

train = pd.read_csv('train.csv')
list_sentences_train = train["comment_text"]

tokenizer = Tokenizer(num_words=NUMBER_OF_UNIQUE_WORDS_CONSIDERED)
tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
X_t = pad_sequences(list_tokenized_train, maxlen=LENGTH_OF_SENTENCES)

inp = Input(shape=(LENGTH_OF_SENTENCES, ))
x = Embedding(NUMBER_OF_UNIQUE_WORDS_CONSIDERED, 128)(inp)
x = LSTM(80, return_sequences=True,name='lstm')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(DROPOUT_FACTOR)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(DROPOUT_FACTOR)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(X_t,y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
model.summary()
model.save('trained_nn')

############################################################
########## Using the NN to fill the submission form
############################################################

#model = load_model('trained_nn')
test = pd.read_csv('test.csv')
list_sentences_test = test["comment_text"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=LENGTH_OF_SENTENCES)

ids = np.array(test["id"]).reshape(len(test["id"]), 1)
predictions = model.predict(X_te)
data = np.append(ids, predictions, axis=1)

df = pd.DataFrame(data=data, columns=["id"] + list_classes)
df.to_csv('submit_form.csv', index=False)
