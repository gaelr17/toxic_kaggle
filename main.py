import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GlobalMaxPool1D, Dense, Dropout
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

train_set = pd.read_csv('train.csv')
sentences_train_set = train_set["comment_text"]

tokenizer = Tokenizer(num_words=NUMBER_OF_UNIQUE_WORDS_CONSIDERED)
tokenizer.fit_on_texts(list(sentences_train_set))
tokenized_train_set = tokenizer.texts_to_sequences(sentences_train_set)

columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_set[columns].values
X_t = pad_sequences(tokenized_train_set, maxlen=LENGTH_OF_SENTENCES)

inputs = Input(shape=(LENGTH_OF_SENTENCES, ))
x = Embedding(NUMBER_OF_UNIQUE_WORDS_CONSIDERED, 128)(inputs)
x = LSTM(80, return_sequences=True,name='lstm')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(DROPOUT_FACTOR)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(DROPOUT_FACTOR)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_t, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.1)
model.summary()
model.save('trained_nn')

############################################################
########## Using the NN to fill the submission form
############################################################

#model = load_model('trained_nn')
test_set = pd.read_csv('test.csv')
sentences_test_set = test_set["comment_text"]
tokenized_test_set = tokenizer.texts_to_sequences(sentences_test_set)
X_te = pad_sequences(tokenized_test_set, maxlen=LENGTH_OF_SENTENCES)

ids = np.array(test_set["id"]).reshape(len(test_set["id"]), 1)
predictions = model.predict(X_te)
data = np.append(ids, predictions, axis=1)

df = pd.DataFrame(data=data, columns=["id"] + columns)
df.to_csv('submit_form.csv', index=False)
