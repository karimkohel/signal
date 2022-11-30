from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.utils import to_categorical
from collectData import actions, noSequences, sequenceLen, dataPath
import numpy as np


def getModel():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

labelMap = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(dataPath, action))).astype(int):
        window = []
        for frame_num in range(sequenceLen):
            res = np.load(os.path.join(dataPath, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

if __name__ == "__main__":

    ### Pre Processing

    ### INITS
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)



    ### Network

    model = getModel()

    model.fit(X_train, y_train, epochs=260, callbacks=[tb_callback])

    model.save("model.h5")

    print(model.summary())