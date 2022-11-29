from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.utils import to_categorical
from collectData import actions, noSequences, sequenceLen, dataPath
import numpy as np

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
