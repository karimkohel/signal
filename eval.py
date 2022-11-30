from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from train import getModel, X_test, y_test

model = getModel()
model.load_weights('model.h5')

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))
