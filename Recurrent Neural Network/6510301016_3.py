import numpy as np
import matplotlib.pyplot as plt
import keras.api.models as mod
import keras.api.layers as lay
from keras.api.optimizers import Adam

pitch = 20
step = 12
N = 100
n_train = int(N*0.7) # 70 % Train Data

def gen_data(x):
    return (x%pitch) / pitch

t = np.arange(1, N+1)
y = np.sin(0.05*t*10) + 0.8 * np.random.rand(N)
y = np.array(y)

def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)

train, test = y[0:n_train], y[n_train:N]

x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before)", train.shape, test.shape)
print("Dimension (AFTER)", x_train.shape, y_test.shape)

model = mod.Sequential()
model.add(lay.SimpleRNN(units=64,
                        input_shape=(step, 1),
                        activation="relu"))
model.add(lay.Dense(units=1))
optimizer = Adam(0.005)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
hist = model.fit(x_train, y_train, epochs=32, batch_size=16, verbose=1)


y_copy, _ = convertToMatrix(y, step)
predict = model.predict(y_copy)

time_pred = np.arange(step, len(y))        # Prediction take "step" inputs before predict

plt.plot(y, "b", label="Original", alpha=0.6)
plt.plot(time_pred, predict, "r--", label="Predicts")
plt.legend(loc="upper left")
plt.show()