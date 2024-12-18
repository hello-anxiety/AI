import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# สร้างข้อมูล
X1, Y1 = make_blobs(n_samples=100, n_features=2, centers=[[2.0, 2.0]], cluster_std=0.75, random_state=69)
X2, Y2 = make_blobs(n_samples=100, n_features=2, centers=[[3.0, 3.0]], cluster_std=0.75, random_state=69)

# รวมข้อมูลและกำหนด label
X = np.vstack((X1, X2))
Y = np.hstack((np.zeros(100), np.ones(100)))  # Class A = 0, Class B = 1

# ใช้ Pandas เพื่อจัดการข้อมูล
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['Label'] = Y

# แบ่งข้อมูล Train และ Test
X_train, X_test, Y_train, Y_test = train_test_split(df[['Feature 1', 'Feature 2']], df['Label'], test_size=0.3, random_state=42)

# มาตรฐานข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล Neural Network
model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),  # ชั้นแรก
    Dense(10, activation='relu'),                    # ชั้นซ่อน
    Dense(1, activation='sigmoid')                  # ชั้น Output
])

# Compile โมเดล
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train โมเดล
model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=2)

# ประเมินผลลัพธ์
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# วาด Decision Boundary
def plot_decision_boundary(X, Y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = scaler.transform(grid)  # Apply the same scaling
    probs = model.predict(grid).reshape(xx.shape)
    
    # วาดเส้นตัด (decision boundary)
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.contour(xx, yy, probs, levels=[0.5], linewidths=2, colors='black')  # เส้นตัดที่แสดงค่าความน่าจะเป็น 0.5

plt.figure(figsize=(8, 6))
plt.suptitle("Neural Network Decision Boundary")
plot_decision_boundary(X, Y, model)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolor='k', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
