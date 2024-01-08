# Aleksandra Imiołek
# Przemysław Imiołek
# Dominik Depta

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Dane MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizacja
train_images, test_images = train_images / 255.0, test_images / 255.0

# Warstwy konwolucyjne
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
model.fit(train_images, train_labels, epochs=5)

# Ocena modelu
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Przewidywanie na wszystkich obrazach z zestawu testowego
predictions = model.predict(test_images)

# Zapis wyników do pliku tekstowego
with open('all_predictions.txt', 'w') as file:
    for i in range(len(test_images)):
        predicted_label = np.argmax(predictions[i])
        actual_label = test_labels[i]
        
        if predicted_label != actual_label:
            file.write(f"Image {i + 1}: Predicted: {predicted_label}, Actual: {actual_label} [MISTAKE]\n")
        else:
            file.write(f"Image {i + 1}: Predicted: {predicted_label}, Actual: {actual_label}\n")

        if i < 5:
            plt.figure(figsize=(2, 2))
            plt.imshow(test_images[i], cmap='gray')
            plt.title(f"Predicted: {predicted_label}, Actual: {actual_label}")
            plt.savefig(f"prediction_image_{i + 1}.png")
            plt.close()

print("Results saved to all_predictions.txt. First 5 predictions saved to prediction_image_*.png files.")