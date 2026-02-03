import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

vocab_size = 10000
max_length = 200

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=vocab_size)

train_data = pad_sequences(train_data, maxlen=max_length)
test_data = pad_sequences(test_data, maxlen=max_length)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(46, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=20,
                    batch_size=512,
                    validation_split=0.2)

results = model.evaluate(test_data, test_labels)

print(f"Test Loss: {results[0]:.3f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")