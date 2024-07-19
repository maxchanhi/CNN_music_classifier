import json
import numpy as np
import keras
import json 
import numpy as np
def load_data(dataset_path):
    with open(dataset_path,"r") as f:
        data = json.load(f)
    
    # Convert list to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])    
    
    return inputs,targets

inputs,targets = load_data("data.json")

from sklearn.model_selection import train_test_split

input_train, input_test, target_train, target_test = train_test_split(inputs, targets, test_size=0.2)
print(input_train.shape, target_train.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D

model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(259, 13)),
    MaxPooling1D(3),
    Conv1D(128, 3, activation='relu'),
    Conv1D(128, 3, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(input_train, target_train, 
                    validation_data=(input_test, target_test),
                    epochs=100, 
                    batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

test_loss, test_acc = model.evaluate(input_test, target_test)
print(f"Test accuracy: {test_acc}")
model.save('music_genre_model.h5')
