from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def makeModel(self):
        model = Sequential() # Type of network

        model.add(Conv2D(32, (3, 3), input_shape=(8, 8, 3), activation='relu')) # Extract features from the board image (convulution layers)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten()) # Changes into 1D vector format for dense

        model.add(Dense(64, activation='relu')) # Makes decision

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear')) # Output

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

