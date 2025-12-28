from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers

class CNNModel:
    def __init__(self, input_shape, output_dim, model_name='SoccerRobotClassifier'):
        """
        Constructor
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = models.Sequential()

        # filter=32 i.e. the number of kernels applied in each convolutional layer
        # input shape, we will experiment with size from 64x64 to 512x512 pixels
        # (from 4x4 to 56x56)
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(self.output_dim, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])