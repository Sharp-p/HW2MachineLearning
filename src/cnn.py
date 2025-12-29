from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

class CNNModel:
    def __init__(self, input_shape, output_dim, model_name='SoccerRobotClassifier'):
        """
        Constructor
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_dim = output_dim # should be 4 for ML HW2

        self.model = None
        self.history = None
        self.is_trained = False


    def build_model(self, learning_rate=0.001, kernel_size=(3, 3), pool_size=(3, 3), kernel_depth=32):
        self.model = models.Sequential()
        # filter=32 i.e. the number of kernels applied in each convolutional layer
        # input shape, we will experiment with size from 64x64 to 512x512 pixels
        # (from 4x4 to 56x56 with 3x3 pool and kernel size)
        self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu', input_shape=self.input_shape))
        self.model.add(layers.MaxPooling2D(pool_size=pool_size))
        self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=pool_size))
        self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(self.output_dim, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, Y_train, epochs=50, batch_size=32, val_split=0.2, callbacks=None):
        if self.model is None:
            self.build_model()

        print("Training CNN for maximum", epochs, " epochs...")
        self.history = self.model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            val_split=val_split,
            verbose=0,
            callbacks=callbacks
        )
        self.is_trained = True
        print("Done!")

    def evaluate(self, X_test, Y_test):
        if not self.is_trained:
            print("CNN model not trained!")
            return None

        # prediction
        y_pred = self.model.predict(X_test)



