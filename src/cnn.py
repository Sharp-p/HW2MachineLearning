import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

class CNNModel:
    def __init__(self, input_shape, output_dim, model_name='SoccerRobotClassifier'):
        """
        Constructor
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_dim = output_dim # should be 5 for ML HW2

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

        # evaluation metrics
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, average='macro') # precision of a positive prediction
        recall = recall_score(Y_test, y_pred, average='macro') # precision in classifiing correctly positives
        f1 = f1_score(Y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(Y_test, y_pred)
        roc_curv = roc_curve(Y_test, y_pred)

        self.save_confusion_matrix(Y_test, y_pred)

        return accuracy, precision, recall, f1, roc_auc, roc_curv

    def save_confusion_matrix(self, Y_test, y_pred):
        cm = confusion_matrix(Y_test, y_pred)
        plt.figure(figsize=(10, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        folder_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(folder_path, '..', 'models', self.model_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        plt.savefig(os.path.join(folder_path, 'conf_'+self.model_name+'.pdf'))
        #plt.show()

    def save_checkpoint(self):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(folder_path, '..', 'models', self.model_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        print("Saving model checkpoint...")
        self.model.save(os.path.join(folder_path, 'model_'+self.model_name+'.keras'))

        




