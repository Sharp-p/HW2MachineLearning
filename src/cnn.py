import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers, utils
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

class CNNModel:
    def __init__(self, input_shape, output_dim, model_name='FootballRobotClassifier'):
        """
        Constructor
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_dim = output_dim # should be 5 for ML HW2

        self.model = None
        self.history = None
        self.is_trained = False


    def build_model(self, learning_rate=0.001, kernel_size=(3, 3), pool_size=(3, 3), kernel_depth=32,
                    dropout_val=0.1, normalize=True, dropout=True):
        self.model = models.Sequential()

        if normalize:
            # normalise the RGB images to be in [0,1] range (NN prefer small numbers)
            self.model.add(layers.Rescaling(1./255, input_shape=self.input_shape))
            self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu'))
        else:
            self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu', input_shape=self.input_shape))
        # filter=32 i.e. the number of kernels applied in each convolutional layer
        # input shape, we will experiment with size from 64x64 to 512x512 pixels
        # (from 4x4 to 56x56 with 3x3 pool and kernel size)

        self.model.add(layers.MaxPooling2D(pool_size=pool_size))
        self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=pool_size))
        self.model.add(layers.Conv2D(kernel_depth, kernel_size, activation='relu'))

        self.model.add(layers.Flatten())
        if dropout: self.model.add(layers.Dropout(dropout_val))
        self.model.add(layers.Dense(64, activation='relu'))
        if dropout: self.model.add(layers.Dropout(dropout_val/2))
        self.model.add(layers.Dense(32, activation='relu'))
        if dropout: self.model.add(layers.Dropout(dropout_val/4))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(self.output_dim, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, dataset_path, size, epochs=50, batch_size=32, val_split=0.2, callbacks=None) -> None:
        """
        Trains the model.
        :param dataset_path: Path to dataset of images divided in folders for each class. The images are expected to be in RGB format.
        :param size: Size of the images in the dataset.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size.
        :param val_split: Validation split.
        :param callbacks: List of callback functions.
        :return: None.
        """
        if self.model is None:
            self.build_model()

        if not os.path.exists(dataset_path):
            raise FileNotFoundError('Dataset not found at {}'.format(dataset_path))

        # generating training and validation dataset
        train_ds = utils.image_dataset_from_directory(dataset_path,
                                                      validation_split=val_split,
                                                      subset="training",
                                                      seed=123,
                                                      image_size=size,
                                                      batch_size=batch_size)
        valid_ds = utils.image_dataset_from_directory(dataset_path,
                                                      validation_split=val_split,
                                                      subset="validation",
                                                      seed=123,
                                                      image_size=size,
                                                      batch_size=batch_size)

        print("Training CNN for maximum", epochs, " epochs...")
        self.history = self.model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            batch_size=batch_size,
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
        recall = recall_score(Y_test, y_pred, average='macro') # precision in classifying correctly positives
        f1 = f1_score(Y_test, y_pred, average='macro')

        # self.save_confusion_matrix(Y_test, y_pred)
        cm = confusion_matrix(Y_test, y_pred)

        return accuracy, precision, recall, f1, cm

    def save_confusion_matrix(self, Y_test, y_pred):
        cm = confusion_matrix(Y_test, y_pred)
        plt.figure(figsize=(10, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix {self.model_name}')

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

    def load_checkpoint(self):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(folder_path, '..', 'models', self.model_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        print("Loading model checkpoint...")
        self.model.load(os.path.join(folder_path, 'model_'+self.model_name+'.keras'))



