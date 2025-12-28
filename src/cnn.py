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

        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))


