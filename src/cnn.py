

class CNNModel:
    def __init__(self, input_dim, output_dim, model_name='SoccerRobotClassifier'):
        """
        Constructor
        """
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
