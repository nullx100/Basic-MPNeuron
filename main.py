import numpy as np

class MPNeuron:
    def __init__(self):
        self.threshold = None
    
    # Funcion de activacion
    def model(self, x):
        # input: [1, 0, 1, 0] [x1, x2, ..., xn]
        return (sum(x) >= self.threshold)

    def predict(self, X):
        # input: [[1, 0, 1, 0], [1, 0, 1, 1]]
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

# Instanciamos la neurona
mp_neuron = MPNeuron()

# Definimos el threshold
mp_neuron.threshold = 3

# Evaluamos diferentes casos de uso
mp_neuron.predict([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]])
