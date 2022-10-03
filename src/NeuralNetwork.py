from calendar import c
from sys import float_info
import numpy as np

# synapses matrix : 2 input neurons, 1 output neuron

class NeuralNetwork():
    
    def __init__(self, hiddenNeuronNb, dataset) -> None:
        
        self._synapses = [
            np.random.randn(2, hiddenNeuronNb),
            np.random.randn(hiddenNeuronNb, 1),
        ];
        self._flowerSize = np.array(list( map( lambda flower: flower.size, dataset) ), dtype=float);
        self._flowerSize = self._flowerSize / np.amax(self._flowerSize, axis=0); #divide by max value to get values between 0 and 1
        self._flowerTypes = np.array( list( map( lambda flower: [flower.type], dataset ) ), dtype=float );

    def _getSigmoid(self, x): return 1/(1 + np.exp(-x));
    def _getSigmoidPrime(self, x): return x * (1 - x);
    
    def _forward(self, input=None): #return result
        if input == None: input = self._flowerSize
        self._sigmoidedHiddenValues = self._getSigmoid(np.dot(input, self._synapses[0] ));
        return self._getSigmoid( np.dot(self._sigmoidedHiddenValues, self._synapses[1]) );

    def _backward(self, result):#directly updater synapses
        #compute the error
        outputDeltaError = (self._flowerTypes - result) * self._getSigmoidPrime(result);
        hiddenNeuronsDeltaError = outputDeltaError.dot(self._synapses[1].T) * self._getSigmoidPrime(self._sigmoidedHiddenValues);
        
        self._synapses[0] += self._flowerSize.T.dot(hiddenNeuronsDeltaError)
        self._synapses[1] += self._sigmoidedHiddenValues.T.dot(outputDeltaError);
        
    def train(self, x):
        for i in range(x):
            print(f"#{i}/{x}")
            result = self._forward();
            self._backward(result);

    def predict(self, input):
        result = self._forward([input]);
        print("Flower is blue" if result < 0.5 else "Flower is red");
