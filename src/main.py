import sys
import numpy as np
from Flower import Flower
from NeuralNetwork import NeuralNetwork
import os


def getFlowerNpArrayFromFile(filepath = os.path.dirname(os.path.abspath(__file__)) + "/../flowerDataset.txt"):
    lines = open(filepath).readlines();
    flowers = []
    
    for line in lines:
        flowerData = line.split()
        flowers.append(
            Flower(
                float(flowerData[0]),
                float(flowerData[1]),
                1 if flowerData[2] == 'red' else 0
            )
        )
    return np.array(flowers);

userInput = np.array((sys.argv[1:])).astype(float)        

neuralNetwork = NeuralNetwork(3, getFlowerNpArrayFromFile());
neuralNetwork.train(30000)
neuralNetwork.predict(userInput);