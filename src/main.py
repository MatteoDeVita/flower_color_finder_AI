import sys
import numpy as np
from Flower import Flower
import os


def getFlowerNpArrayFromFile(filepath = os.path.dirname(os.path.abspath(__file__)) + "/../flowerDataset.txt"):
    lines = open(filepath).readlines();
    flowers = []
    
    for line in lines:
        flowerData = line.split()
        flowers.append(Flower(float(flowerData[0]), float(flowerData[1]), flowerData[2] ))
    return np.array(flowers);

dataset = getFlowerNpArrayFromFile();

userInput = np.array(sys.argv[1:]).astype(float)        

