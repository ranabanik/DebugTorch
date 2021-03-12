"""Adaptation from Andrey Karpathy"""
import random
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self.backward = lambda: None

    def __add__(self, other):


    def __radd__(self, other):


tf = Value(5)
print(tf.a)
