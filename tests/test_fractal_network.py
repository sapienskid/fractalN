import unittest
import numpy as np
from src.models.fractal_neural_network import FractalNeuralNetwork

class TestFractalNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.network = FractalNeuralNetwork()
    
    def test_initialization(self):
        self.assertIsNotNone(self.network)
    
    def test_fractal_dimension(self):
        self.assertEqual(self.network.fractal_dimension, 1.5)

if __name__ == '__main__':
    unittest.main()
