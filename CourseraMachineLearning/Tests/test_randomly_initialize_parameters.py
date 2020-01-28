import unittest
from CourseraMachineLearning.Utility.neuralnetworks import randomly_initialize_parameters
import numpy as np

import math

class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_randomly_initialize_parameters(self):
        nodes_in_incoming_layer = 3
        nodes_in_outgoing_layer = 2

        given_results = np.zeros((nodes_in_incoming_layer,nodes_in_outgoing_layer))

        R = randomly_initialize_parameters(nodes_in_incoming_layer, nodes_in_outgoing_layer)

        self.assertNotEqual(given_results.all(), R.all())


if __name__ == '__main__':
    unittest.main()