import unittest
from CourseraMachineLearning.Utility.neuralnetworks import compute_gradients_with_back_propagation, randomly_initialize_parameters
import numpy as np

import math

class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_compute_gradients_with_back_propagation(self):
        nodes_in_incoming_layer = 3
        nodes_in_outgoing_layer = 5
        num_labels = 3
        number_of_training_samples = 5

        theta_1 = randomly_initialize_parameters(nodes_in_outgoing_layer, nodes_in_incoming_layer + 1)
        theta_2 = randomly_initialize_parameters(num_labels, nodes_in_outgoing_layer + 1)
        X = randomly_initialize_parameters(number_of_training_samples, nodes_in_incoming_layer)
        y = np.array((1,2,3,3,1))
        y = y.reshape(1,len(y))

        X = X.transpose()

        parameters = np.hstack((theta_1.transpose().flatten(), theta_2.transpose().flatten()))
        grad_bp = compute_gradients_with_back_propagation(parameters,
                                                          nodes_in_incoming_layer,
                                                          nodes_in_outgoing_layer,
                                                          X, y, num_labels,
                                                          lambda_for_regularization=0)
        #TODO: implement gradient calculation with numerical gradient
        print(grad_bp)

if __name__ == '__main__':
    unittest.main()