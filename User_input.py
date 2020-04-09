from get_equivalent_letter import get_letter
# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import Network2
import numpy as np
from NN_two_stage.SecondNN import get_let_from_2nd_nn_ijltIL1
from NN_two_stage.SecondNN import get_let_from_2nd_nn_ceg


def get_string_from_nn(all_letters):
    net = Network2.Network([1024, 30, 66], cost=Network2.CrossEntropyCost)

    biases_saved = np.load('biases.npy', allow_pickle=True, encoding='latin1')
    weights_saved = np.load('weights.npy', allow_pickle=True, encoding='latin1')

    # all_letters = np.load('all_letters.npy')
    # all_letters = all_letters.tolist()

    word_string = ""
    i = 0
    for x in all_letters:
        output = np.argmax(net.feedforward(x, biases_saved=biases_saved, weights_saved=weights_saved))

        # second stage classification below
        if output in (18, 19, 21, 29, 44, 47, 1):
            output = get_let_from_2nd_nn_ijltIL1(x)
        elif output in (12, 14, 42):
            output = get_let_from_2nd_nn_ceg(x)

        word_string = word_string + get_letter(output)
        i = i + 1

    return word_string
