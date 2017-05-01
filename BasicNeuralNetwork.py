#!/usr/bin/env python
################################################################################
# cool resources:
#
# https://archive.ics.uci.edu/ml/datasets.html
# https://archive.ics.uci.edu/ml/datasets/Dow+Jones+Index
# http://cs229.stanford.edu/proj2013/DaiZhang-MachineLearningInStockPriceTrendForecasting.pdf
# http://www.cs.bris.ac.uk/~flach/mlbook/materials/mlbook-beamer.pdf
# Source: http://aaronyool.blogspot.com.au/2017/04/the-beginnings-to-neural-network-module.html

# Modified for Python3 by Coder420 (Karan Goda)


from numpy import exp, array, random, dot, mean, abs, atleast_1d
from datetime import datetime
import _pickle as cPickle


class FFNeuralNetwork():
    # feed forward NN
    def __init__(self, inputs, outputs, layers, bias = 1):
        if layers < 3:
            print("Neural net must be >= 3")
            exit(0)
        # seed the generator, numpy requires this to work with time seed
        #random.seed([int(time.time() * 1e9) % 4294967296])
        random.seed(13) # debugging

        self.inputs   = inputs
        self.outputs  = outputs
        self.layers   = layers
        self.bias     = bias
        self.synapses = list()

        # number of synapses is one less than number of layers
        for s in range(self.layers - 1):
            if s == (self.layers - 2): # if this is the final synapse layer
                self.synapses.append(2 * random.random((self.inputs + 1, self.outputs)) - self.bias)
            elif s == 0: # if this is the input synapse layer
                self.synapses.append(2 * random.random((self.inputs, self.inputs + 1)) - self.bias)
            else:
                # this is a middle layer
                self.synapses.append(2 * random.random((self.inputs + 1, self.inputs + 1)) - self.bias)

    # sigmoid function, describes an S shaped curve
    # pass the weighted sum of the inputs to normalize between 1 and 0
    def __sigmoid(self, x, deriv=False):
        if deriv:
            # gradient of sigmoid curve
            return x * (1 - x)


        else:
            return 1 / (1 + exp(-x))

    # if there is enough divergence in the result, and the pattern of that
    # divergence matches the expected output, turn this on in think for faster training
    def __rectify(self, x):
        return (x + mean(x)) - 1

    def train(self, train_inputs, train_outputs, iterations):
        for j in range(iterations):
            # forward propogate layers
            layers = [train_inputs]
            for l in range(self.layers - 1):
                layers.append(self.__sigmoid(dot(layers[l], self.synapses[l])))

            # back propogate errors
            error  = train_outputs - layers[-1]
            deltas = [error * self.__sigmoid(layers[-1], deriv=True)]
            if (j % 10000) == 0:
                print("[+] %i Error: %s" % (j, mean(abs(error))))

            for s in range(len(self.synapses) - 1):
                index = (len(self.synapses) - 1) - s
                error = deltas[s].dot(self.synapses[index].T)
                deltas.append(error * self.__sigmoid(layers[index], deriv=True))

            # update synapses
            for s in reversed(range(len(self.synapses))):
                self.synapses[s] += layers[s].T.dot(deltas[(len(self.synapses) - 1) - s])
        return layers[-1]


    def think(self, x, rectify=False):
        # forward propogate and return a result
        layers = [x]
        for l in range(self.layers - 1):
            layers.append(self.__sigmoid(dot(layers[l], self.synapses[l])))
        return self.__rectify(layers[-1]) if rectify else layers[-1]

    def save_brain(self, s):
        pickle.dump(self.synapses, open(s, "wb" ))

    def restore_brain(self, s):
        self.synapses = pickle.load(open(s, "rb" ))

if __name__ == "__main__":
    import sys, os

    clock_flip = False
    total_start_clock = 0
    def timer():
        global clock_flip
        global total_start_clock
        if not clock_flip:
            clock_flip = True
            total_start_clock = datetime.now()
        else:
            clock_flip = False
            total_stop_clock = datetime.now()
            total_time = total_stop_clock - total_start_clock
            print("[*] Finished, time: %s" % total_time)

    def draw_shape(x):
        x = x.tolist()
        for v in range(len(x)):
            sys.stdout.write("%s%s" % (u"o" if x[v] else " ", "\n" if (v + 1) % 3 == 0 else ""))

    inputs = array([
    # diagonal
    [1,0,0,
     0,1,0,
     0,0,1],
    [0,0,1,
     0,1,0,
     1,0,0],
    [0,0,0,
     0,1,0,
     0,0,1],
    [0,0,0,
     0,1,0,
     1,0,0],
    [0,1,0,
     0,0,1,
     0,0,0],
    [0,1,0,
     1,0,0,
     0,0,0],
    [0,0,0,
     0,1,0,
     1,0,0],
    [0,0,0,
     0,1,0,
     0,0,1],
    [0,0,0,
     1,0,0,
     0,1,0],
    [0,1,0,
     1,0,1,
     0,1,0],
    [1,1,0,
     0,1,1,
     0,0,1],
    [0,0,0,
     1,1,0,
     0,1,1],
    [1,0,0,
     1,1,0,
     0,1,1],
    [0,1,0,
     1,0,1,
     0,0,0],
    [0,0,0,
     0,1,1,
     1,1,0],

    #horizontal
    [0,0,0,
     1,1,1,
     0,0,0],
    [1,1,1,
     0,0,0,
     0,0,0],
    [0,0,0,
     0,0,0,
     1,1,1],

    #verticle
    [0,1,0,
     0,1,0,
     0,1,0],
    [0,0,1,
     0,0,1,
     0,0,1],
    [1,0,0,
     1,0,0,
     1,0,0],
    [0,0,0,
     0,0,1,
     0,0,1],
    [0,0,0,
     1,0,0,
     1,0,0],
    [0,0,0,
     1,0,1,
     1,0,1],

    # weird shapes
    [1,1,1,
     1,0,0,
     1,0,0],

    [1,1,1,
     1,0,1,
     1,1,1],

    [1,1,1,
     1,1,0,
     1,0,1],

    [1,1,1,
     1,1,1,
     1,1,1],

    [1,0,1,
     1,0,1,
     1,1,1],

    [0,0,1,
     0,0,1,
     1,1,1],

    [1,0,0,
     1,0,0,
     1,1,1],

    [0,0,0,
     0,0,1,
     1,1,1],

    [0,0,0,
     1,0,0,
     1,1,1],

    [1,1,1,
     0,0,0,
     1,1,1],

    [0,0,0,
     1,1,1,
     1,1,1],

    [1,0,1,
     1,0,1,
     1,0,1],

    [0,1,1,
     0,1,1,
     0,1,1],

    [0,1,0,
     1,1,1,
     0,0,0],

    [0,1,0,
     1,0,1,
     0,0,0],

    [0,0,0,
     0,1,0,
     1,0,1],

    [0,0,0,
     0,1,0,
     1,1,1],

    [0,1,0,
     1,0,1,
     1,1,1],
    ])

    outputs = array([
       # d h v
       #diagonal
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [1,0,0],

       #horizontal
        [0,1,0],
        [0,1,0],
        [0,1,0],

       #verticle
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],

       #shape
        [0,1,1],
        [0,1,1],
        [1,1,1],
        [1,1,1],
        [0,1,1],
        [0,1,1],
        [0,1,1],
        [0,1,1],
        [0,1,1],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [1,1,0],
        [1,0,0],
        [1,0,0],
        [1,1,0],
        [1,1,1],
    ])

    # test matrix
    test = array([
        [0,1,0,
         1,0,1,
         1,1,1],

        [0,0,0,
         1,0,1,
         1,1,1],

        [1,1,1,
         1,0,1,
         1,1,1],

        [0,0,0,
         1,1,1,
         1,1,1],

        [0,0,0,
         0,1,1,
         1,1,0],

        [0,0,0,
         0,1,0,
         1,1,1],

        ])

    #initialize a neural network
    nn = FFNeuralNetwork(9, 3, 6, bias=1)

    # restore synapses
    if os.path.exists("shapes.pkl"):
        nn.restore_brain("shapes.pkl")

    #train this baby
    cycles = 5000000
    print("[*] Training for %i cycles" % cycles)

    timer()
    nn.train(inputs, outputs, cycles)
    timer()

    # store synapses after training
    nn.save_brain("binary_counter.pkl")

    print("[*] Getting test result")
    timer()
    result = nn.think(test, rectify=True)
    timer()

    print("[+] Results")
    for i in range(len(result)):
        draw_shape(test[i])
        if result[i][0] > 0:
            sys.stdout.write("d")
        if result[i][1] > 0:
            sys.stdout.write("h")
        if result[i][2] > 0:
            sys.stdout.write("v")
        sys.stdout.write("\n\n")
