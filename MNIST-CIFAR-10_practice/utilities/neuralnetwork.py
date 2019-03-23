import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        #Initalize the list of weights martices
        #Store the network architecture and learning rateself.

        #Layers: [2,2,1] imply 2 - 2 -1 architecture.

        self.W =[]
        self.layers = layers
        self.alpha = alpha

        # Initalize weights
        # arange(start, stop), so 0, 1, ..., len(layers) -2
        for i in np.arange(0, len(layers) - 2):
            #Rnadom weights

            # np.random.randn(x, y) Return 2D matrix x by y
            # + 1 for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)

            # Scale w by dividing by sqrt of # of nodes
            # This normalizes the varianceself.

            # append adds to the end of the array.
            self.W.append(w / np.sqrt(layers[i]))

        # We stopped at len(layers) - 2 because the last two layers are a special case.
        # Output do not need bias.
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # print out 2-2-1
        return "NeuralNetwork: {}".format(
        "-".join(str(i) for i in self.layers))

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))


    # For back propergation, activation must be differentiable
    def sigmoid_deriv(self, x):
        return x * (1-x)

    # We train out function here.
    # X is training data
    # y is the corresponding calss label
    def fit(self, X, y, epochs = 1000, displayUpdate = 100):
        #np.c_(x,y) concatenates side by side
        #X.shape[0] => n X.shape[1] => m
        # add a column of 1's; np.ones(n, m(optional))

        X = np.c_[X, np.ones(X.shape[0])]

        for epoch in np.arange(0, epochs):
            #loop over each data point and training

            # x (element of X), target (element of y)
            # zip(x,y) return tuples: (x1, y1),(x2, y2) ...
            for (x, target) in zip(X, y):
                #  Make prediction, backcompute with this.
                self.fit_partial(x, target)

            # Display message

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                epoch + 1, loss))

    #The heart of backpropagation
    # 2 parameters => indivdual data point & class label
    def fit_partial(self, x, y):
        # Atleast_2d return an atleast 2d array.
        # A will be responsible for storing output activations.
        A = [np.atleast_2d(x)]

        # Feed Forward
        # 0, 1 ..., n layers
        for layer in np.arange(0, len(self.W)):
            # Feed foward the activation at current layer.

            net = A[layer].dot(self.W[layer])


            out = self.sigmoid(net)

            # Add to the list of activations.
            A.append(out)

        # Back Propagation
        # First, compute the difference.

        # -1 index points to the last layer, aka output layer.
        error = A[-1] - y

        # We build our list of derivatives.
        D = [error * self.sigmoid_deriv(A[-1])]

        # Ignore the last 2 layers as we taken account of these already.
        for layer in np.arange(len(A) - 2, 0, -1):

            # Delta of current layer is last layer dotted with weight matrix
            delta = D[-1].dot(self.W[layer].T)

            # d Error / d Out * d Out / d net
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # We need to reverse the deltas.
        D = D[::-1]

        # Weight Update

        for layer in np.arange(0, len(self.W)):

            # Dot product of layer activation and deltas.

            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    def predict(self, X, addBias = True):
        # Initalize as input features.
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)

        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
