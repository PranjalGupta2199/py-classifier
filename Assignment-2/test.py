import numpy as np
from neural_net import NeuralNet

nn = NeuralNet(
    learning_rate=1,
    layer_dims=[2, 3, 1],
    actn_fn=['sigmoid', 'sigmoid'],
    initializer='random'
)
# print(nn)

X = np.array([[0,0], [0,1], [1,0], [1,1]]).T
Y = np.array([[0], [1], [1], [0]]).T

nn.layers[0].w = np.array([[0.1, 0.6], [0.2, 0.4], [0.3, 0.7]])
nn.layers[0].b = np.array([[0], [0], [0]])

nn.layers[1].w = np.array([[0.1, 0.4, 0.9]])
nn.layers[1].b = np.array([[0]])

for i in range (5000):
    forward_val = nn(X)
    # print ("Forward :", forward_val)
    print ("Error: ", nn.error(Y, nn.layers[-1].activation_fn.prev))
    nn.backward(X, Y, forward_val)

y_pred = nn.classify(nn(X).reshape(-1))
print (y_pred)
print (nn.score(Y.reshape(-1), y_pred))