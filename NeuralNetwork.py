import numpy as np
import time

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.ip_size = input_size
        self.hd_size = hidden_size
        self.op_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size)  ## Weight of hidden layer [784, 20]
        self.b1 = np.zeros((1, hidden_size))                ## Bias of hidden layer [1, 20]
        self.W2 = np.random.randn(hidden_size, output_size) ## Weight of output layer [20, 10]
        self.b2 = np.zeros((1, output_size))                ## Bais of output layer [1, 20]

    def activate(self, x):
        return np.maximum(0,x)  ## ReLU function -- if x>0 ? x:0

    def soft_max(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1          ## input.weighy:1 + bias:1
        self.a1 = self.activate(self.z1)                ## activating z1 using ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2    ## hidden.weighy:2 + bias:2
        self.probs = self.soft_max(self.z2)             ## finding the probability %
        return self.probs

    def loss(self, X, y):
        num_examples = len(X)
        corect_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        return 1/num_examples * data_loss
    
    def back_prop(self, X, y):
        learning_rate = 0.01

        error = self.probs
        error[range(len(X)), y] -= 1
        error /= len(X)

        dW2 = np.dot(self.a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)

        hidden_error = np.dot(error, self.W2.T)
        hidden_error[self.a1 <= 0] = 0

        dW1 = np.dot(X.T, hidden_error)
        db1 = np.sum(hidden_error, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, eterate):
        st = time.time()
        sample_size = X.shape[0]
        sub_matrix_size = 10
        for e in range(eterate):
            print('\033[0m\033[31mTraining session :', e, 'Starting time :', time.ctime())
            for i in range(0, sample_size, sub_matrix_size):
                sub_matrix_X = X[i:i+sub_matrix_size]
                sub_matrix_y = y[i:i+sub_matrix_size]

                self.forward(sub_matrix_X)
                loss = self.loss(sub_matrix_X, sub_matrix_y)
                self.back_prop(sub_matrix_X, sub_matrix_y)


                if i%5000 == 0:
                    tt = round(time.time()-st, 5)
                    print('\033[0mTime passes :\033[32m',tt,'\033[0m\tTotal data loss :\033[32m', round(loss, 3))

        print('\n\033[0m\033[2mtraining complete\033[0m\n')

    def predict(self, X):
        self.forward(X)
        return self.probs