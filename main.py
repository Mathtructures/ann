import numpy as np
import matplotlib.pyplot as plt


class Ann:

    def data_init(self, test_func=None, train_data=None, test_data=None):
        layers = [4, 5, 1]
        learningRate = 0.1
        weights = dict()
        biases = dict()
        if test_func != None:
            train_data, test_data = test_func(inputs=layers[0], batch_size=100)
        for i in range(len(layers[:-1])):
            weights[i] = np.random.randint(
                size=(layers[i+1], layers[i]), low=0, high=10)/10
            biases[i] = np.random.randint(
                size=(layers[i+1], 1), low=0, high=10)/10

        return {
            'layers': layers,
            'numLayers': len(layers) - 1,
            'learningRate': learningRate,
            'weights': weights,
            'biases': biases,
            'trainData': train_data,
            'testData': test_data,
            'layerInputs': dict(),
            'layerOutputs': dict(),
            'testResult': {
                'LMS_Hist': []
            }
        }

    def sigmoid(self, x):
        result = 1/(1 + np.exp(-x)) - 0.5
        return result

    def d_sigmoid(self, x):
        result = self.sigmoid(x)*(self.sigmoid(x) - 1)
        return result

    def activation_func(self, x, func):
        result = func(x)
        return result

    def d_f(self, x, d_func):
        result = d_func(x)
        return result

    def layer_input(self, data, layer):
        result = data['layerInputs'][layer]
        return result

    def layer_output(self, data, layer):
        result = data['layerOutputs'][layer]
        return result

    def forward_propagation(self, data, train_data_i):
        n = train_data_i[0]
        for i in range(data['numLayers']):
            data['layerInputs'][i] = n
            W = data['weights'][i]
            b = data['biases'][i]
            a = self.activation_func(np.dot(W, n) + b, func=self.sigmoid)
            data['layerOutputs'][i] = a
            n = a
        return data

    def neural_jacobian(self, data, layer,):
        W = data['weights'][layer]
        J = np.dot(W, np.diag(
            self.d_f(self.layer_input(data, layer), d_func=self.d_sigmoid)))
        return J

    def sensitivity(self, layer, last_layer, train_data_i, J):
        if layer < last_layer:
            S = np.dot(J.T, self.sensitivity(layer, last_layer))
        else:
            a = self.layer_output(last_layer)
            t = train_data_i[1]
            targetError = t - a
            S = np.dot(J, targetError)
        return S

    def update_network(self, data, train_data_i):
        weights = data['weights']
        biases = data['biases']
        lr = data['learningRate']
        M = data['numLayers']
        for m in range(data['numLayers']):
            W = weights[m]
            J = self.neural_jacobian(data=data, layer=m)
            W = W + lr*np.dot(self.sensitivity(m, M,
                              train_data_i, J), self.layer_output(m-1))
            b = biases[m]
            b = b + lr*self.sensitivity(m, M, train_data_i, J)
            weights[m] = W
            biases[m] = b
        data['weights'] = weights
        data['biases'] = biases
        return data

    def train(self, data):
        train_data = data['trainData']
        for i in train_data.keys():
            data = self.forward_propagation(data, train_data[i])
            data = self.update_network(data, train_data[i])
        return data

    def test(self, data):
        test_data = data['testData']
        for i in test_data.keys():
            data = self.forward_propagation(data, test_data[i])
            LMS = self.compare_results(data, test_data[i], method='LMS')
            data['testResult']['LMS_Hist'].append(LMS)
        return data

    def compare_results(data, test_data_i, method='LMS'):
        M = data['numLayers']
        a = data['layerOutputs']
        t = test_data_i
        if method == 'LMS':
            result = self.get_lms(t, a)
        return result

    def get_lms(self, t, a):
        result = sum((t - a)**2)
        return result


if __name__ == '__main__':
    def myFunc(a):
        r = 0
        for i in range(len(a)):
            r += a[i]**i
        return r

    def test_func(inputs, batch_size):
        train_data = dict()
        test_data = dict()
        for i in range(batch_size):
            if i/batch_size < 0.8:
                train_data[i] = dict()
                Xi = [np.random.randint(0, 100)/np.random.randint(1, 100)
                      for j in range(inputs)]
                train_data[i][0] = np.reshape(np.array(Xi), (inputs, 1))
                train_data[i][1] = myFunc(train_data[i][0])
            else:
                test_data[i] = dict()
                Xi = [np.random.randint(0, 100)/np.random.randint(1, 100)
                      for j in range(inputs)]
                test_data[i][0] = np.reshape(np.array(Xi), (inputs, 1))
                test_data[i][1] = myFunc(test_data[i][0])

        return train_data, test_data

    def main():
        MyAnn = Ann()
        data = MyAnn.data_init(test_func=test_func)
        data = MyAnn.train(data)
        data = MyAnn.test(data)
        plt.plot(data['testResult']['LMS_Hist'])
        return 0
    main()
