import numpy as np
# import matplotlib.pyplot as plt


class ArtNeuNet:
    def __init__(self, hidden_layers, inputs_len, targets_len):
        # self.activation_function_type = 'sigmoid'
        self.activation_function_type = 'relu'
        # self.activation_function_type = 'relu_pow'
        # self.activation_function_type = 'sin'
        # self.activation_function_param = 1
        self.activation_function_param = .1
        # self.activation_function_param = 2
        self.all_layers = [inputs_len] + hidden_layers + [targets_len]

        self.learning_rate = 1e-2

        self.a_i = {
            i: dict() for i in range(1, len(self.all_layers))
        }
        self.Z_i = {
            i: dict() for i in range(1, len(self.all_layers))
        }
        self.epoch = 0
        self.network_gradient = dict()

    def initialize_params(self,):
        W_domain = 1e3
        b_domain = 1
        scale_w = .1/W_domain
        scale_b = .01/b_domain
        params = {
            i: {
                'W': scale_w*np.random.randint(low=-W_domain, high=W_domain, size=(self.all_layers[i], self.all_layers[i-1])),
                'b': scale_b*np.random.randint(low=-b_domain, high=b_domain, size=(self.all_layers[i], 1))
            }
            for i in range(1, len(self.all_layers))
        }
        # params = {
        #     i: {
        #         'W': np.random.randint(low=1e6, high=1e6+1, size=(self.all_layers[i], self.all_layers[i-1]))/1e6,
        #         'b': np.random.randint(low=0, high=1, size=(self.all_layers[i], 1))/1e6
        #     }
        #     for i in range(1, len(self.all_layers))
        # }

        return params

    def set_data(self, data_pairs, train_part=0.5):
        all_inputs = data_pairs['inputs']
        all_targets = data_pairs['targets']
        dataset_size = len(all_inputs[0, :])
        train_index = [0, int(train_part*dataset_size)+1]
        test_index = [int(train_part*dataset_size)+1, dataset_size]
        train_inputs = all_inputs[:, train_index[0]:train_index[1]]
        test_inputs = all_inputs[:, test_index[0]:test_index[1]]
        train_targets = all_targets[:, train_index[0]:train_index[1]]
        test_targets = all_targets[:, test_index[0]:test_index[1]]
        train_data = {
            'inputs': train_inputs,
            'targets': train_targets,
        }
        test_data = {
            'inputs': test_inputs,
            'targets': test_targets,
        }
        data = {
            'train': train_data,
            'test': test_data,
        }
        return data

    def activation_function(self, z):
        type = self.activation_function_type
        p = self.activation_function_param
        if type == 'sigmoid':
            a = 1/(1+np.e**-z)
            return a
        elif type == 'relu':
            a = z.copy()
            a[a < 0] = p*a[a < 0]
            return a
        elif type == 'relu_pow':
            a = z.copy()
            a[a < 0] = 0.1*a[a < 0]
            a[a > 0] = a[a > 0]**p
            return a
        elif type == 'sin':
            a = np.sin(p*z)
            return a
        elif type == 'step':
            a = z.copy()
            a[a < p] = 0
            a[a > p] = 1
            return a
        elif type == 'poly':
            a = z.copy()
            return a**p

    def activation_function_derivation(self, z):
        type = self.activation_function_type
        p = self.activation_function_param
        if type == 'sigmoid':
            f = self.activation_function(z)
            df = f*(f - 1)
            return df
        elif type == 'relu':
            da = z.copy()
            da[da < 0] = p
            da[da > 0] = 1
            return da
        elif type == 'relu_pow':
            da = z.copy()
            da[da < 0] = 0.1
            da[da > 0] = p*da[da > 0]**(p - 1)
            return da
        elif type == 'sin':
            a = p*np.cos(p*z)
            return a

    def get_layer_output(self, i, input_vec, params):
        W = params[i]['W']
        b = params[i]['b']
        Z = np.dot(W, input_vec) + b
        return Z

    def get_next_layer_input(self, pre_output):
        a = self.activation_function(pre_output)
        return a

    def forward_propagation(self, input_vec, params):
        a = input_vec.copy()
        self.a_i[0] = a.copy()
        for i in range(1, len(self.all_layers)):
            Z = self.get_layer_output(i, a, params)
            a = self.get_next_layer_input(Z)
            self.Z_i[i] = Z.copy()
            self.a_i[i] = a.copy()
        return a

    def update_params(self, params, netgrad):
        for L in params.keys():
            dW = netgrad['dW'][L]
            params[L]['W'] -= self.learning_rate*dW
            db = netgrad['db'][L]
            params[L]['b'] -= self.learning_rate*db
        return params

    def dloss_dz_m(self,params,Sm,dLoss_dO,i,Zi):
        if i+1 in Sm.keys():
            W = params[i+1]['W']
            SM = Sm[i+1]
            dai_dZi = self.activation_function_derivation(Zi)
            dZ1_dZ0 = np.dot(W,np.diag(dai_dZi.T.tolist()[0]))
            S = np.dot(dZ1_dZ0.T,SM)
        else:
            dai_dZi = self.activation_function_derivation(Zi)
            S = dLoss_dO*dai_dZi
        return S

    def unify_vector(self,V,accept_small=False):
        if sum(V**2)**0.5 < 1 and accept_small:
            return V
        V = V/sum(V**2)**0.5
        return V

    def get_network_gradient(self, params, O, T,input_set):
        dLoss_dO = -2*(T - O)
        # dLoss_dO = sum(dLoss_dO)
        Sm = dict()
        dLoss_dW = dict()
        dLoss_db = dict()
        for i in range(len(self.all_layers)-1, 0, -1):
            ai = self.a_i[i][:,input_set]
            ai = np.reshape(ai,(len(ai),1))
            Zi = self.Z_i[i][:,input_set]
            Zi = np.reshape(Zi,(len(Zi),1))
            S = self.dloss_dz_m(params,Sm,dLoss_dO,i,Zi)
            S = self.unify_vector(S,accept_small=True)
            Sm[i] = S
            a = self.a_i[i-1][:,input_set]
            a = np.reshape(a,(1,len(a)))
            dLoss_dW[i] = np.dot(S,a)
            dLoss_db[i] = S.copy()
        network_gradient = {
            'dW':dLoss_dW,
            'db':dLoss_db,
        }
        return network_gradient

    def backward_propagation(self, input_vec, output_vec, target_vec, params,input_set):
        network_gradient = self.get_network_gradient(
            params, output_vec, target_vec,input_set)
        params = self.update_params(params,network_gradient)
        return params

    def train(self, train_data, epochs, params):
        # fig_train = plt.figure(0)
        losses = list()
        # for i in range(epochs):
        do_train = True
        do_train_count = 1
        max_loss_old = 1e10
        LR_change_1 = 0
        LR_change_2 = 0
        LR_change_3 = 0
        input_mat = train_data['inputs']
        target_mat = train_data['targets']
        while do_train:
            output_mat = self.forward_propagation(input_mat, params)
            # loss = ((target_mat - output_mat)**2)**0.5
            losses = np.sum((target_mat - output_mat)**2, axis=0)**0.5
            max_loss_new = max(losses)
            # if max_loss_new < max_loss_old:
            #     self.learning_rate *= 0.5
            # else:
            #     self.learning_rate /= 0.5

            print(max_loss_new)
            if max_loss_new <= 1e-1:
                do_train_count -= 1
            elif max_loss_new > max_loss_old:
                do_train_count += 1

            # if max_loss_new < max_loss_old and (max_loss_old - max_loss_new) < .001:
            #     self.learning_rate *= (max_loss_old - max_loss_new)

            if max_loss_new < 1 and LR_change_1:
                self.learning_rate *= 0.1
                LR_change_1 -= 1

            if max_loss_new < 0.5 and LR_change_2:
                self.learning_rate *= 0.5
                LR_change_2 -= 1

            if max_loss_new < 0.2 and LR_change_3:
                self.learning_rate *= 0.5
                LR_change_3 -= 1


            if do_train_count:
                do_train = True
            else:
                do_train = False
            max_loss_old = max_loss_new

            # if max_loss_new > 1e10:
            #     do_train = False
            # losses.append(loss)
            # plt.plot(i,loss,'r.')
            # plt.show(block=False)
            # plt.pause(0.001)
            for i in range(len(input_mat[0, :])):
                input_vec = input_mat[:, i]
                target_vec = target_mat[:, i]
                # output_vec = output_mat[:, i]
                input_vec = np.reshape(input_vec, (len(input_vec), 1))
                target_vec = np.reshape(target_vec, (len(target_vec), 1))
                output_vec = self.forward_propagation(input_vec, params)
                output_vec = np.reshape(output_vec, (len(output_vec), 1))
                params = self.backward_propagation(
                    input_vec, output_vec, target_vec, params,input_set=0)
        return params

    def test(self, test_data, params):
        inputs_vec = test_data['inputs']
        targets_vec = test_data['targets']
        prediction_vec = self.forward_propagation(inputs_vec, params)
        for i in range(len(inputs_vec[0, :])):
            print(f"inputs: {inputs_vec[:, i]}")
            print(f"prediction: {prediction_vec[:, i]}")
            print(f"target: {targets_vec[:, i]}")
            print(f"\n")


def XOR_operator(x, y):
    vec_sum = [1 if (x[i] and not y[i]) or (not x[i] and y[i])
               else 0 for i in range(len(x))]
    vec_sum = np.array(vec_sum)
    vec_sum = np.reshape(vec_sum,(1,len(vec_sum)))
    return vec_sum


def AND_operator(x, y):
    vec_sum = [1 if x[i] and y[i] else 0 for i in range(len(x))]
    f = np.array(vec_sum)
    f = np.reshape(f,(1,len(f)))
    return f

def OR_operator(x, y):
    vec_sum = [1 if x[i] or y[i] else 0 for i in range(len(x))]
    f = np.array(vec_sum)
    f = np.reshape(f,(1,len(f)))
    return f


def power(x, p):
    f = np.reshape(x**p,(1,len(x)))
    return f

def sumul(x):
    f = np.zeros((1,len(x[0,:])))
    f[0,:] = x[0,:] * x[1,:]
    # f[0,:] = x[0,:] + 3*x[1,:]
    # f[0,:] = x[0,:] + x[1,:]
    # f[1,:] = x[0,:] - x[1,:]
    # f[1,:] = ((x[0,:] - x[1,:])**2)**0.5
    return f

def sine(x,w):
    fx = np.sin(w*x)
    return fx

if __name__ == '__main__':
    # inputs = np.random.randint(low=-4, high=4, size=(1, 100))
    # inputs = np.random.randint(low=-20, high=20, size=(1, 100))
    inputs = np.random.random(size=(2,100))*20
    # inputs = np.random.randint(low=0, high=2, size=(2, 100))
    # targets = power(inputs[0, :], 2)
    # targets = inputs[0, :]*inputs[1, :]
    # targets = XOR_operator(inputs[0, :], inputs[1, :])
    targets = OR_operator(inputs[0, :], inputs[1, :])
    inputs = np.round(inputs,decimals=1)
    # targets = sumul(inputs)
    # targets = sine(inputs,.2)
    data_pairs = {
        'inputs': inputs,
        'targets': targets,
    }
    ann = ArtNeuNet(hidden_layers=[2], inputs_len=len(
        inputs[:, 0]), targets_len=len(targets[:, 0]))
    params = ann.initialize_params()
    data = ann.set_data(data_pairs=data_pairs, train_part=0.8)
    params = ann.train(train_data=data['train'], epochs=300, params=params)
    test_results = ann.test(test_data=data['test'], params=params)
    print('Done\n\n\n')
    # ann.set
