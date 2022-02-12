#-- coding: utf-8 --
#@Time : 2021/3/27 20:40
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : RVFL.py
#@Software: PyCharm

import numpy as np


class RVFL:
    """A simple RVFL classifier or regression.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """
    def __init__(self, n_nodes, lam, w_random_vec_range, b_random_vec_range, activation, same_feature=False,
                 task_type='classification'):
        assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        self.task_type = task_type

    def train(self, data, label, n_class):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        data = self.standardize(data)  # Normalization data
        n_sample = len(data)
        n_feature = len(data[0])
        self.random_weights = self.get_random_vectors(n_feature, self.n_nodes, self.w_random_range)
        self.random_bias = self.get_random_vectors(1, self.n_nodes, self.b_random_range)

        h = self.activation_function(np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias))
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        if self.task_type == 'classification':
            y = self.one_hot(label, n_class)
        else:
            y = label
        if n_sample > (self.n_nodes + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict(self, data):
        """

        :param data: Predict data.
        :return: When classification, return Prediction result and probability.
                 When regression, return the output of rvfl.
        """
        data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            return result, proba
        elif self.task_type == 'regression':
            return output

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return accuracy.
                 When regression return MAE.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            result = np.argmax(output, axis=1)
            acc = np.sum(np.equal(result, label)) / len(label)
            return acc
        elif self.task_type == 'regression':
            mae = np.mean(np.abs(output - label))
            return mae

    @staticmethod
    def get_random_vectors(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x):
        if self.same_feature is True:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x, axis=0)
            return (x - self.data_mean) / self.data_std

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


if __name__ == '__main__':
    import sklearn.datasets as sk_dataset


    def prepare_data_classify(proportion):
        dataset = sk_dataset.load_breast_cancer()
        label = dataset['target']
        data = dataset['data']
        n_class = len(dataset['target_names'])

        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]
        data_train = data[train_index]
        label_train = label[train_index]
        data_val = data[val_index]
        label_val = label[val_index]
        return (data_train, label_train), (data_val, label_val), n_class

    def prepare_data_regression(proportion):
        dataset = sk_dataset.load_diabetes()
        label = dataset['target']
        data = dataset['data']

        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)

        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]
        data_train = data[train_index]
        label_train = label[train_index]
        data_val = data[val_index]
        label_val = label[val_index]
        return (data_train, label_train), (data_val, label_val)

    # Classification
    num_nodes = 2  # Number of enhancement nodes.
    regular_para = 1  # Regularization parameter.
    weight_random_range = [-1, 1]  # Range of random weights.
    bias_random_range = [0, 1]  # Range of random weights.

    train, val, num_class = prepare_data_classify(0.8)
    rvfl = RVFL(n_nodes=num_nodes, lam=regular_para, w_random_vec_range=weight_random_range,
                     b_random_vec_range=bias_random_range, activation='relu', same_feature=False,
                     task_type='classification')
    rvfl.train(train[0], train[1], num_class)
    prediction, proba = rvfl.predict(val[0])
    accuracy = rvfl.eval(val[0], val[1])
    print('Acc:', accuracy)

    # Regression
    num_nodes = 10  # Number of enhancement nodes.
    regular_para = 1  # Regularization parameter.
    weight_random_range = [-1, 1]  # Range of random weights.
    bias_random_range = [0, 1]  # Range of random weights.

    train, val = prepare_data_regression(0.8)
    rvfl = RVFL(n_nodes=num_nodes, lam=regular_para, w_random_vec_range=weight_random_range,
                     b_random_vec_range=bias_random_range, activation='relu', same_feature=False,
                     task_type='regression')
    rvfl.train(train[0], train[1], 0)
    prediction = rvfl.predict(val[0])
    mae = rvfl.eval(val[0], val[1])
    print('MAE:', mae)
