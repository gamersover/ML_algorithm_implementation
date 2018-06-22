#coding:utf-8
"""
logistic回归算法实现
"""
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressor(object):
    def __init__(self, lr=0.01, batch_size=1, epochs=50):
        """
                        为什么batch_size不是1的时候，学习很差
        """
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y, shuffle=True, verbose=0, optimizer='sgd'):
        self.x = np.hstack([np.ones([X.shape[0], 1]), X])
        self.y = y.reshape([y.shape[0], 1])
        self.verbose = verbose
        self.optimizer = optimizer
        self.shuffle = shuffle
        self._num_data = self.x.shape[0]
        self._num_feat = self.x.shape[1]
        self._optimize()

    def _optimize(self):
        self.w = np.random.randn(self._num_feat, 1)

        for epoch in range(self.epochs):
            batches = self._get_batch()
            total_loss = 0
            for batch_x, batch_y in batches:
                logits = 1 / (1 + np.exp(-batch_x @ self.w))
                loss = batch_y - logits
                total_loss += np.sum(np.abs(loss))
                grad_w = (loss.T @ batch_x).T
                self.w += self.lr * grad_w / batch_x.shape[0]
            avg_loss = total_loss / self._num_data

            if self.verbose == 1:
                print("epoch:{} avg_loss:{}".format(epoch, avg_loss))

    def _get_batch(self):
        if self.shuffle:
            idx = np.arange(self._num_data)
            np.random.shuffle(idx)
            shuffle_x = self.x[idx, :]
            shuffle_y = self.y[idx, :]

        num_batch = np.ceil(self._num_data / self.batch_size).astype(int)
        for i in range(num_batch):
            start = i * self.batch_size
            end = min(start+self.batch_size, self._num_data)
            yield shuffle_x[start:end, :], shuffle_y[start:end, :]

    def predict(self, x):
        logits = 1 / (1 + np.exp(-x @ self.w))
        pre_label = np.where(logits<=0.5, 0, 1)
        return pre_label

    def get_accuracy(self):
        self.pre_label = self.predict(self.x)
        return np.mean(np.equal(self.pre_label, self.y).astype(int))


if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    y1 = 2 * x + 4 + np.random.random(100)
    y2 = 2 * x + 3 + np.random.random(100)
    data = np.hstack([np.array([x, y1]), np.array([x, y2])]).T
    label = np.concatenate([np.ones(100).ravel(), np.zeros(100).ravel()])
    lr = LogisticRegressor(epochs=100)
    lr.fit(data, label, verbose=0)
    print(lr.get_accuracy())
    w = lr.w
    border = -w[1]/w[2] * x - w[0]/w[2]
    plt.scatter(x, y1)
    plt.scatter(x, y2)
    plt.plot(x, border)
    plt.show()
