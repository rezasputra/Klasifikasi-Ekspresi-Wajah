import numpy as np
import math

class Kernel:
    def __init__(self, kernel='rbf', gamma=10):
        self.kernel = kernel
        self.gamma = float(gamma)

    def calculate(self, x, z):
        return self._rbf_kernel(x, z)

    def _rbf_kernel(self, x, z):
        return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x, z).T, np.subtract(x, z)))


class SVM:
    def __init__(self, lamda=0.1, n_iter=100, gamma=10, C=1):
        self.lamda = lamda
        self.iteration = n_iter
        self.C = C

        self.gamma = gamma
        self.kernel = 'rbf'
        self.kernel_function = Kernel(self.kernel, gamma)

        self.b = 0
        self.alpa = 0

    def transform(self, X):
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.kernel_function.calculate(X[i], X[j])
        return K

    def hessian(self, kernel, y):
        hessian = np.zeros([kernel.shape[0], kernel.shape[0]])
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[0]):
                hessian[i, j] = (y[i] * y[j]) * (kernel[i, j] * math.pow(self.lamda, 2))
        return hessian

    # Mendapatkan nilai maksimal dan minimal alfa
    def getAlfaMax(self, alfa, y):
        pos = []
        neg = []
        for i in range(len(alfa)):
            if y[i] == 1:
                pos.append(alfa[i])
            elif y[i] == -1:
                neg.append(alfa[i])
        return max(pos), max(neg)

    # Menghitung nilai Bias
    def getB(self, alfa, kernel, y):
        alfa = list(alfa)
        pos, neg = self.getAlfaMax(alfa, y)
        indexPos = alfa.index(pos)
        indexNeg = alfa.index(neg)
        pos = 0
        neg = 0

        # Menghitung bobot kelas positif dan negatif
        for i in range(len(kernel)):
            pos += alfa[i] * y[i] * kernel[i][indexPos]
            neg += alfa[i] * y[i] * kernel[i][indexNeg]

        b = 0.5 * (pos + neg)
        return b

    def fit(self, X, y):
        # Buat Kernel Dulu
        self.kernel_rbf = self.transform(X)
        mat_hessian = self.hessian(self.kernel_rbf, y)

        a, b = mat_hessian.shape
        alfa = np.zeros(a)
        e = np.zeros(b)
        d = np.zeros(b)
        iterasi = 0

        while iterasi < self.iteration:
            for i in range(a):
                for j in range(b):
                    e[i] += mat_hessian[i, j] * alfa[j]
            for i in range(a):
                maks = max((self.gamma * (1 - e[i])), -alfa[i])
                d[i] += min(maks, (self.C - alfa[i]))
                alfa[i] = alfa[i] + d[i]
            iterasi += 1

        self.alpa = alfa

        self._support_labels = y
        self._support_vectors = X

    def signature(self, X):
        return np.where(X > 0, 1, -1)

    def hypothesis(self, X):
        b = self.getB(self.alpa, self.kernel_rbf, self._support_labels)
        hypothesis = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            # Calculate hypothesis for each given set of features
            h = 0
            for alpha, y, X_sv in zip(self.alpa, self._support_labels, self._support_vectors):
                h += alpha * y * self.kernel_function.calculate(X[i], X_sv)
            hypothesis[i] = h
        hypothesis = hypothesis
        return hypothesis

    def predict(self, X):
        return self.signature(self.hypothesis(X))


class MulticlassSVM:
    def __init__(self, lamda=0.000001, gamma=10, C=1, n_iter=10):
        self.lamda = lamda
        self.iteration = n_iter
        self.C = C

        self.gamma = float(gamma)
        self.kernel = 'rbf'
        self.kernel_function = Kernel(self.kernel, gamma)

    def _get_nummber_of_categories(self, labels):
        return len(np.unique(labels))

    def _create_one_vs_many_labels(self):
        for label in np.unique(self.y):
            self.labels[label] = np.copy(self.y)
            self.labels[label][self.labels[label] != label] = -1
            self.labels[label][self.labels[label] == label] = 1

    def _fit_one_vs_many_classifiers(self):
        for label in self.labels:
            print(f'SVM One-to-many untuk kelas: {label}')
            self.classifiers[label] = SVM(
                lamda=self.lamda, n_iter=self.iteration,
                gamma=self.gamma, C=self.C
            )
            self.classifiers[label].fit(self.X, self.labels[label])

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = {}
        self.classifiers = {}
        self.num_of_categories = self._get_nummber_of_categories(y)

        print(f'Terdapat {self.num_of_categories} Klasifikasi Biner...')

        # Creating separate output labels for different binary classifiers
        self._create_one_vs_many_labels()

        # Training different binary classifiers
        self._fit_one_vs_many_classifiers()

    def predict(self, X, flag):

        # Creating predictions for each binary classifier
        predictions = {}
        for label in self.classifiers:
            predictions[label] = self.classifiers[label].hypothesis(X)

        # Determining the class based on a maximun distance from a hyperplane
        y = []
        for p0, p1, p2, p3, p4, p5, p6 in zip(predictions[0], predictions[1], predictions[2], predictions[3],
                                              predictions[4], predictions[5], predictions[6]):
            if flag:
                p0 = abs(p0)
                p1 = abs(p1)
                p2 = abs(p2)
                p3 = abs(p3)
                p4 = abs(p4)
                p5 = abs(p5)
                p6 = abs(p6)

                single_observation_predictions = [p0, p1, p2, p3, p4, p5, p6]
                y.append(single_observation_predictions.index(max([p0, p1, p2, p3, p4, p5, p6])))
            else:
                single_observation_predictions = [p0, p1, p2, p3, p4, p5, p6]
                y.append(single_observation_predictions.index(max([p0, p1, p2, p3, p4, p5, p6])))
        return y

    def accuracy(y_pred, y_test):
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                correct += 1
        return correct / float(len(y_test)) * 100.0