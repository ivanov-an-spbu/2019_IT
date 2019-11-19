import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def load_csv(filename):
    df = pd.read_csv(filename)
    times = pd.to_datetime(df['Timestamp'], format=" %d/%m/%Y %H:%M:%S AM")
    df['Attack'] = df['Normal/Attack'] == 'Attack'

    df = df[['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'FIT201', 'Attack']]
    return times.as_matrix(), df.as_matrix().astype(float)


def mov_av(data, window_size = 10):
    weights = np.ones(window_size)
    a = np.convolve(data[:,0], weights, 'valid') / window_size
    res = np.empty((a.shape[0], data.shape[1]))
    res[:,0] = a
    for i in range(1, data.shape[1]):
        res[:,i] = np.convolve(data[:,i], weights, 'valid') / window_size
    res[res[:,-1]>0, -1] = 1
    return res


def LinearSystem_Predict(Xtrain, epoch_n=10, index = 0):
    k1= 0.001035
    k2=-0.00102
    A = np.array([1, k1, k2])

    X = Xtrain

    for i in range(X.shape[0]-1):
        ynext = np.dot(X[i], A)
        if i%epoch_n != 0:
            X[i+1, index] = ynext
    return X


if __name__ == "__main__":
    t, X = load_csv('P1.csv')
    X = X / X.max(0)

    #---------------------------------------------------------------------------
    X_train = X[:, [1, 2, 3]]
    t_train = t[:]
    X_eq_1 = LinearSystem_Predict(X_train.copy(), epoch_n=10000, index=0).copy()


    colors = ['r', 'y', 'c']
    names = ['attack MV101', 'attack P102', 'attack LIT101']

    X[X[:, 2] < 0.5, 2] = 0.5
    X[:, 3]-=1


    plt.plot(t, X[:, 1], alpha=0.7, label='LIT101 (sensor)')
    plt.plot(t, X[:, 2], alpha=0.4, label='MV101 (actuator)')
    plt.plot(t, X[:, 3], alpha=0.4, label='P101 (actuator)')
    plt.plot(t, X_eq_1[:, 0], color='r',  alpha=0.9, label='LIT101 (prediction)')

    plt.legend()
    plt.show()