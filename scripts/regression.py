import os, cv2
import matplotlib.pyplot as plt
import numpy as np

def same_size(X,y):
    minlength = min(y.shape[0],X.shape[0])
    print minlength
    y = y[:minlength]
    X = X[:minlength]
    return X,y

def load_model(data_name):
    data = np.load(data_name)
    pos = data['pos']
    feat = data['feat']

    inds = np.random.choice(a=len(feat),size=4*len(feat))
    if len(inds)%2 == 1:
        inds=inds[:-1]
    nd = len(inds)/2 # number of deltas
    X_train = feat[inds[:nd]] - feat[inds[nd:]]
    y_train = pos[inds[:nd]] - pos[inds[nd:]]
    X_train, y_train = same_size(X_train, y_train)
    print X_train.shape, y_train.shape

    # begin training
    from sklearn import linear_model
    model = linear_model.Lasso(alpha = 0.00001)
    model.fit(X_train, y_train)
    return model
