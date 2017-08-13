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
    vel = data['vel']
    pos = data['pos']
    feat = data['feat']
    y = pos[30:]-pos[:-30]

    from scipy.ndimage.filters import uniform_filter1d
    #compute the gradient of the feature vector
    X = feat[30:]-feat[:-30]
    X = uniform_filter1d(X,5,axis=0) # smooth the input data by 5

    print X.shape, y.shape
    X,y = same_size(X,y)

    print X.shape, y.shape

    # begin training
    from sklearn import linear_model
    model = linear_model.Lasso(alpha =1)

    inds = sorted(np.random.choice(len(X), 500, replace=False))
    X_train = X[inds]
    y_train = y[inds]
    model.fit(X_train, y_train)
    return model

if __name__=='__main__':

    data_name = 'data2_grid.npz'
    # load model
    model = load_model(data_name)
    
    data = np.load(data_name)
    vel = data['vel']
    pos = data['pos']
    feat = data['feat']
    y = pos[30:]-pos[:-30]

    from scipy.ndimage.filters import uniform_filter1d
    #compute the gradient of the feature vector
    X = feat[30:]-feat[:-30]
    # Use the model to predict the velocity/direction of end effector
    target = 0  # target index  
    X = feat - feat[target]
    y = pos - pos[target]

    y_pred = model.predict(X)
    for i in range(y_pred.shape[1]):
        plt.plot(range(y_pred.shape[0]),y_pred[:,i])
        plt.plot(range(y.shape[0]),y[:,i])
        plt.show()
