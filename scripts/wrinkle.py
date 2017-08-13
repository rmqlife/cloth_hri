import os, cv2
import matplotlib.pyplot as plt
import numpy as np

def mk(s):#morph kernel
    return np.ones((s,s),np.uint8)

def gabor_feat(img, num_theta = 8, grid = 40, show_fg=False,show_step=False):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ks = 101 # kernel size
    thresh = 200
    
    avg = np.zeros(gray.shape)
    count = 0
    
    rows, cols = gray.shape

    hist = []
    for i in range(num_theta):
        g_kernel1 = cv2.getGaborKernel((ks, ks), sigma = 6.0, theta = i*np.pi/num_theta, lambd = 10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel1)
        count = count+1
        
        avg = avg + filtered
        
        count = 0
        for col in np.arange(0, cols, grid):
            for row in np.arange(0, rows, grid):
                block = filtered[row:row+grid,col:col+grid]
                val =  float(np.sum(block))
                hist.append(val)
        if show_step:
            plt.imshow(filtered)
            plt.show()
    avg = avg/float(num_theta)
    avg = np.uint8(avg)
    return avg, hist

if __name__ == "__main__":
    im = cv2.imread('cloth.jpg')
    avg, hist = gabor_feat(im)
    print hist
