import os, cv2
import matplotlib.pyplot as plt
import numpy as np

# extract the foreground
def get_fg(img):
    markers = np.uint8(np.zeros((480,640)))
    markers[420:, 280:360] =1
    # set background as 2
    markers[:,0:20] = 2
    markers[:,-20:] = 2
    markers[0:20,:] = 2
    markers_rgb = markers.astype(np.int32)
    markers_rgb = cv2.watershed(img,markers_rgb)
    fg = np.uint8(markers_rgb == 1)
    return fg

def mk(s):#morph kernel
    return np.ones((s,s),np.uint8)

def gabor_feat(img, num_theta = 4, show_fg=False,show_step=False):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fg = get_fg(img)
    shrink_fg = cv2.erode(fg, mk(15), iterations = 2)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask = fg)
    if show_fg:
        plt.imshow(gray,cmap='gray')
        plt.show()
    
    ks = 101 # kernel size
    thresh = 200
    
    avg = np.zeros(gray.shape, dtype=bool)
    count = 0
    
    rows, cols = gray.shape
    grid = 80

    hist = []
    for i in range(num_theta):
        g_kernel1 = cv2.getGaborKernel((ks, ks), sigma = 6.0, theta = i*np.pi/num_theta, lambd = 10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel1)
        count = count+1
        
        filtered = np.array(filtered>thresh, dtype=bool)
        filtered = filtered & shrink_fg
        avg = avg | filtered
        
        count = 0
        for col in np.arange(0, cols, grid):
            for row in np.arange(0, rows, grid):
                block = filtered[row:row+grid,col:col+grid]
                val =  np.sum(block>0)
                hist.append(val)
        #hist.append(np.sum(filtered>0))
        if show_step:
            plt.imshow(filtered)
            plt.show()
            
    return avg, hist

if __name__ == "__main__":
    im = cv2.imread('cloth.jpg')
    avg, hist = gabor_feat(im)
    print hist
