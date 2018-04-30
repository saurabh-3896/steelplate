import cv2
import numpy as np
import matplotlib.pyplot as plt
path="histogram\\"
for i in range(6):
    img=cv2.imread(path+str(i)+".bmp",0)
    plt.hist(img.ravel(),256,[0,256])
    plt.xlabel("gray level intensities")
    plt.ylabel("pixel count")
    plt.savefig(path+str(i))

    plt.show()
