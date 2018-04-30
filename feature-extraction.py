##########################              Importing libraries                 #######################################
import numpy as np
import cv2
import os
from skimage.feature import greycomatrix,greycoprops
import pandas as pd

INPUT_SCAN_FOLDER='database\\' # path where the dataset is stored

slices=[]
labels=[]
label=-1

##########################              read images and get labels                 #######################################
for dirName, subdirList, fileList in os.walk(INPUT_SCAN_FOLDER):

        for filename in fileList:
            if ".bmp" in filename.lower():
                slices.append(cv2.imread(os.path.join(dirName, filename),0))
                labels.append(label)
        label=label+1


print("done")

proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
featlist= ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','Label']
properties =np.zeros(5)
glcmMatrix = []
final=[]

##########################              extract GLCM features                 #######################################
for i in range(len(slices)):
    img = slices[i]
    glcmMatrix=(greycomatrix(img, [1], [0], levels=256))    #calculate GLCM matrix

    for j in range(0, len(proList)):
        properties[j]=(greycoprops(glcmMatrix, prop=proList[j]))    #get features


    features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4],labels[i]]) #append features with labels
    final.append(features)

df = pd.DataFrame(final,columns=featlist)
filepath="features.xlsx"  #path where to save the features
df.to_excel(filepath)     #write the extracted features+labels as excel file








