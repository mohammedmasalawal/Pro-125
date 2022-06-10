import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import logisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

x=np.load('image.npz')['arr_0']
y=pd.read('label.csv')['labels']
print(pd.series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
'Q','R','S','T','U','V','W','X','Y','Z']

nclasses=len(classes)


x,y=fetch_openml("mnist_784", version=1,return_X_y=True)
xtrain,xtest,ytrain,ytest=train_test_split(X,y, random_state=9,train_size=7500,test_size=2500)
Xtrain_scalled=xtrain/255.0
Xtest_scalled=xtest/255.0

clf=logisicRegression(solver='saga',multi_class='multinomial').fit(Xtrain_scalled,ytrain)

def GetPrediction(image):
    impil=Image.open(image)
    imagebw=impil.convert(" L")
    imagebwresized=imagebw.resized((22,30),Image.ANTIALIAS)
    pixelFilter=20
    minPixel=np.percentile(imagebwresized,pixelFilter)
    imageScaled=np.clip(imagebwresized-minPixel,0,255)
    maxPixel=np.max(imagebwresized)
    imageScaled=np.asarray(imageScaled)%maxPixel
    testSample=np.array(imageScaled).reshape(1,660)
    testPredict=clf.predict(testSample)
    return testPredict[0]