import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io
import random
import pickle

import tensorflow  as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

DIR = "Classification/img98/img98.bmp"

model = load_model('cansız.model')
bidon = pickle.loads(open('ka.bin', "rb").read())

lb = bidon[0]
SIZE = bidon[1]

print("categories", lb);
print("Size:", SIZE);

mat = scipy.io.loadmat("Classification/img98/img98_epithelial.mat")
epithelial = (mat.__getitem__("detection"))
print("ep:" + str(len(epithelial)))

mat = scipy.io.loadmat("Classification/img98/img98_fibroblast.mat")
fibroblast = (mat.__getitem__("detection"))
print("fib:" + str(len(fibroblast)))

mat = scipy.io.loadmat("Classification/img98/img98_inflammatory.mat")
inflammatory = (mat.__getitem__("detection"))
print("inf:" + str(len(inflammatory)))

mat = scipy.io.loadmat("Classification/img98/img98_others.mat")
others = (mat.__getitem__("detection"))
print("others:" + str(len(inflammatory)))

img = cv2.imread(DIR, cv2.IMREAD_COLOR)

x = SIZE
y = SIZE
a = 0
b = 0
c = 0
d = 0
totalpic = 0
rightpic = 0
for i in range(0, 470, int(SIZE / 2)):
    for j in range(0, 470, int(SIZE / 2)):

        img2 = img[i:(i + x), j:(j + y)]
        im_ready = np.array(img2).reshape(-1, SIZE, SIZE, 3)
        preds = model.predict(im_ready)
        res = preds[0][preds.argmax(axis=-1)[0]]


        if (preds.argmax(axis=-1)[0] == 0):
            a += 1
            totalpic += 1
            for ep in epithelial:
                if(ep[0]>j and ep[0]<(j+SIZE) and (ep[1]>i and ep[1]<(i+SIZE))):
                    rightpic += 1
        if (preds.argmax(axis=-1)[0] == 1):
            b += 1
            totalpic += 1
            for fib in fibroblast:
                if(fib[0]>j and fib[0]<(j+SIZE) and (fib[1]>i and fib[1]<(i+SIZE))):
                    rightpic += 1
        if (preds.argmax(axis=-1)[0] == 2):
            c += 1
            totalpic += 1
            for inf in inflammatory:
                if(inf[0]>j and inf[0]<(j+SIZE) and (inf[1]>i and inf[1]<(i+SIZE))):
                    rightpic += 1
        if (preds.argmax(axis=-1)[0] == 3):
            d += 1
            #for ot in others:
             #   if(ot[0]>j and ot[0]<(j+SIZE) and (ot[1]>i and ot[1]<(i+SIZE))):
              #      rightpic += 1
        print(preds.argmax(axis=-1)[0])
        print(res, "  ", i, j)
        label = lb[preds.argmax(axis=-1)[0]]

print(a, b, c, d,rightpic,totalpic,rightpic/totalpic)

cv2.imwrite("new.bmp", img)
