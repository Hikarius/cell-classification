import scipy.io
import numpy as np
import cv2
import os

import random
import pickle

w = "img"
ep=0

for i in range(1, 90):
    way = w + str(i)
    img = cv2.imread((way + "/" + way + ".bmp"), cv2.IMREAD_COLOR)

    epway = (way + "/" + way + "_epithelial.mat")
    mat = scipy.io.loadmat(epway)
    epithelial = (mat.__getitem__("detection"))
    for j in range(len(epithelial)):
        y = int(epithelial[j][0])
        x = int(epithelial[j][1])
        if (x > 10 & y > 10 & x < 489 & y < 489):
            ep += 1
            newep = img[(x - 13):(x + 13), (y - 13):(y + 13)]
            newepway = "cells/epithelial/" + str(ep) + ".bmp"
            cv2.imwrite(newepway, newep)

    fibway = (way + "/" + way + "_fibroblast.mat")
    mat = scipy.io.loadmat(fibway)
    fibroblast = (mat.__getitem__("detection"))
    for j in range(len(fibroblast)):
        y = int(fibroblast[j][0])
        x = int(fibroblast[j][1])
        if (x > 10 & y > 10 & x < 489 & y < 489):
            ep += 1
            newfib = img[(x - 13):(x + 13), (y - 13):(y + 13)]
            newfibway = "cells/fibroblast/" + str(ep) + ".bmp"
            cv2.imwrite(newfibway, newfib)

    infway = (way + "/" + way + "_inflammatory.mat")
    mat = scipy.io.loadmat(infway)
    inflammatory = (mat.__getitem__("detection"))
    for j in range(len(inflammatory)):
        y = int(inflammatory[j][0])
        x = int(inflammatory[j][1])
        if (x > 10 & y > 10 & x < 489 & y < 489):
            ep += 1
            newinf = img[(x - 13):(x + 13), (y - 13):(y + 13)]
            newinfway = "cells/inflammatory/" + str(ep) + ".bmp"
            cv2.imwrite(newinfway, newinf)

    otherway = (way + "/" + way + "_others.mat")
    mat = scipy.io.loadmat(otherway)
    others = (mat.__getitem__("detection"))
    for j in range(len(others)):
        y = int(others[j][0])
        x = int(others[j][1])
        if (x > 10 & y > 10 & x < 489 & y < 489):
            ep += 1
            newother = img[(x - 13):(x + 13), (y - 13):(y + 13)]
            newotherway = "cells/others/" + str(ep) + ".bmp"
            cv2.imwrite(newotherway, newother)
