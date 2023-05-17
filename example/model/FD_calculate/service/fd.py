from sklearn.linear_model import LinearRegression
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2
import sys
from collections import Counter
import math
from PIL import Image
from scipy.stats import linregress
from sklearn import metrics
import pyomo
from pyomo.environ import ConcreteModel, Var, Binary, Objective, ConstraintList, minimize, SolverFactory
from pyomo.environ import *

# solver_path = 'D:\\cbc\\bin\\cbc.exe'
# solver = SolverFactory('cbc', executable=solver_path)

img = np.array(mpimg.imread(os.path.join(
    "C:/Users/zhao xianmin/Desktop/way/output.png")))


# 03 convert image into gray scale


def rgbgray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


gray_img = rgbgray(np.copy(img))
# 04 plot a histrogram to pick a threshold

t = 0.4
maximum = 1
cv2.threshold(gray_img, t, maximum, cv2.THRESH_BINARY)
len(cv2.threshold(gray_img, t, maximum, cv2.THRESH_BINARY)) == 2
ret, thresh = cv2.threshold(gray_img, t, maximum, cv2.THRESH_BINARY)
# 05 compare input colorful and output binary images

# 06 remove 4 sides
ima = np.copy(thresh)
xu = ima.shape[1]//2
Yu = 0
for yu in range(ima.shape[0]):
    if ima[yu, xu] > 0:
        Yu = yu
        break
for yu in range(Yu-18, Yu+58):  # down&UP
    for xu in range(ima.shape[1]):
        ima[yu, xu] = 0

imb = np.copy(ima)
yl = imb.shape[0]//2
Xl = 0
for xl in range(imb.shape[1]):
    if imb[yl, xl] > 0:
        Xl = xl
        break
for xl in range(Xl-20, Xl+40):  # right&LEFT
    for yl in range(imb.shape[0]):
        imb[yl, xl] = 0

imc = np.copy(imb)
xb = imc.shape[1]//2
Yb = imc.shape[0] - 1
for yb in range(imc.shape[0]-1, 0-1, -1):
    if imc[yb, xb] > 0:
        Yb = yb
        break
for yb in range(Yb-5, Yb+5):  # down
    for xb in range(imc.shape[1]):
        imc[yb, xb] = 0

imd = np.copy(imc)
yr = imd.shape[0]//2
Xr = imd.shape[1] - 1
for xr in range(imd.shape[1]-1, 0-1, -1):
    if imd[yr, xr] > 0:
        Xr = xr
        break
for xr in range(Xr-5, Xr+5):
    for yr in range(imd.shape[0]):
        imd[yr, xr] = 0

# 07 extract perimeters of each region


def boundary(img):
    im = np.copy(img)
    for y in range(img.shape[0]-1):
        for x in range(img.shape[1]-1):
            # 255 white
            # 128 gray
            # 0 black
            if im[y, x] > 0:  # not black
                # if (y == 0) or (x ==0) or (y == img.shape[0]-1) or (x == img.shape[1]-1): #on the square boundary
                # im[y,x] = 255 # white
                # surrounded by nonblack pixels just like itself
                if (im[y+1, x] > 0 and im[y-1, x] > 0 and im[y, x-1] > 0 and im[y, x+1] > 0):
                    im[y, x] = 128  # gray
                else:  # surrounded by at least one black pixel
                    im[y, x] = 255  # white
    return im


im = boundary(thresh)
ime = boundary(imd)

# 08 remove interiors
arr = np.copy(ime)
for y in range(arr.shape[0]):
    for x in range(arr.shape[1]):
        if arr[y, x] == 128:
            arr[y, x] = 0

# 09 set different labels to bright pixels


def setlabel(y, x, current_label, img, labels):
    h = img.shape[0]
    w = img.shape[1]
    if y < 1 or y > h - 1 or x < 1 or x > w - 1:
        return  # function ends
    labels[y, x] = current_label
    if y < h-1 and img[y+1, x] > 0 and labels[y+1, x] == 0:
        setlabel(y+1, x, current_label, img, labels)
    if y < h-1 and img[y-1, x] > 0 and labels[y-1, x] == 0:
        setlabel(y-1, x, current_label, img, labels)
    if x < w-1 and img[y, x+1] > 0 and labels[y, x+1] == 0:
        setlabel(y, x+1, current_label, img, labels)
    if x < w-1 and img[y, x-1] > 0 and labels[y, x-1] == 0:
        setlabel(y, x-1, current_label, img, labels)
    if img[y+1, x+1] > 0 and labels[y+1, x+1] == 0:
        setlabel(y+1, x+1, current_label, img, labels)
    if img[y+1, x-1] > 0 and labels[y+1, x-1] == 0:
        setlabel(y+1, x-1, current_label, img, labels)
    if img[y-1, x+1] > 0 and labels[y-1, x+1] == 0:
        setlabel(y-1, x+1, current_label, img, labels)
    if img[y-1, x-1] > 0 and labels[y-1, x-1] == 0:
        setlabel(y-1, x-1, current_label, img, labels)


def cclabel(img):
    current_label = 1
    labels = np.zeros(shape=img.shape)
    for y in range(2, img.shape[0]-2):
        for x in range(2, img.shape[1]-2):
            # pixel is not black and not re-labelled to 1,2,3,....
            if img[y, x] > 0 and labels[y, x] == 0:
                setlabel(y, x, current_label, img, labels)
                current_label += 1
                # print(current_label)
    return labels


im2 = cclabel(arr)  # arr = ime = boundary(imd)

# 10 Count pixels
# Counter() => count of each element
cc_p = Counter()

# copy the image to arr2
arr2 = np.copy(im2)  # labeled
# find the sizes of different connected components
for y in range(arr2.shape[0]):
    for x in range(arr2.shape[1]):
        #
        if arr2[y, x] in cc_p.keys():  # if a pixel
            cc_p[arr2[y, x]] += 1
        #
        else:
            cc_p[arr2[y, x]] = 1
num = len(cc_p) - 1
# print(num)

# remove 0, the background label
cc_p.pop(0, None)
# loop through the labels and increase their intensities
cc_labels = [key for key in cc_p.keys()]
for y in range(arr2.shape[0]):
    for x in range(arr2.shape[1]):
        if arr2[y, x] in cc_labels:
            arr2[y, x] += (255-num)

# loop through the image again and increase the contrast between neighboring components
for y in range(arr2.shape[0]):
    for x in range(arr2.shape[1]):
        if arr2[y, x] % 2 == 0:
            arr2[y, x] /= 2
# plt.subplots(1, 5, figsize=(20, 20))
# plt.subplot(1, 5, 1), plt.imshow(imd, "gray"), plt.title(
#     "Square boundaries removed"), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 5, 2), plt.imshow(ime, "gray"), plt.title(
#     "Perimeters extracted"), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 5, 3), plt.imshow(arr, "gray"), plt.title(
#     "Interiors removed"), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 5, 4), plt.imshow(im2, "gray"), plt.title(
#     "Perimeters labeled"), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 5, 5), plt.imshow(arr2, "gray"), plt.title(
#     "Pixels counted"), plt.xticks([]), plt.yticks([])

# 11 Determine top 3 perimeters
cc_p
cc_p.most_common(3)
# pick count index 35, 12, and 26 with largest perimeters
data_list = list(arr2.flatten())
counts = Counter(data_list)
counts.most_common(4)[1]  # 3 top most common
# Keep largest Domain
most_common = counts.most_common(3+1)
extract_largest_domain1 = np.copy(arr2)
for y in range(arr2.shape[0]-1):
    for x in range(arr2.shape[1]-1):
        if extract_largest_domain1[y, x] != most_common[1][0]:
            extract_largest_domain1[y, x] = 0

extract_largest_domain2 = np.copy(arr2)
for y in range(arr2.shape[0]):
    for x in range(arr2.shape[1]-1):
        if extract_largest_domain2[y, x] != most_common[2][0]:
            extract_largest_domain2[y, x] = 0

extract_largest_domain3 = np.copy(arr2)
for y in range(arr2.shape[0]-1):
    for x in range(arr2.shape[1]-1):
        if extract_largest_domain3[y, x] != most_common[3][0]:
            extract_largest_domain3[y, x] = 0

# Box Counting Heuristic 1


def opt_fun(boxsizes):

    boxcounts = np.array(boxsizes.shape)
    boxcounts = []
    counts = []
    px = []
    py = []
    for k in boxsizes:
        model = ConcreteModel()
        model.Y = Var(range(165), range(165), within=Binary, initialize=0)
        model.obj = Objective(
            expr=sum(model.Y[i, j]for i in range(165) for j in range(165)), sense=minimize)
        model.c = ConstraintList()
        for i in range(ylb, yub+1):
            for j in range(xlb, xub+1):
                if extract_largest_domain1[i, j] > 0:
                    model.c.add(sum(model.Y[m, n] for m in range(
                        i-k+1, i+1) for n in range(j-k+1, j+1)) >= 1)
        solver = SolverFactory('cbc')
        solver.solve(model)
        #m = k//2 - 1
        #boxcounts[m] = model.obj()
        obj = model.obj()
        #print("Box Size :", k, "Objective :", model.obj())
        counts.append(obj)
        indexref = 0
        jlist = []
        ilist = []
        for i in range(165):
            for j in range(165):
                # if model.Y[i, j] == 1:
                #print("Box Position",i,j)
                jlist.append(j)
                ilist.append(i)
        px.append(jlist)
        py.append(ilist)
    return px, py, counts


def make_plots(px, py, boxsizes):
    for i in range(len(px)):
        im = np.array(
            extract_largest_domain1[ylb-1:yub+2, xlb-1:xub+2], dtype=np.uint8)
        fig, ax = plt.subplots(1)
        ax.imshow(im, "gray")
        for j in range(len(px[i])):
            square = patches.Rectangle((px[i][j]-(xlb-1)-0.5, py[i][j]-(
                ylb-1)-0.5), boxsizes[i], boxsizes[i], linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(square)
        plt.show()


def plot_FD(boxsize, boxcounts):
    x = np.log10(boxsize)
    y = np.log10(boxcounts)

    # print("log(boxsize): ", x)
    # print("log(boxcounts): ", y)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = slope*np.asarray(x) + intercept
    mse = metrics.mean_squared_error(y, y_pred)
    csfont = {'fontname': 'Times New Roman'}
    plt.title("FD= {:.2f} with MSE ={:.2E}".format(
        abs(slope), mse), **csfont, fontsize=18)
    plt.xlabel('log(box size)', fontsize=16)
    plt.ylabel('log(box count)', fontsize=16)
    plt.scatter(x, y)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(x, y_pred, "r")
    plt.savefig('ConfM1.png', dpi=1200, bbox_inches='tight')


def evaluate(boxsizes):
    px, py, counts = opt_fun(boxsizes)
    # print("box sizes", boxsizes)
    # print("box counts", counts)
    # make_plots(px, py, boxsizes)
    plot_FD(boxsizes, counts)


def evaluate1(boxsizes):
    px, py, counts = opt_fun(boxsizes)
    # print("box sizes", boxsizes)
    # print("box counts", counts)
    plot_FD(boxsizes, counts)


def boxcountn(img, n):
    count = 0
    for y in range((arr2.shape[0])//n):
        for x in range((arr2.shape[1])//n):
            current_count = 0
            for i in range(n):
                for j in range(n):
                    if img[y*n+i, x*n+j] > 0:
                        current_count += 1
            if current_count != 0:
                count += 1
                # break
    return count


average = np.array([0, 0, 0, 0])
logboxcounts = np.array([0.001, 0.001, 0.001, 0.001])
for i in range(1, 5):
    n = 2*i
    average[i-1] = (boxcountn(extract_largest_domain1, 2*i)+boxcountn(
        extract_largest_domain2, 2*i)+boxcountn(extract_largest_domain3, 2*i))/3
    #print("box size is:", 2*i, "average box count is:", average[i-1])
    #print("log box size is:",  math.log(2*i,10), "log average box count is", math.log(average[i-1],10))
    logboxcounts[i-1] = math.log(average[i-1], 10)
# print(average)
# print(boxcountn(extract_largest_domain3, 2))
x = np.array([0.301029996, 0.602059991, 0.77815125,
              0.903089987]).reshape((-1, 1))
y = logboxcounts
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
# print(y, 'slope:', model.coef_)
# D = print(-model.coef_)
# D

y1 = 0  # top left
x1 = 0
for y in range(arr2.shape[0]-1):
    for x in range(arr2.shape[1]-1):
        if extract_largest_domain1[y, x] != 0:
            y1 = y

for x in range(arr2.shape[1]-1):
    for y in range(arr2.shape[0]-1):
        if extract_largest_domain1[y, x] != 0:
            x1 = x
xub = x1
yub = y1
y4 = 0
x4 = 0
ynew = [y for y in range(arr2.shape[0]-1)]
ynew.reverse()  # Bottom-up approach
xnew = [x for x in range(arr2.shape[1]-1)]
xnew.reverse()  # Bottom-up approach
for y in ynew:
    for x in xnew:
        if extract_largest_domain1[y, x] != 0:
            y4 = y
for x in xnew:
    for y in ynew:
        if extract_largest_domain1[y, x] != 0:
            x4 = x
xlb = x4
ylb = y4
intrest = np.copy(extract_largest_domain1)

for yi in range(y4, y1+1):
    for xj in range(x4, x1+1):
        intrest[yi, xj] = 5


comb1 = np.array([2, 4, 6, 8, 10])

evaluate(comb1)


def out():
    comb1 = np.array([2, 4, 6, 8, 10])

    evaluate(comb1)
    # return 10
