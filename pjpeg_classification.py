# USAGE
# python3 pjpeg_classification.py --prototxt bvlc_googlenet.prototxt 
# --model bvlc_googlenet.caffemodel --labels synset_words.txt --val_labels val.txt 
# --imagedir test_images

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import glob
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import random
import csv

from prunedJPEG import pjpeg

random.seed(0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=None,
    help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
    help="path to ImageNet labels (i.e., syn-sets)")
ap.add_argument("-v", "--val_labels", required=True,
    help="path to validation labels (i.e., val.txt)")

ap.add_argument("-id", "--imagedir",
    help="Name of the folder containing images to perform detection on. Folder must contain only images.", default=None)

args = vars(ap.parse_args())


IM_DIR = args["imagedir"]
IM_NAME = args["image"]
val_labels = args["val_labels"]


# get path to current working directory
CWD_PATH = os.getcwd()

# define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')
else:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)


# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

with open(val_labels) as inf:
        reader = csv.reader(inf, delimiter=" ")
        first_col = list(zip(*reader))[0]

with open(val_labels) as inf:
        reader = csv.reader(inf, delimiter=" ")
        second_col = list(zip(*reader))[1]

counter = 1
image_number = len(images)
count_found_fjpeg = 0
count_found_pjpeg = np.zeros([image_number, 64])
pjpeg_image_size = np.zeros([image_number, 64])
fjpeg_image_size = []

for (x, image_path) in enumerate(images):
    print("No." + str(counter) + " of total " + str(image_number) + " images is processed")

    # load the input image from disk
    image = cv2.imread(image_path)
    image_name = image_path.split("/")[-1]
    image_label = int(second_col[first_col.index(str(image_name))])

    # full jpeg image classification
    # (104, 117, 123) are the mean values of RGB for the ImageNet training set
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    net.setInput(blob)
    preds = net.forward()
    preds = preds.reshape((1, len(classes)))
    idxs = np.argsort(preds[0])[::-1][:5]

    for (i, idx) in enumerate(idxs):
        if  image_label == idx:
            count_found_fjpeg += 1
            break

    fjpeg_image_size.append(len(open(image_path, 'rb').read()))

    # generate 64 pruned JPEG images    
    pjpeg_images, pjpeg_image_size[x] = pjpeg(image)

    # pruned JPEG images classification
    for n in range(64):
        blob = cv2.dnn.blobFromImage(pjpeg_images[n], 1, (224, 224), (104, 117, 123))
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        net.setInput(blob)
        preds = net.forward()
        preds = preds.reshape((1, len(classes)))
        idxs = np.argsort(preds[0])[::-1][:5]

        for (i, idx) in enumerate(idxs):
            if image_label == idx:
                count_found_pjpeg[x, n] += 1
                break

    counter += 1

# fjpeg: full JPEG, pjpeg: pruned JPEG
prob_fjpeg = np.ones(64) * (count_found_fjpeg / image_number)
prob_pjpeg = np.sum(count_found_pjpeg, axis=0) / image_number
avg_fjpeg_image_size = np.ones(64) * (np.sum(fjpeg_image_size) / image_number)
avg_pjpeg_image_size = np.sum(pjpeg_image_size, axis=0) / image_number

plt.figure(1)
plt.plot(range(64), prob_fjpeg)
plt.plot(range(64), prob_pjpeg)
plt.xlabel('No. of pruned DCT coefficients', size = 15)
plt.ylabel('Classification accuracy', size = 15)
plt.ylim(-0.05, 1.05)
plt.legend(['Full JPEG images', 'Pruned JPEG images'], fontsize = 'large')
plt.grid()

plt.figure(2)
plt.plot(range(64), avg_fjpeg_image_size / 1e3)
plt.plot(range(64), avg_pjpeg_image_size / 1e3)
plt.xlabel('No. of pruned DCT coefficients', size = 15)
plt.ylabel('Size (kB)', size = 15)
plt.legend(['Full JPEG images', 'Pruned JPEG images'], fontsize = 'large')
plt.grid()


# curve fitting
x = avg_pjpeg_image_size / avg_fjpeg_image_size
x = np.insert(x, 0, 0)
y = prob_pjpeg
y = np.insert(y, 0, 0)
prob_fjpeg = np.insert(prob_fjpeg, 0, prob_fjpeg[0])

def func(x, b, c):
    return max(prob_pjpeg)*np.exp((b)*np.exp(c*x))

x_data = np.linspace(0,1,num=100)
popt, pcov = curve_fit(func, x, y)
y_data = func(x_data, *popt)
SSE = np.sum((func(x, *popt)-y) ** 2)
MSE = SSE / len(x)
SST = np.sum((y - np.mean(y)) ** 2)
R_square = 1 - SSE / SST
sigma = np.sqrt(np.sum((y - np.mean(y)) ** 2)/len(y))
# a = max(prob_pjpeg)
print(MSE)


plt.figure(3)
plt.plot(x_data, [prob_fjpeg[0]]*len(x_data), 'b-', label='Full JPEG')
plt.scatter(x,y, label='Experimental results')
plt.plot(x_data, y_data, 'r--', label='Fitted curve')
plt.xlabel('Normalized pruned JPEG size', size = 18)
plt.ylim(-0.05, 1.05)
plt.ylabel('Accuracy', size = 18)
plt.ylim(-0.05, 1.05)
plt.legend(loc=4, fontsize=12)
plt.grid()

plt.show()
