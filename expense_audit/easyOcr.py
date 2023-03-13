import os
import cv2
import easyocr
import numpy as np
from pylab import rcParams
from pdf2image import convert_from_path

# import matplotlib.pyplot as plt


# reader = easyocr.Reader(['en'], gpu=False)

dir_path = os.path.dirname(os.path.realpath(__file__))
path = f"{dir_path}\\storage_imgs\\comprovante_2.png"
# results = reader.readtext(path)


# read the image
img = cv2.imread(path, 0)

# find the white rectangle
th = img.copy()
th[th < 200] = 0

bbox = np.where(th > 0)
y0 = bbox[0].min()
y1 = bbox[0].max()
x0 = bbox[1].min()
x1 = bbox[1].max()

# crop the region of interest (ROI)
img = img[y0:y1, x0:x1]

# histogram equalization
equ = cv2.equalizeHist(img)
# Gaussian blur
blur = cv2.GaussianBlur(equ, (5, 5), 1)

# manual thresholding
th2 = 60  # this threshold might vary!
equ[equ >= th2] = 255
equ[equ < th2] = 0

# Now apply the OCR on the processed image
rcParams["figure.figsize"] = 8, 16
reader = easyocr.Reader(["en"])

output = reader.readtext(equ)

for (bbox, text, prob) in output:
    # prob = round(prob, 2)
    # if prob >= 0.3:
    print(f"[INFO] {prob}: {text}")


""" for (bbox, text, prob) in results:
    # if int(prob) >= 0.3:
    print(f"[INFO] {round(prob, 2)}: {text}") """


# plt.imshow(bbox)
# plt.show()
