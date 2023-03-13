import cv2
import numpy as np


def improve_search_accuracy(img):
    resize = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    white = np.where(thresh == 255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    crop = gray[ymin:ymax, xmin:xmax]
    hh, ww = crop.shape
    thresh2 = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1.1)
    kernel = np.ones((1, 7), np.uint8)
    morph = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    morph = 255 - morph
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)


    bbox = img.copy()
    cv2.rectangle(bbox, (x + xmin, y + ymin), (x + xmin + w, y + ymin + h), (0, 0, 255), 1)

    # test if contour touches sides of image
    # if x == 0 or y == 0 or x + w == ww or y + h == hh:
    #     print('region touches the sides')
    # else:
    #     print('region does not touch the sides')

    # plt.imshow(bbox)
    # plt.show()
    return bbox
