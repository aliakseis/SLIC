import cv2

import os

from skimage.segmentation import slic
from skimage.util import img_as_float

import numpy as np

from keras.models import load_model

from keras import backend as K

#import matplotlib.pyplot as plt

import time

import concurrent.futures

import sys


sys.stderr = LoggerStream()

# input image dimensions
img_rows, img_cols = 96, 96

model = load_model(modelPath)

def DetectRunway(image):
    
    t1 = time.perf_counter()
    
    segments = slic(img_as_float(image), n_segments = 200, sigma = 5)

    #all_x = []

    unique_segments = np.unique(segments)
    n_superpixels = unique_segments.shape[0]

    t2 = time.perf_counter()
    print(f"Executed slic() in {t2 - t1:0.4f} seconds")

    all_x = [None] * n_superpixels


    def handle_segment(i):
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        #mask[labels_out == i] = 255
        mask[segments == unique_segments[i]] = 255
        
        m = cv2.moments(mask, True);

        x = m["m10"]/m["m00"]
        y = m["m01"]/m["m00"]

        img = cv2.bitwise_and(image, image, mask = mask)

        #crop
        img = img[int(y - img_rows/2) : int(y + img_rows/2), int(x - img_cols/2) : int(x + img_cols/2)]

        old_size = img.shape[:2]
        delta_w = img_cols - old_size[1]#img.cols
        delta_h = img_rows - old_size[0]#img.rows
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)



    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Start the load operations and mark each future with its URL
        future_to_i = {executor.submit(handle_segment, i): i for i in range(n_superpixels)}
        for future in concurrent.futures.as_completed(future_to_i):
            i = future_to_i[future]
            try:
                all_x[i] = [future.result()]
            except Exception as exc:
                print('%d generated an exception: %s' % (i, exc))
            #else:
            #    print('%r page is %d bytes' % (url, len(data)))
    
    all_x = np.concatenate(all_x)

    t25 = time.perf_counter()
    print(f"Handled segments in {t25 - t2:0.4f} seconds")
    
    if K.image_data_format() == 'channels_first':
        all_x = all_x.reshape(all_x.shape[0], 3, img_rows, img_cols)
    else:
        all_x = all_x.reshape(all_x.shape[0], img_rows, img_cols, 3)

    all_x = all_x.astype('float32')
    all_x /= 255

    t3 = time.perf_counter()
    print(f"Reshaped segments in {t3 - t25:0.4f} seconds")
    
    ynew = model.predict(all_x)

    t4 = time.perf_counter()
    print(f"Predicted classes in {t4 - t3:0.4f} seconds")
    
    mask = np.zeros(np.concatenate((image.shape[:2], [2])), dtype = "float32")

    for (i, segVal) in enumerate(unique_segments):
        #if ynew[i] == 0:
        #    continue
        #mask[labels_out == i] = 255
        mask[segments == segVal] = ynew[i]

    return mask
