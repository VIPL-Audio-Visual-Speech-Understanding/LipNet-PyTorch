# coding: utf-8
import random
import cv2
import numpy as np


def HorizontalFlip(batch_img):
    # (75, 50, 100, 3)
    if random.random() > 0.5:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def FrameRemoval(batch_img, p = 0.05):
    i = 0
    for j in range(batch_img.shape[0]):
        if(random.random() > p):
            batch_img[i] = batch_img[j]
            i += 1
    batch_img = batch_img[:i]
    return batch_img
    
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img
