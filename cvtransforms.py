# coding: utf-8
import random
import cv2
import numpy as np


def HorizontalFlip(batch_img, p=0.5):
    # (T, H, W, C)
    if random.random() > p:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img
    
def FrameRemoval(batch_img, p=0.05):
    # (T, H, W, C)
    for i in range(batch_img.shape[0] - 1):
        if random.random() < p:
            batch_img[i + 1,...] = batch_img[i,...]
    return batch_img
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img
