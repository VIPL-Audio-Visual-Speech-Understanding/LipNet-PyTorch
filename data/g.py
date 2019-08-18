import os
import glob
import random


spks = glob.glob('/ssd/GRID/lip/s*')

random.seed(0)
for spk in spks:
    videos = glob.glob(spk + '/video/mpg_6000/*')
    videos = filter(lambda path: len(os.listdir(path)) == 75, videos)
    videos = list(videos)
    random.shuffle(videos)
    for video in videos[:255]:
        print(video.replace('/ssd/GRID/lip/', ''))