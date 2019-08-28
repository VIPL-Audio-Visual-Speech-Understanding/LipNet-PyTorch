import cv2
import json
import numpy as np
from multiprocessing import Pool, Process, Queue
import time
import os


def get_position(size, padding=0.25):
    
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                    0.553364, 0.490127, 0.42689]
    
    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                    0.784792, 0.824182, 0.831803, 0.824182]
    
    x, y = np.array(x), np.array(y)
    
    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))

def cal_area(anno):
    return (anno[:,0].max() - anno[:,0].min()) * (anno[:,1].max() - anno[:,1].min()) 


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
 
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
 
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])



    
def anno_img(img_dir, anno_dir, save_dir):

    files = list(os.listdir(img_dir))
    files = [file for file in files if(file.find('.jpg') != -1)]
    shapes = []
    for file in files:
        img = os.path.join(img_dir, file)
        anno = os.path.join(anno_dir, file).replace('.jpg', '.txt')
           
        I = cv2.imread(img)
        count = 0
    
        with open(anno, 'r') as f:
            annos = [line.strip().split('\t') for line in f.readlines()]
            if(len(annos) == 0): return
            for (i, anno) in enumerate(annos):
                x, y = [], []
                for p in anno:
                    _, __ = p[1:-1].split(',')
                    _, __ = float(_), float(__)
                    x.append(_)
                    y.append(__)   
                annos[i] = np.stack([x, y], 1)
                
        anno = sorted(annos, key = cal_area, reverse=True)[0]        
        shape = []
        
        shapes.append(anno[17:])
        
    
    front256 = get_position(256)
    M_prev = None
    for (shape, file) in zip(shapes, files):
        img = os.path.join(img_dir, file)
        I = cv2.imread(img)
        M = transformation_from_points(np.matrix(shape), np.matrix(front256))
        img = cv2.warpAffine(I, M[:2], (256, 256))
        (x, y) = front256[-20:].mean(0).astype(np.int32)
        w = 160//2
        img = img[y-w//2:y+w//2,x-w:x+w,...]
        cv2.imwrite(os.path.join(save_dir, file), img)


def run(files):
    tic = time.time()
    count = 0
    print('n_files:{}'.format(len(files)))
    for (img_dir, anno_dir, save_dir) in files:
        anno_img(img_dir, anno_dir, save_dir)
        count += 1
        if(count % 1000 == 0):
            print('eta={}'.format((time.time()-tic)/(count) * (len(files) - count) / 3600.0))

if(__name__ == '__main__'):
    with open('grid.txt', 'r') as f:
        data = [line.strip() for line in f.readlines()]
        data = list(set([os.path.split(file)[0] for file in data]))

    
    annos = [name.replace('GRID/6k_video_imgs', 'GRID/landmarks') for name in data]  
    targets = [name.replace('GRID/6k_video_imgs', 'GRID/lip') for name in data]  
    
    for dst in targets:
        if(not os.path.exists(dst)):
            os.makedirs(dst)
    
    data = list(zip(data, annos, targets))
    processes = []
    n_p = 8
    bs = len(data) // n_p
    for i in range(n_p):
        if(i == n_p - 1):
            bs = len(data)
        p = Process(target=run, args=(data[:bs],))
        data = data[bs:]
        p.start()
        processes.append(p)
        
    assert(len(data) == 0)
    for p in processes:
        p.join()
