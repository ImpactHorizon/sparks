import cv2
from functools import reduce
from itertools import product
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
from random import randint

TILE_SIZE = (512, 512)

def accumulated_histogram(image, histogram):
    list(map(lambda x: cv2.calcHist([image], 
                                    [x], 
                                    None, 
                                    [256], 
                                    [0, 256], 
                                    hist=histogram[:,x].reshape(256, 1), 
                                    accumulate=True), 
                range(2)))
    return histogram

def calc_number_of_samples(filename, meat_percentage, sample_size):
    op = openslide.OpenSlide(filename)
    scales = map(lambda x, y: x/y, TILE_SIZE, sample_size)
    return (reduce(lambda x, y: x*y, 
                    map(lambda x, y, z: x/y*z, op.dimensions, 
                            TILE_SIZE, scales)) * meat_percentage)

def calculate_otsu(histData, bins=256):
    hist_sum = np.sum(np.multiply(np.arange(float(bins)), np.ravel(histData)))

    sumB = 0.0
    wB = 0.0
    wF = 0.0
    varMax = 0.0
    threshold = 0
    total = np.sum(histData)

    for t in range(0, bins):
        wB = wB + np.ravel(histData)[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF <= 0:
            break
        sumB = sumB + t * np.ravel(histData)[t]
        mB = sumB / wB
        mF = (hist_sum - sumB) / wF
        varBetween = wB * wF * (mB - mF) * (mB - mF)

        if (varBetween > varMax):
            varMax = varBetween
            threshold = t
    return threshold

def check_coords(handler, xy, thresholds, tval, read_size, mask_handler):
    image = handler.read_region((xy[0], xy[1]), 0, read_size)
    hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    image_size = hsv[:,:,0].size    
    val = (len(np.where((hsv[:,:,0] > thresholds[0]) & 
                        (hsv[:,:,1] > thresholds[1]))[0])) / float(image_size)
    if not val > tval:
        return None
    mask = None
    if mask_handler:
        mask = mask_handler.read_region((xy[0], xy[1]), 0, read_size)
    tumor = check_tumor_value(mask)
    return image, xy[0], xy[1], tumor

def check_tumor_value(mask):    
    if mask:
        numpy_mask = np.array(mask, dtype=np.uint8)
        return np.count_nonzero(numpy_mask[:,:,0])/float(numpy_mask.size/4)
    return 0.0

def concat(image):
    return np.vstack(image)

def consume(queue, event, args, func):
    while not event.is_set():
        func(queue, args)

def get_tile(x, y, handler):
    return (np.array(handler.read_region((x, y), 0, TILE_SIZE), dtype=np.uint8), 
            int(x/TILE_SIZE[0]), 
            int(y/TILE_SIZE[1]))

def init_openslide(filename, maskname):
    handler = openslide.OpenSlide(filename)
    ret = {"handler" : handler}
    if maskname:
        mask_handler = openslide.OpenSlide(maskname)
        ret["mask_handler"] = mask_handler
    else:
        ret["mask_handler"] = None
    return ret

def init_sampler_coords(filename):
    op = openslide.OpenSlide(filename)
    size = op.dimensions
    coords = list(product(range(int(size[0]/TILE_SIZE[0])), 
                            range(int(size[1]/TILE_SIZE[1]))))
    return {"coords" : coords}

def init_sampler_directories(target_dir, class_dirs):
    for class_dir in class_dirs:
        if not os.path.exists(os.path.join(target_dir, class_dir)):
            os.makedirs(os.path.join(target_dir, class_dir))
    return {}

def make_coords(filename, coords):
    handler = openslide.OpenSlide(filename)
    size = handler.dimensions
    for x in range(0, size[0], TILE_SIZE[0]):
        for y in range(0, size[1], TILE_SIZE[1]):
            coords.put({'x': x, 'y': y})

def process_tiles_parallel(filename, parts, images):
    op=openslide.OpenSlide(filename)
    part, total = parts.get()
    size = op.dimensions
    stopy = int(floor((size[1]/total)*(part+1) / 512) * 512)
    starty = int(ceil((size[1]/total)*part / 512) * 512)
    for x in range(0, size[0], TILE_SIZE[0]):
        for y in range(starty, stopy, TILE_SIZE[1]):
            images.put((op.read_region((x, y), 0, TILE_SIZE), x, y))
    parts.task_done()  

def produce(queue, event, args, func):
    while not event.is_set():
        if not queue.full():
            func(queue, args)

def pull_coords(coords, distribution, **kwargs):  
    draw = coords[np.random.choice(np.arange(0, distribution.size), 
                                            p=distribution)]
    xy = list(map(lambda x: draw[x] * 512 + randint(0, 512) - 256, range(2)))
    return xy

def save_histogram_with_otsu(name, histograms, otsu):
    figure, axarr = plt.subplots(3, sharex=True)
    plt.suptitle(name)
    for x, otsu_value in zip(range(3), otsu):
        axarr[x].bar(np.arange(0, 256), 
                        np.log2(np.where(histograms[x] != 0, 
                                            histograms[x], 
                                            1)), 
                        1.0)
        axarr[x].grid(True)
        axarr[x].set_ylabel("log2")
        axarr[x].axvline(x=otsu_value, color="r")

    axarr[0].set_xlim(0, 255)
    axarr[0].set_title('Hue')
    axarr[1].set_title('Saturation')
    axarr[2].set_title('Value')
    return plt

def save_image(image, x, y, tumor, target_dir):
    if tumor == 1.0:
        target_dir = os.path.join(target_dir, "tumor")
    elif tumor > 0.0:
        target_dir = os.path.join(target_dir, "boundaries")
    else:
        target_dir = os.path.join(target_dir, "healthy")
    image.save(os.path.join(target_dir, "%s_%s_%s.jpg" % (x, y, tumor)), 
                format="JPEG", 
                quality=50)
    return (x, y)

def save_thresholds_heatmap(hmap, hist, bins, heatmap_otsu):        
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    figure, axarr = plt.subplots(2)
    axarr[0].bar(center, hist, width=width)
    axarr[0].grid(True)
    axarr[0].set_xlim(0.0, 1.0)
    axarr[0].set_title('Values histogram')
    axarr[0].set_ylabel("Probability %")
    axarr[0].axvline(x=heatmap_otsu, color="r")
    axarr[1].set_title('Values heatmap')
    axarr[1].imshow(hmap, cmap='hot', interpolation='nearest')    
    return plt    

def scan(image, hmap, x, y, thresholds):    
    image_size = image[:,:,0].size    
    val = (len(np.where((image[:,:,0] > thresholds[0]) & 
                        (image[:,:,1] > thresholds[1]))[0])) / float(image_size)
    hmap[x, y] = val    
    return hmap
    

def to_hsv(image, **kwargs):    
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)