import cv2
from functools import reduce
from itertools import product
from math import ceil, floor, log1p
import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
from random import randint
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema

TILE_SIZE = (512, 512)

def accumulated_histogram(image, histogram):
    hist = list(map(lambda x: cv2.calcHist([image], 
                                    [x], 
                                    None, 
                                    [256], 
                                    [0, 256]), 
                    range(3)))
    for x in range(3):
        histogram[x] = np.add(hist[x].ravel(), histogram[x].ravel())                                
    return histogram

def basename(filename):
    return os.path.basename(filename).split(".")[0]

def calc_number_of_samples(filename, meat_percentage, sample_size):
    op = openslide.OpenSlide(filename)
    scales = map(lambda x, y: x/y, TILE_SIZE, sample_size)
    return int((reduce(lambda x, y: x*y, 
                    map(lambda x, y, z: x/y*z, op.dimensions, 
                            TILE_SIZE, scales)) * meat_percentage))

def calc_size_in_tiles(filename, tile_size=TILE_SIZE):
    op = openslide.OpenSlide(filename)
    return list(map(lambda x, y: int(x/y), op.dimensions, tile_size))

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
    max_hue = max(thresholds[0])
    min_hue = min(thresholds[0])
    val = (len(np.where(((hsv[:,:,0] > max_hue) | (hsv[:,:,0] < min_hue)) &
                        (hsv[:,:,1] > thresholds[1]) & 
                        (hsv[:,:,2] > 50))[0])) / float(image_size)
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

def circular_otsu(hist, init_val):
    k1 = [init_val]
    k2 = []
    i = 0
    while True:
        rolled = np.roll(hist, -k1[-1])        
        k2.append(calculate_otsu(rolled, hist.size))
        rolled = np.roll(hist, -k2[-1])
        k1.append(calculate_otsu(rolled, hist.size))
        if i > 1:            
            if (k1[-1] == k1[-2] and k2[-1] == k2[-2]):
                return (k1[-1], k2[-1])
        i += 1
    print(k1, k2)

def concat(image):
    return np.vstack(image)

def consume(queue, event, args, func):
    while not event.is_set():
        func(queue, args)

def detect_peaks(hist, count=2):
    hist_copy = hist    
    peaks = len(argrelextrema(hist_copy, np.greater, mode="wrap")[0])
    sigma = log1p(peaks)
    print(peaks, sigma)
    while (peaks > count):
        new_hist = gaussian_filter(hist_copy, sigma=sigma)        
        peaks = len(argrelextrema(new_hist, np.greater, mode="wrap")[0])
        if peaks < count:
            peaks = count + 1
            sigma = sigma * 0.5
            continue
        hist_copy = new_hist
        sigma = log1p(peaks)
    print(peaks, sigma)
    return argrelextrema(hist_copy, np.greater, mode="wrap")[0]

def get_mini(filename):
    handler = openslide.OpenSlide(filename)
    size = handler.level_dimensions[-1]
    return np.array(handler.read_region((0, 0), handler.level_count-1, size),
                    dtype=np.uint8)


def get_tile(x, y, handler, tile_size=TILE_SIZE, **kwargs):
    return (np.array(handler.read_region((x, y), 0, tile_size), dtype=np.uint8), 
            int(x/tile_size[0]), 
            int(y/tile_size[1]))

def init_openslide(filename, maskname=None):
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
        try:
            make_directory(os.path.join(target_dir, class_dir))
        except:
            continue
    return {}

def make_coords(filename, coords):
    handler = openslide.OpenSlide(filename)
    size = handler.dimensions
    for x in range(0, size[0], TILE_SIZE[0]):
        for y in range(0, size[1], TILE_SIZE[1]):
            coords.put({'x': x, 'y': y})

def make_directory(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir

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

def samples_heatmap(x, y, samples_map, read_size):
    samples_map[int(x/read_size[0]), int(y/read_size[1])] += 1
    return samples_map

def save_heatmap(heatmap, mask):
    plt.clf()
    xmin, xmax, ymin, ymax = 0, heatmap.shape[1], heatmap.shape[0], 0
    extent = xmin, xmax, ymin, ymax
    alpha=1.0
    if mask is not None:
        alpha=0.5
        xmin, xmax, ymin, ymax = (0, max(heatmap.shape[1], mask.shape[1]), 
                                    max(heatmap.shape[0], mask.shape[0]), 0)
        extent = xmin, xmax, ymin, ymax
        plt.imshow(mask, extent=extent)
        plt.hold(True)
    plt.suptitle("Heatmap of sampled tiles.")
    plt.imshow(heatmap, cmap='gnuplot', interpolation='nearest', extent=extent,
                alpha=alpha)
    return plt

def save_histogram_with_otsu(name, histograms, otsu):
    plt.clf()
    figure, axarr = plt.subplots(3)
    figure.tight_layout()
    for x, otsu_value in zip(range(3), otsu):
        axarr[x].bar(np.arange(0, histograms[x].size), 
                        np.log2(np.where(histograms[x] >= 1, 
                                            histograms[x], 
                                            1)), 
                        1.0)
        axarr[x].grid(True)
        axarr[x].set_ylabel("log2")
        for val in otsu_value:
            axarr[x].axvline(x=val, color="r")
        axarr[x].set_xlim(0, histograms[x].size)
    
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

def save_thresholds_heatmap(hmap, hist, bins, heatmap_otsu, mini): 
    plt.clf()       
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    ax = plt.subplot(1, 2, 1)
    ax.bar(center, hist, width=width)
    ax.grid(True)
    ax.set_xlim(0.0, 1.0)
    ax.set_title('Values histogram')
    ax.set_ylabel("Percent of samples (%)")
    ax.axvline(x=heatmap_otsu, color="r")  
    ax = plt.subplot(2, 2, 2)    
    ax.set_title('Values heatmap')
    ax.imshow(hmap, cmap='hot', interpolation='nearest')
    ax = plt.subplot(2, 2, 4)
    ax.set_title("Original mini")
    ax.imshow(mini)
    return plt    

def scan(image, x, y, thresholds):    
    image_size = image[:,:,0].size   
    max_hue = max(thresholds[0])
    min_hue = min(thresholds[0])    
    val = (len(np.where(((image[:,:,0] > max_hue) | 
                        (image[:,:,0] < min_hue)) &
                        (image[:,:,1] > thresholds[1]) &
                        (image[:,:,2] > 50))[0])) / float(image_size)  
    return (x, y, val)

def set_hmap(hmap, x, y, val):
    hmap[x, y] = val
    return hmap

def to_hsv(image, **kwargs):    
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)