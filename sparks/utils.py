import cv2
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import openslide

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

def calculate_otsu(histData):
    hist_sum = np.sum(np.multiply(np.arange(256.0), np.ravel(histData)))

    sumB = 0.0
    wB = 0.0
    wF = 0.0
    varMax = 0.0
    threshold = 0
    total = np.sum(histData)

    for t in range(0, 256):
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

def concat(image):
    return np.vstack(image)

def consume(queue, event, args, func):
    while not event.is_set():
        func(queue, args)

def get_tile(x, y, handler):
    return (np.array(handler.read_region((x, y), 0, TILE_SIZE), dtype=np.uint8), 
            x, 
            y)

def init_openslide(filename):
    handler = openslide.OpenSlide(filename)
    return {"handler" : handler}

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

def save_histogram_with_otsu(name, histograms, otsu, filename):
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
    plt.savefig(filename)

def save_pdf_distribution(name, pdfs, filename):
    pass

def scan(image, coords, x, y, thresholds):    
    image_size = image[:,:,0].size    
    val = (len(np.where((image[:,:,0] > thresholds[0]) & 
                        (image[:,:,1] > thresholds[1]))[0])) / float(image_size)
    coords.append((x, y, val))
    return coords
    

def to_hsv(image, **kwargs):    
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)