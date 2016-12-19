import cv2
from math import ceil, floor
import numpy as np
import openslide

def accumulated_histogram(image, histogram):
    new_histogram = list(map(lambda x: cv2.calcHist([image[:,:,x]], 
                                                [0], 
                                                None, 
                                                [256], 
                                                [0, 256], 
                                                hist=histogram[x], 
                                                accumulate=True), 
                    range(3)))
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

def consume(queue, event, args, func):
    while not event.is_set():
        func(queue, args)

def get_tile(x, y, handler):
    image = handler.read_region((x, y), 0, TILE_SIZE)
    return cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)

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