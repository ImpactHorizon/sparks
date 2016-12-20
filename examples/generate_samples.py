from counter import Counter
import cv2
from datetime import datetime
from multiprocessing import Pool, Process, JoinableQueue, Queue, Event
import numpy as np
import openslide
import os
from pdf import MY_PDF
from random import randint, seed, random
import sys
import time
import utils
import scipy.stats as st
from math import floor

CPU_COUNT = 20
TILE_SIZE = (128, 128)
SEED_VAL = 13
TVAL = 0.25

VAL_TRAIN_PROP = 0.3

SUBDIRS = ['train/%s/healthy', 'train/%s/boundaries', 'train/%s/tumor',
            'validation/%s/healthy', 'validation/%s/boundaries', 
            'validation/%s/tumor']

def scan(images, args):
    thresholds = args[0]
    try:
        image, x, y = images.get(timeout=0.3)    
    except:
        return    
    hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    image_size = hsv[:,:,0].size    
    val = (len(np.where((hsv[:,:,0] > thresholds[0]) & 
                        (hsv[:,:,1] > thresholds[1]))[0])) / float(image_size)
    if val > TVAL:
        xvals = args[1].get()
        xvals.append(x)        
        args[1].put(xvals)        
        yvals = args[2].get()        
        yvals.append(y)
        args[2].put(yvals)
        args[3].increment()
    images.task_done()

def sample_get(candidates, event, args):
    size = args[0]
    op_meat = openslide.OpenSlide(args[1])  
    op_mask = None
    xpdf, ypdf = args[3]    
    if args[2]:
        op_mask = openslide.OpenSlide(args[2])
    indexes = np.arange(0, 100)
    while not event.is_set():
        x = int(floor(np.random.choice(indexes, p=xpdf.buckets) / 
                    float(100)*size[0] + (randint(0, 512) - 256)))
        y = int(floor(np.random.choice(indexes, p=ypdf.buckets) / 
                    float(100)*size[1] + (randint(0, 512) - 256)))
        if op_mask:
            mask = op_mask.read_region((x, y), 0, TILE_SIZE)
        else:
            mask = None      
        try:      
            candidates.put((op_meat.read_region((x, y), 0, TILE_SIZE), 
                            mask,
                            x, 
                            y,), timeout=0.3)
        except:
            continue
    #print 'done get'

def check_in_mask(mask):    
    if mask:
        numpy_mask = np.array(mask, dtype=np.uint8)
        return np.count_nonzero(numpy_mask[:,:,0])/float(numpy_mask.size/4)
    return 0.0

def check_candidate(candidates, args):
    samples = args[0]
    thresholds = args[1]
    target_dir = args[2]
    name = args[3]
    try:
        image, mask, x, y = candidates.get(timeout=0.3)
    except Queue.Empty:
        #print 'timeout get candidate'        
        pass
    else:
        tumor = check_in_mask(mask)
        hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
        image_size = hsv[:,:,0].size 
        val = (len(np.where((hsv[:,:,0] > thresholds[0]) & 
                            (hsv[:,:,1] > thresholds[1]))[0])) / float(image_size)
        if val > TVAL:
            if random() < VAL_TRAIN_PROP:
                target_dir = os.path.join(target_dir, "validation")
            else:
                target_dir = os.path.join(target_dir, "train")
            target_dir = os.path.join(target_dir, name)
            if (tumor > 0.0) and (tumor < 1.0):
                target_dir = os.path.join(target_dir, "boundaries")
            elif tumor == 0.0:
                target_dir = os.path.join(target_dir, "healthy")
            else:
                target_dir = os.path.join(target_dir, "tumor")        
            image.save(os.path.join(target_dir, "%s_%s_%s.jpg" % (x, y, tumor)), 
                        format="JPEG", 
                        quality=50)
            try:
                samples.put("DONE", timeout=0.3)
            except:
                #print 'samples full'            
                pass
        candidates.task_done()

if __name__ == '__main__': 
    args = sys.argv[1:]
    if len(args) == 0:
        raise AttributeError("Missing filename.")
    if len(args) < 2:
        raise AttributeError("Missing thresholds.")    

    FILE = args[0]
    THRESH = args[1]
    tumor = True

    file_type = os.path.basename(THRESH).split("_")[1]
    name = "_".join(os.path.basename(THRESH).split(".")[0].split("_")[1:])

    if file_type == "Normal":
        tumor = False
   
    MASK = None
    TARGET = None
    if tumor:
        if len(args) < 3:
            raise AttributeError("Missing mask.")
        MASK = args[2]    

        if len(args) == 4:
            TARGET = args[3]
    else:
        if len(args) == 3:
            TARGET = args[2]    

    for sub in SUBDIRS:
        if not os.path.exists(os.path.join(TARGET, sub % name)):
            os.makedirs(os.path.join(TARGET, sub % name))
    
    seed(SEED_VAL)
    op = openslide.OpenSlide(FILE)
    size = op.dimensions    
    
    Xvalues = Queue(1)
    Yvalues = Queue(1)
    counter = Counter(0)
    Xvalues.put([])
    Yvalues.put([])

    start = datetime.now()

    with open(THRESH, "r") as file_handle:
        line = file_handle.read()
        otsu = map(lambda x: int(x), line.split(' '))
    
    parts = JoinableQueue(CPU_COUNT)
    images = JoinableQueue(1024)

    map(lambda i: parts.put((i, CPU_COUNT)), xrange(0, CPU_COUNT))
    producers = map(lambda x: Process(target=utils.process_tiles_parallel, 
                                        args=(FILE, parts, images)), 
                    xrange(0, CPU_COUNT))
    map(lambda proc: proc.start(), producers)
    consumers_events = map(lambda x: Event(), xrange(0, CPU_COUNT))
    consumers = map(lambda x: Process(target=utils.consume, 
                                        args=(images,
                                                consumers_events[x], 
                                                (otsu, 
                                                    Xvalues, 
                                                    Yvalues, 
                                                    counter), 
                                                scan)),
                    xrange(0, CPU_COUNT))
    map(lambda proc: proc.start(), consumers)

    parts.join()
    images.join()
    map(lambda event: event.set(), consumers_events)
    
    print counter.value()
    xvals = Xvalues.get()
    yvals = Yvalues.get()
    cval = counter.value()     
    area = cval/float(size[0]*size[1]/512.0/512.0)
    number_of_samples = int(size[0] * size[1] / (TILE_SIZE[0] * TILE_SIZE[1])
                            * area)
    print number_of_samples, area, size
    candidates = JoinableQueue(CPU_COUNT)    
    samples = JoinableQueue(number_of_samples)

    Xpdf = MY_PDF(a=0, b=size[0])
    Xpdf.feed(xvals)
    Ypdf = MY_PDF(a=0, b=size[1])
    Ypdf.feed(yvals)

    consumers_events = map(lambda x: Event(), xrange(0, CPU_COUNT))
    consumers = map(lambda x: Process(target=utils.consume, 
                                        args=(candidates,
                                                consumers_events[x], 
                                                (samples, otsu, TARGET, name), 
                                                check_candidate)),
                    xrange(0, CPU_COUNT))
    for p in consumers:
        p.daemon = True
    map(lambda proc: proc.start(), consumers)

    producers_events = map(lambda x: Event(), xrange(0, CPU_COUNT))
    producers = map(lambda x: Process(target=sample_get, 
                                        args=(candidates,
                                                producers_events[x], 
                                                (size, 
                                                    FILE, 
                                                    MASK, 
                                                    (Xpdf, Ypdf)))),
                    xrange(0, CPU_COUNT))
    for p in producers:
        p.daemon = True
    map(lambda proc: proc.start(), producers)

    while not samples.full():
        if samples.qsize() > 0:
            print samples.qsize()
        time.sleep(10)           
  
    map(lambda event: event.set(), producers_events)
    candidates.join()
    map(lambda event: event.set(), consumers_events)
    time.sleep(1)
    end = datetime.now()
    print end-start