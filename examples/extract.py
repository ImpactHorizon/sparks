import cv2
from datetime import datetime
from multiprocessing import Process, JoinableQueue, Queue
import numpy as np
import os
import sys
import utils

CPU_COUNT = 4
TVAL = 0.3
TARGET = "./extracted"

def extract(images, args):
    thresholds = args[0]
    target_dir = args[1]
    image, x, y = images.get()    
    hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    image_size = hsv[:,:,0].size    
    val = (len(np.where((hsv[:,:,0] > thresholds[0]) & 
                        (hsv[:,:,1] > thresholds[1]))[0])) / float(image_size)
    if val > TVAL:
        image.save(os.path.join(target_dir, "%s_%s.jpg" % (x, y)), 
                    format="JPEG", 
                    quality=50)
    images.task_done()

if __name__ == '__main__': 
    args = sys.argv[1:]
    if len(args) == 0:
        raise AttributeError("Missing filename.")
    if len(args) < 2:
        raise AttributeError("Missing thresholds.")

    FILE = args[0]
    THRESH = args[1]

    if len(args) == 3:
        TARGET = args[2]

    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    parts = JoinableQueue(CPU_COUNT)
    images = JoinableQueue(1024)

    start = datetime.now()

    with open(THRESH, "r") as file_handle:
        line = file_handle.read()
        otsu = map(lambda x: int(x), line.split(' '))

    map(lambda i: parts.put((i, CPU_COUNT)), xrange(0, CPU_COUNT))
    producers = map(lambda x: Process(target=utils.process_tiles_parallel, 
                                        args=(FILE, parts, images)), 
                    xrange(0, CPU_COUNT))
    map(lambda proc: proc.start(), producers)
    consumers = map(lambda x: Process(target=utils.consume, 
                                        args=(images, 
                                                (otsu, TARGET), 
                                                extract)),
                    xrange(0, 5))
    map(lambda proc: proc.start(), consumers)
    parts.join()
    images.join()
    map(lambda proc: proc.terminate(), consumers) 

    end = datetime.now()
    print end-start