import argparse
import cv2
from datetime import datetime
from multiprocessing import Event, JoinableQueue, Process, Queue
import numpy as np
import time
import utils

CPU_COUNT = 8

def accumulated_histogram(images, args):
    histogram = args[0]
    try:
        image, x, y = images.get(timeout=0.3)
    except:
        return
    hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    current_histogram = histogram.get()    
    new_histogram = list(map(lambda x: cv2.calcHist([hsv[:,:,x]], 
                                                [0], 
                                                None, 
                                                [256], 
                                                [0, 256], 
                                                hist=current_histogram[x], 
                                                accumulate=True), 
                    range(3)))
    histogram.put(new_histogram)
    images.task_done() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--input', type=str, nargs=1, required=True,
                    help='Path to file used for calculating Otsu threshold.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                    help='Path to file where to store thresholds.')
    args = parser.parse_args()
    FILE = args.input[0]
    TARGET = args.output[0]

    parts = JoinableQueue(CPU_COUNT)
    images = JoinableQueue(1024)
    histogram = Queue(1)
    histogram.put(list(map(lambda x: np.zeros([256, 1], dtype=np.float32), 
                            range(3))))

    start = datetime.now()      

    list(map(lambda i: parts.put((i, CPU_COUNT)), 
                            range(CPU_COUNT)))   
    producers = map(lambda x: Process(target=utils.process_tiles_parallel, 
                                        args=(FILE, parts, images)), 
                    range(CPU_COUNT))
    list(map(lambda proc: proc.start(), producers))
    consumers_events = list(map(lambda x: Event(), range(CPU_COUNT)))
    consumers = map(lambda x: Process(target=utils.consume, 
                                        args=(images, consumers_events[x],
                                                (histogram,), 
                                                accumulated_histogram)),
                    range(CPU_COUNT))
    list(map(lambda proc: proc.start(), consumers))
    parts.join()
    images.join()
    list(map(lambda event: event.set(), consumers_events))
    list(map(lambda proc: proc.join(), consumers))
    list(map(lambda proc: proc.join(), producers))
    otsu = map(lambda x: utils.calculate_otsu(x), histogram.get())

    with open(TARGET, "w") as file_handle:
        file_handle.write(" ".join('%s' % x for x in otsu))

    end = datetime.now()
    print (end-start)