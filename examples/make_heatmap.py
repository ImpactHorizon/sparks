import argparse
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import openslide
from sparks import utils
from sparks.multiprocessor import MultiProcessor
from sparks.pdf import MY_PDF

def generate_samples(filename, thresholds, target, mask=None):
    read_tiles = MultiProcessor(functions=[utils.get_tile, utils.to_hsv], 
                                output_names=['image', 'x', 'y'], 
                                initializator=utils.init_openslide,
                                initializator_args={"filename": filename},
                                max_size=1024,
                                threads_num=6)

    scan_tiles = MultiProcessor(functions=[partial(utils.scan,
                                                    thresholds=thresholds)], 
                                    output_names=['coords'], 
                                    max_size=1,
                                    threads_num=4,
                                    update=True)

    read_tiles.init_input_queue(partial(utils.make_coords, filename))
    scan_tiles.set_input_queue(read_tiles.get_output())    
    read_tiles.start()    
    scan_tiles.put_into_out_queue([[]])
    scan_tiles.start()
    read_tiles.join()
    scan_tiles.join()

    coords = scan_tiles.get_output().get()

    coords = coords['coords'] 
    op = openslide.OpenSlide(filename)
    size = op.dimensions
    a = np.zeros((int(size[0]/512), int(size[1]/512)))
    for coord in coords:
        a[int(coord[0]/512), int(coord[1]/512)] = coord[2]

    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--input', type=str, nargs=1, required=True,
                        help='Path to file used for generating samples.')
    parser.add_argument('--thresholds', type=str, nargs=1, required=True,
                        help='Path to file with otsu thresholds in HSV.')
    parser.add_argument('--mask', type=str, nargs=1, required=False, 
                        help='Path to mask file with tumor regions.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                        help='Path to file where to write samples.')
    args = parser.parse_args()
    FILE = args.input[0]
    if args.mask:
        MASK = args.mask[0]
    else:
        MASK = None
    THRESHOLDS = args.thresholds[0]
    TARGET = args.output[0]

    with open(THRESHOLDS, "r") as file_handle:
        line = file_handle.read()
        otsu = list(map(lambda x: int(x), line.split(' ')))

    start = datetime.now()
    generate_samples(FILE, otsu, TARGET, MASK)
    end = datetime.now()
    print (end-start)    