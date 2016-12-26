import argparse
from datetime import datetime
from functools import partial
import numpy as np
import os
from sparks.multiprocessor import MultiProcessor
from sparks import utils

def otsu(infile, outfile):
    read_tiles = MultiProcessor(functions=[utils.get_tile, utils.to_hsv], 
                                output_names=['image'], 
                                initializator=utils.init_openslide,
                                initializator_args={"filename": infile},
                                max_size=1024,
                                threads_num=6)

    make_histogram = MultiProcessor(functions=[utils.accumulated_histogram], 
                                    output_names=['histogram'], 
                                    max_size=1,
                                    threads_num=3,
                                    update=True)

    read_tiles.init_input_queue(partial(utils.make_coords, infile))
    make_histogram.set_input_queue(read_tiles.get_output())    
    read_tiles.start()    
    make_histogram.put_into_out_queue([np.zeros([256, 2], dtype=np.float32)])
    make_histogram.start()
    read_tiles.join()
    make_histogram.join()

    histogram = make_histogram.get_output().get()['histogram']    
    otsu = list(map(lambda x: utils.calculate_otsu(histogram[:,x]), range(2)))

    utils.save_histogram_with_otsu(os.path.basename(infile), 
                                    list(map(lambda x: histogram[:,x], 
                                                range(2))), 
                                    otsu, 
                                    outfile + ".png")

    with open(outfile, "w") as file_handle:
        file_handle.write(" ".join('%s' % x for x in otsu))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--input', type=str, nargs=1, required=True,
                    help='Path to file used for calculating Otsu threshold.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                    help='Path to file where to store thresholds.')
    args = parser.parse_args()
    FILE = args.input[0]
    TARGET = args.output[0]

    start = datetime.now()
    otsu(FILE, TARGET)
    end = datetime.now()
    print (end-start)