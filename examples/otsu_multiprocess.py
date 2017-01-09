import argparse
from datetime import datetime
from functools import partial
import numpy as np
from sparks.multiprocessor import MultiProcessor
from sparks import utils

def otsu(filename):
    read_tiles = MultiProcessor(functions=[utils.get_tile, utils.to_hsv], 
                                output_names=['image'], 
                                initializator=utils.init_openslide,
                                initializator_args={"filename": filename},
                                max_size=1024,
                                threads_num=6)

    make_histogram = MultiProcessor(functions=[utils.accumulated_histogram], 
                                    output_names=['histogram'], 
                                    max_size=1,
                                    threads_num=3,
                                    update=True)

    read_tiles.init_input_queue(partial(utils.make_coords, filename))
    make_histogram.set_input_queue(read_tiles.get_output())    
    read_tiles.start()    
    make_histogram.put_into_out_queue([3*[np.zeros([256], dtype=np.float32)]])
    make_histogram.start()
    read_tiles.join()
    make_histogram.join()

    histogram = make_histogram.get_output().get()['histogram']
    otsu = tuple(map(lambda x: (utils.calculate_otsu(histogram[x]), ), 
                        range(1,3)))
    hue_hist = histogram[0][:180].ravel()
    peaks = utils.detect_peaks(hue_hist)    
    otsu = (utils.circular_otsu(hue_hist, int((peaks[0]+peaks[1])/2)), ) + otsu    
    histogram[0] = hue_hist
    plot = utils.save_histogram_with_otsu(utils.basename(filename), 
                                            list(map(lambda x: histogram[x], 
                                                        range(3))), 
                                            otsu)
    return (plot, otsu) 

def save_results(otsus, plot, dir_path):
    hist_path = utils.os.path.join(dir_path, "histograms.png")
    otsu_path = utils.os.path.join(dir_path, "otsus.txt")
    plot.savefig(hist_path)    
    with open(otsu_path, "w") as file_handle:
        for val in otsus:
            file_handle.write(" ".join('%s' % x for x in val))
            file_handle.write("\n")

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
    plot, otsus = otsu(FILE)
    dir_path = utils.make_directory(utils.os.path.join(
                                                TARGET, utils.basename(FILE)))
    save_results(otsus, plot, dir_path)
    end = datetime.now()
    print (end-start)