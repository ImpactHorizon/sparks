import argparse
from datetime import datetime
from functools import partial
import numpy as np
from sparks.multiprocessor import MultiProcessor
from sparks import utils

def moving_average(a, n=3) :
    ret = a.ravel()
    ret = np.cumsum(ret, dtype=np.float32)
    ret = np.insert(ret, 0, [a[-1]]*int((n-1)/2))
    ret = np.insert(ret, ret.size, [a[0]]*int((n-1)/2))
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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
    make_histogram.put_into_out_queue([np.zeros([256, 3], dtype=np.float32)])
    make_histogram.start()
    read_tiles.join()
    make_histogram.join()

    histogram = make_histogram.get_output().get()['histogram']
    otsu = list(map(lambda x: utils.calculate_otsu(histogram[:,x]), range(3)))

    hsv_old = -1
    hsv_new = otsu[0]
    while abs(hsv_new - hsv_old) > 2:
        print(hsv_old, hsv_new)
        histogram[:,0] = moving_average(histogram[:,0])
        hsv_old = hsv_new
        hsv_new = utils.calculate_otsu(histogram[:,0])
    otsu[0] = hsv_new
    plot = utils.save_histogram_with_otsu(utils.basename(filename), 
                                            list(map(lambda x: histogram[:,x], 
                                                        range(3))), 
                                            otsu)
    return (plot, otsu) 

def save_results(otsu, plot, dir_path):
    hist_path = utils.os.path.join(dir_path, "histograms.png")
    otsu_path = utils.os.path.join(dir_path, "otsus.txt")
    plot.savefig(hist_path)    
    with open(otsu_path, "w") as file_handle:
        file_handle.write(" ".join('%s' % x for x in otsus))

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