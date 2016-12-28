import argparse
from datetime import datetime
from functools import partial
import numpy as np
from sparks import utils
from sparks.multiprocessor import MultiProcessor

def make_heatmap(filename, thresholds):
    read_tiles = MultiProcessor(functions=[utils.get_tile, utils.to_hsv], 
                                output_names=['image', 'x', 'y'], 
                                initializator=utils.init_openslide,
                                initializator_args={"filename": filename},
                                max_size=1024,
                                threads_num=6)

    scan_tiles = MultiProcessor(functions=[partial(utils.scan,
                                                    thresholds=thresholds)], 
                                    output_names=['hmap'], 
                                    max_size=1,
                                    threads_num=4,
                                    update=True)

    read_tiles.init_input_queue(partial(utils.make_coords, filename))
    scan_tiles.set_input_queue(read_tiles.get_output())    
    read_tiles.start()    
    scan_tiles.put_into_out_queue([np.zeros(utils.calc_size_in_tiles(filename), 
                                            dtype=np.float64)])
    scan_tiles.start()
    read_tiles.join()
    scan_tiles.join()

    output = scan_tiles.get_output().get()
    hmap = output['hmap']
    hist, bins = np.histogram(hmap, bins=100, range=(0.0, 1.0), density=True)
    heatmap_otsu = utils.calculate_otsu(hist, 100)/100.0
    plot = utils.save_thresholds_heatmap(hmap, 
                                            hist, 
                                            bins, 
                                            heatmap_otsu)  
    meat_percentage = (np.count_nonzero(np.where(hmap > heatmap_otsu, 
                                                    1, 0)) / hmap.size) 
    hmap_norm = hmap / np.sum(hmap)
    return (plot, heatmap_otsu, meat_percentage, hmap_norm)  

def save_results(hmap_otsu, meat_percentage, hmap, plot, dir_path):
    heat_path = utils.os.path.join(dir_path, "heatmap.png")
    params_path = utils.os.path.join(dir_path, "params.txt")
    distribution_path = utils.os.path.join(dir_path, "distribution")
    plot.savefig(heat_path)
    with open(params_path, "w") as file_handle:
        file_handle.write(str(hmap_otsu) + " " + str(meat_percentage))
    with open(distribution_path, "w") as file_handle:
        hmap.tofile(file_handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--input', type=str, nargs=1, required=True,
                        help='Path to file used for generating samples.')
    parser.add_argument('--thresholds', type=str, nargs=1, required=True,
                        help='Path to file with otsu thresholds in HSV.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                        help='Path to file where to write samples.')
    args = parser.parse_args()
    FILE = args.input[0]
    THRESHOLDS = args.thresholds[0]
    TARGET = args.output[0]

    with open(THRESHOLDS, "r") as file_handle:
        line = file_handle.read()
        otsu = list(map(lambda x: int(x), line.split(' ')))

    start = datetime.now()
    plot, hmap_otsu, meat_percentage, hmap = make_heatmap(FILE, otsu)
    dir_path = utils.make_directory(utils.os.path.join(
                                                TARGET, utils.basename(FILE)))
    save_results(hmap_otsu, meat_percentage, hmap, plot, dir_path)
    end = datetime.now()
    print (end-start)    