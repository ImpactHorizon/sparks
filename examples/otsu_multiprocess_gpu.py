import argparse
from datetime import datetime
from functools import partial
import numpy as np
from multiprocessor import MultiProcessor
import utils
import utils_cuda

# there is no gain with gpu, main part of processing is reading the tile

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
    read_tiles = MultiProcessor(functions=[utils.get_tile], 
                                output_names=['image'], 
                                initializator=utils.init_openslide,
                                initializator_args={"filename": FILE},
                                max_size=1024,
                                threads_num=8)

    gpu_process = MultiProcessor(functions=[utils_cuda.gpu_process], 
                                    output_names=['histogram'], 
                                    max_size=1,
                                    threads_num=1,
                                    update=True)

    read_tiles.init_input_queue(partial(utils.make_coords, FILE))
    gpu_process.set_input_queue(read_tiles.get_output())
    read_tiles.start()   
    gpu_process.put_into_out_queue(np.zeros([256, 2], dtype=np.float32))
    gpu_process.start()
    read_tiles.join()
    gpu_process.join()

    histogram = gpu_process.get_output().get()['histogram']    
    otsu = list(map(lambda x: utils.calculate_otsu(histogram[:,x]), range(2)))

    with open(TARGET, "w") as file_handle:
        file_handle.write(" ".join('%s' % x for x in otsu))

    end = datetime.now()
    print (end-start)