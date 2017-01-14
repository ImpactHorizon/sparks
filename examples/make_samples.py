import argparse
from datetime import datetime
from functools import partial
import numpy as np
from sparks.multiprocessor import MultiProcessor
from sparks import utils
from sparks.counter import Counter

SAMPLE_SIZE = (128, 128)

def make_samples(filename, thresholds, tval, meat_percentage, distribution, 
                    target_dir, mask=None):
    samples_dir = utils.os.path.join(target_dir, "samples")
    number_of_samples = utils.calc_number_of_samples(filename,
                                                        meat_percentage, 
                                                        SAMPLE_SIZE)
    print(number_of_samples)  
    produce_coords = MultiProcessor(functions=[partial(utils.pull_coords, 
                                                    distribution=distribution)], 
                                    output_names=['xy'],
                                    initializator=utils.init_sampler_coords,
                                    initializator_args={"filename": filename},
                                    max_size=4096,
                                    threads_num=3,
                                    mode="producer")
    check_coords = MultiProcessor(functions=[partial(utils.check_coords, 
                                                    read_size=SAMPLE_SIZE,
                                                    tval=tval,
                                                    thresholds=thresholds)], 
                                    output_names=['image', 
                                                    'x', 
                                                    'y', 
                                                    'tumor'],
                                    initializator=utils.init_openslide,
                                    initializator_args={"filename": filename,
                                                        "maskname": mask},
                                    max_size=4096,
                                    threads_num=6,
                                    counter=Counter(number_of_samples),
                                    mode="producer")
    store_image = MultiProcessor(functions=[partial(utils.save_image, 
                            target_dir=samples_dir)],
                            initializator=utils.init_sampler_directories,
                            initializator_args={"target_dir": samples_dir,
                                                "class_dirs": ["healthy", 
                                                                "tumor", 
                                                                "boundaries"]}, 
                            output_names=['x', 'y'],
                            max_size=number_of_samples,
                            threads_num=4,
                            mode="consumer")    

    check_coords.set_input_queue(produce_coords.get_output())   
    store_image.set_input_queue(check_coords.get_output())
    produce_coords.start() 
    check_coords.start()
    store_image.start()
    check_coords.join()
    produce_coords.join(clear_out=True)
    store_image.join()
    output = store_image.get_output()
    hmap = np.zeros(utils.calc_size_in_tiles(filename, SAMPLE_SIZE), 
                    dtype=np.uint16)
    while not output.empty():
        sample = output.get()
        utils.samples_heatmap(sample['x'], sample['y'], hmap, SAMPLE_SIZE)
    mini = None
    if mask:
        mini = utils.get_mini(mask)
    plot = utils.save_heatmap(hmap.swapaxes(0, 1), mini)
    return (plot, )

def save_results(plot, dir_path):
    heat_path = utils.os.path.join(dir_path, "samples.png")
    plot.savefig(heat_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--input', type=str, nargs=1, required=True,
                        help='Path to file used for generating samples.')
    parser.add_argument('--params', type=str, nargs=1, required=True,
                        help='Path to file with params (meat percentage and ' + 
                        'meat threshold).')
    parser.add_argument('--distribution', type=str, nargs=1, required=True,
                        help='Path to file with distribution of probabilites' + 
                        ' where meat is.')
    parser.add_argument('--thresholds', type=str, nargs=1, required=True,
                        help='Path to file with otsu thresholds in HSV.')
    parser.add_argument('--mask', type=str, nargs=1, required=False,
                        help='Path to mask file with tumor regions.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                        help='Path to file where to write samples.')
    args = parser.parse_args()
    FILE = args.input[0]
    PARAMS = args.params[0]
    DISTRIBUTION = args.distribution[0]
    THRESHOLDS = args.thresholds[0]
    TARGET = args.output[0]
    MASK = None    

    if args.mask:
        MASK = args.mask[0]

    with open(PARAMS, "r") as file_handle:
        line = file_handle.read()        
        tval, meat_percentage = list(map(lambda x: float(x), line.split(' ')))

    with open(THRESHOLDS, "r") as file_handle:
        line = file_handle.read()
        otsu = list(map(lambda y: list(map(lambda z: int(z), y)), 
                            tuple(map(lambda x: x.split(" "), 
                                        line.split("\n")[:-1]))))

    vals_dist = np.fromfile(DISTRIBUTION)
    start = datetime.now()
    dir_path = utils.make_directory(utils.os.path.join(
                                                TARGET, utils.basename(FILE)))
    plot, = make_samples(FILE, otsu, tval, meat_percentage, vals_dist, dir_path, 
                        MASK)
    save_results(plot, dir_path)
    end = datetime.now()
    print (end-start) 