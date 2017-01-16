import argparse
from datetime import datetime
from functools import partial
from make_heatmap import make_heatmap, save_results as save_heatmap
from make_samples import make_samples, save_results as save_samples
import numpy as np
from os.path import exists, join, basename
from otsu_multiprocess import otsu, save_results as save_otsu
from prepare_tf_record import process_dataset
from sparks.gdrive_reader import GDriveReader
from sparks.multiprocessor import MultiProcessor
from sparks import utils

NOT_EXHAUSTED_TUMORS = ['tumor_015', 'tumor_018', 'tumor_020', 'tumor_029', 
                        'tumor_033', 'tumor_044', 'tumor_046', 'tumor_051',
                        'tumor_054', 'tumor_055', 'tumor_079', 'tumor_092',
                        'tumor_095', 'test_114']

class ProcessingUnit():
    def __init__(self, name, name_id, mask=None, mask_id=None):
        self.name = name
        self.name_id = name_id
        self.mask = mask
        self.mask_id = mask_id

    def __str__(self):
        return str(self.name) + " " + str(self.mask != None)

def process(unit, target_dir, records_dir, folds):
    start = datetime.now()
    base_name = unit.name.split(".")[0]
    dir_name = join(target_dir, base_name)    
    if not exists(dir_name):
        return
    print("Processing", base_name)

    filename = join(dir_name, unit.name)
    if exists(filename):
        print("File exists.")

    thresholds_file = join(dir_name, "otsus.txt")
    if exists(thresholds_file):
        print("Skipping thresholds...")        
        with open(thresholds_file, "r") as file_handle:
            line = file_handle.read()
            thresholds = list(map(lambda y: list(map(lambda z: int(z), y)), 
                                    tuple(map(lambda x: x.split(" "), 
                                                line.split("\n")[:-1]))))
    else:
        print("Calculating thresholds...")
        plot, thresholds = otsu(filename)
        save_otsu(thresholds, plot, dir_name)

    params_file = join(dir_name, "params.txt")
    distribution_file = join(dir_name, "distribution")
    if exists(join(dir_name, "distribution")):
        print("Skipping heatmap...")
        with open(params_file, "r") as file_handle:
            line = file_handle.read()        
            tval, meat_percentage = list(map(lambda x: float(x), 
                                                line.split(' ')))        
    else:
        print("Calculating heatmap...")
        plot_heat, tval, meat_percentage, hmap_norm = make_heatmap(
                                                        filename, thresholds)
        save_heatmap(tval, meat_percentage, hmap_norm, plot_heat, dir_name)

    labels_file = join(dir_name, "labels.txt")
    if exists(join(dir_name, "samples.png")):
        print("Skipping samples.")
    else:
        print("Making samples...")
        mask = None
        if unit.mask is not None:
            mask = join(dir_name, unit.mask)
        vals_dist = np.fromfile(join(dir_name, "distribution"))
        plot_samples, = make_samples(filename, thresholds, tval, 
                                meat_percentage, vals_dist, dir_name, mask)
        save_samples(plot_samples, dir_name)
        labels = ["healthy", "tumor"]
        if base_name in NOT_EXHAUSTED_TUMORS:
            labels = ['tumor']

        with open(labels_file, "w") as file_handle:
            file_handle.write("\n".join('%s' % x for x in labels))

    if exists(join(dir_name, "DONE")):
        print("Skipping conversion to TF records.")
    else:
        print("Converting to TF records...")
        process_dataset(base_name, join(dir_name, "samples"), folds, 
                        labels_file, records_dir)
        with open(join(dir_name, "DONE"), "w") as file_handle:
            file_handle.write("FINISHED")

    end = datetime.now()
    print("Done", base_name, "in", end-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                        help='Path to directory store intermediate results.')
    parser.add_argument('--records', type=str, nargs=1, required=True,
                        help='Path to directory where to write TF records.')
    parser.add_argument('--folds', type=int, nargs=1, required=True,
                        help='Number of folds to create.')
    args = parser.parse_args()

    TARGET = args.output[0]  
    RECORDS = args.records[0]  
    FOLDS = args.folds[0]

    units = []
    gdrive = GDriveReader()
    files_list = gdrive.list("title contains '.tif'")

    files = {x[1].lower() : x[0] for x in files_list}

    for file in sorted(files.keys()):
        if "mask" in file:
            continue
        if "tumor" in file or "test" in file:
            parts = file.split(".")
            mask_key = parts[0] + "_mask." + parts[1]
            if mask_key in files:
                unit = ProcessingUnit(file, 
                                        files[file], 
                                        mask_key, 
                                        files[mask_key])
            else:
                unit = ProcessingUnit(file, files[file])
        elif "normal" in file:
            unit = ProcessingUnit(file, files[file])
        else:
            continue
        units.append(unit)  

    for u in units:
        process(u, TARGET, RECORDS, FOLDS) 

