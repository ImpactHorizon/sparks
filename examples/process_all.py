from sparks.blob_reader import BlobReader
from sparks.blob_saver import BlobSaver
from multiprocessing import JoinableQueue, Process, Event
from sparks import utils
from tempfile import NamedTemporaryFile
import time
import os
import sys
import subprocess

IN_ACC = "camelyon16data"
IN_KEY = "5juqtl5oUnYS3W7CRX3qNfCnYp5ReEh1RHv7AEMIx9Nu9ryL7K7xL/4y7vOH6aN/SFh5CeSaIognarZaRyeTnA=="
IN_CONTAINER = "camelyon16"
IN_PREPATH = "TrainingData/Train_"

OUT_PREPATH = None
OUT_CONTAINER = "otsu"
OUT_ACC = "samplessw"
OUT_KEY = "2wr3eLjg+olIVZmEyGF+FUEBLO0KyXgcv2NgXslQmcnR5Lrv1egHbDXstSNXKu+BzgvU2XNgUo6lRRX/dVbrUA=="

def check_for_mask(blob):
    if blob.lower().find("tumor") > 0:
        return True
    else:
        return False

def download_blob(blobs, args):
    acc, key, container = args[0]
    files = args[1]
    blob = blobs.get()
    reader = BlobReader(acc, key, container)
    mask = None
    mask_name = None
    if (check_for_mask(blob.name)):
        number = os.path.basename(blob.name).split(".")[0].split("_")[1]
        mask = ("TrainingData/Ground_Truth/Mask/Tumor_%s_Mask.tif" % (number))
        print("Downloading %s ..." % (mask))
        mask_handle = NamedTemporaryFile(dir="C://temp", delete=False)
        reader.to_file(mask_handle, mask)
        mask_name = mask_handle.name
    print("Downloading %s ..." % (blob.name))
    handle = NamedTemporaryFile(dir="C:/temp", delete=False)
    reader.to_file(handle, blob.name)    
    files.put((blob.name, handle.name, mask, mask_name))
    blobs.task_done()

def process(files, args):
    try:
        blob_name, handle_name, mask_name, mask_handle = files.get(timeout=0.3)    
    except:
        return
    runner = args[0]
    acc, key, container = args[1]
    filename = os.path.basename(blob_name).split(".")[0]
    prefix = runner.split(".")[0]
    outname = prefix + "_" + filename
    threshname = "./otsu_" + filename
    with open(handle_name, "r") as handle:
        print("Processing... %s" % (handle_name))        
        command = ["python", runner, handle_name, threshname]
        if mask_handle:
            command.append(mask_handle)
        command.append("/ssd_data/samples_best")        
        ret = subprocess.call(command)
        if not ret == 0:
            print("Processing failed for %s." % (handle_name))
    os.remove(handle_name)
    if mask_handle:
        os.remove(mask_handle)
    saver = BlobSaver(acc, key, container, "")      
    saver(os.path.basename(outname), "OK")
    files.task_done()

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        raise AttributeError("Missing runner.")

    RUNNER = args[0]

    downloads = JoinableQueue()
    files = JoinableQueue(5)
    download_event = Event()
    process_event = Event()
    in_reader = BlobReader(IN_ACC, IN_KEY, IN_CONTAINER)    
    blobs = in_reader.list(IN_PREPATH)
    out_reader = BlobReader(OUT_ACC, OUT_KEY, OUT_CONTAINER)
    outs = out_reader.list(OUT_PREPATH)    

    print(len(blobs), len(outs))
    for x in outs:
        print(x.name)
    for x in blobs:
        print(x.name)
    blobs = list(filter(lambda x: os.path.basename(x.name).split(".")[0] not in         
                    list(map(lambda x: "_".join(x.name.split("_")[1:]).split(".")[0], 
                        outs)), blobs))
    print(len(blobs))
    list(map(lambda x: downloads.put(x), blobs))

    downloaders = map(lambda x: Process(target=utils.consume, 
                                        args=(downloads,
                                                download_event, 
                                                ((IN_ACC, IN_KEY, IN_CONTAINER), 
                                                    files), 
                                                download_blob)), 
                    range(0, 2))
    list(map(lambda proc: proc.start(), downloaders))
    
    processors = map(lambda x: Process(target=utils.consume, 
                                        args=(files,
                                                process_event, 
                                                (RUNNER, 
                                                    (OUT_ACC, 
                                                        OUT_KEY, 
                                                        OUT_CONTAINER)), 
                                                process)), 
                    range(0, 1))
    list(map(lambda proc: proc.start(), processors))

    downloads.join()
    files.join()
    download_event.set()
    process_event.set()