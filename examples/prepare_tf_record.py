import argparse
from datetime import datetime
import multiprocessing
import numpy as np
import os
import random
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import threading


class ImageCoder(object):

    def __init__(self):
        self._sess = tf.Session()
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', 
                                                    quality=100)
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, 
                                                    channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                                feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _is_png(filename):
    return '.png' in filename

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(image_buffer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': _int64_feature(label),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example

def _process_image(filename, coder):
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    image = coder.decode_jpeg(image_data)

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, output_directory):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                                 ranges[thread_index][1],
                                 num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.2d-of-%.2d' % (name, shard, num_shards)    
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], 
                                    dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(image_buffer, label)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
                (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
            (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, texts, labels, num_shards, threads,
                            output_directory):
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    spacing = np.linspace(0, len(filenames), threads + 1).astype(np.int)
    ranges = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    print('Launching %d threads for spacings: %s' % (threads, ranges))
    sys.stdout.flush()

    coord = tf.train.Coordinator()

    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards, output_directory)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
            (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir, labels_file):
    labels_map = {"healthy" : 0, "tumor" : 1}

    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []    

    for text in unique_labels:
        label_index = labels_map[text]
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        if len(matching_files) < 30 and len(matching_files) > 0:
            factor = int(30 / len(matching_files)) + 1
            matching_files *= factor

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        print('Finished finding files in %d of %d classes.' % (
                label_index+1, len(unique_labels)))

    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
            (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels

def process_dataset(name, directory, num_folds, labels_file, output_directory):
    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, texts, labels, num_folds, num_folds, 
                            output_directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--in_dir', type=str, nargs=1, required=True,
                        help='Path to file used for generating samples.')
    parser.add_argument('--folds', type=int, nargs=1, required=True,
                        help='Path to file with params (meat percentage and ' + 
                        'meat threshold).')
    parser.add_argument('--labels_file', type=str, nargs=1, required=True,
                        help='Path to file with distribution of probabilites' + 
                        ' where meat is.')
    parser.add_argument('--output', type=str, nargs=1, required=True,
                        help='Path to file with otsu thresholds in HSV.')
    args = parser.parse_args()

    in_dir = args.in_dir[0]
    folds = args.folds[0]
    labels_file = args.labels_file[0]
    output = args.output[0]

    name = in_dir.split("\\")[-2]
    process_dataset(name, in_dir, folds, labels_file, output)