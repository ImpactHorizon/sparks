from io import StringIO
import openslide
import os
from PIL import Image 
import re
from sparks import utils
import tensorflow as tf

IMAGE_SIZE = 128
CHANNELS = 3
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

def read(filename_queue):
    class CAM17Record(object):
        pass

    result = CAM17Record()

    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = CHANNELS

    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)

    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1)
        }
    features = tf.parse_single_example(value, feature_map)
    result.label = tf.cast(features['image/class/label'], dtype=tf.int8)

    image_buffer = features['image/encoded']
    image = tf.image.decode_jpeg(image_buffer, channels=CHANNELS)

    depth_major = tf.reshape(image,
                                [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

def read_slide(coords_queue, handler):
    class CAM17Tile(object):
        pass

    result = CAM17Tile()

    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = CHANNELS

    reader = tf.ReaderBase()
    coords, unused = reader.read(coords_queue)

    image, x, y = tf.placeholder(utils.get_tile(coords[0], coords[1], handler, 
                                    (IMAGE_SIZE, IMAGE_SIZE)))
    image = image[:,:,:3]

    depth_major = tf.reshape(image,
                                [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                                [image, label],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                                [image, label],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def _generate_image_batch(image, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    images = tf.train.batch([image],
                            batch_size=batch_size,
                            num_threads=num_preprocess_threads,
                            capacity=min_queue_examples + 3 * batch_size)
    return images


def distorted_inputs(data_dir, batch_size, leave_idx):
    files = os.listdir(data_dir)
    filenames = []    
    all_folds = None
    for file in files:
        match = re.match('(normal|test|tumor)_\d\d\d-(\d\d)-of-(\d\d)', file)
        if match:
            if all_folds is None:
                all_folds = int(match.group(3))
            else:
                if all_folds != int(match.group(3)):
                    raise ValueError("Datasets have different folds value.")
            if leave_idx == int(match.group(2)):
                continue
            else:
                filenames.append(os.path.join(data_dir, file))

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,
                                                    max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                                lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CAM17 images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=True)

def inputs(data_dir, use_fold, batch_size):
    files = os.listdir(data_dir)
    filenames = []    
    all_folds = None
    for file in files:
        match = re.match('(normal|test|tumor)_\d\d\d-(\d\d)-of-(\d\d)', file)
        if match:
            if all_folds is None:
                all_folds = int(match.group(3))
            else:
                if all_folds != int(match.group(3)):
                    raise ValueError("Datasets have different folds value.")
            if use_fold == int(match.group(2)):
                filenames.append(os.path.join(data_dir, file))                
            else:
                continue

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

    float_image = tf.image.per_image_standardization(resized_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                                min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=False)

def inputs_from_slide(filename, batch_size):        

    handler = utils.init_openslide(filename)['handler']

    print("kak")
    coords_queue = tf.train.input_producer(utils.make_coords_list(handler))
    print("kak2")

    read_input = read_slide(coords_queue, handler)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    float_image = tf.image.per_image_standardization(reshaped_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                                min_fraction_of_examples_in_queue)

    return _generate_image_batch(float_image, min_queue_examples, batch_size,
                                    shuffle=False)