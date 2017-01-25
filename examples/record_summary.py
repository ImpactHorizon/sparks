from os import listdir, path
import tensorflow as tf
import time
from sparks import tf_input

feature_map = {
        #'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
        #                                    default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1)
        }
    
total_tumors = 0
total_samples = 0
target_dir = "C:\\workspace\\data\\learning_data"

with open(path.join(target_dir, "stats.txt"), "a") as file_handle:
        for file in listdir(target_dir):
            tf.reset_default_graph()
            if 'normal' in file:
                continue
            if file == "stats.txt":
                continue
            print(file)   
            samples = 0
            tumors = 0
            start=time.time()
            full_path = path.join(target_dir, file)
            for record in tf.python_io.tf_record_iterator(full_path):                    
                samples += 1

            if samples == 0:
                continue
                  
            filename_queue = tf.train.string_input_producer([full_path])
            read_input = tf_input.read(filename_queue)
            images, label_batch = tf.train.batch(
                                [read_input.uint8image, read_input.label],
                                batch_size=samples,
                                num_threads=8,
                                capacity=samples) 
              
            total = tf.reduce_sum(tf.cast(label_batch, tf.int32))  

            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                tumors = sess.run(total)
                coord.request_stop()
                coord.join(threads)             

            print(time.time()-start)

            file_handle.write("%s: %d %d\n" % (file, tumors, samples))
            total_tumors += tumors
            total_samples += samples            
        file_handle.write("Total: %d %d" % (total_tumors, total_samples))