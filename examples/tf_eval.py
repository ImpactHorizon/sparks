import argparse
from datetime import datetime
import math
import numpy as np
import time
import tensorflow as tf
from sparks import tf_core

def eval_once(model_dir, saver, summary_writer, top_k_op, summary_op, 
                num_examples, batch_size):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(num_examples / batch_size))
            true_count = 0
            total_sample_count = num_iter * batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate(eval_dir, model_dir, data_dir, batch_size, use_fold, num_examples):
    with tf.Graph().as_default() as g:
        images, labels = tf_core.inputs(data_dir, use_fold, batch_size)

        logits = tf_core.inference(images, batch_size)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages = tf.train.ExponentialMovingAverage(
                                                 tf_core.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        for key in variables_to_restore.keys():
            print(key)
        input(len(variables_to_restore))
        saver = tf.train.Saver(variables_to_restore)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.train.SummaryWriter(eval_dir, g)        

        eval_once(model_dir, saver, summary_writer, top_k_op, summary_op, 
                    num_examples, batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--eval_dir', type=str, nargs=1, required=True,
                        help='Path to directory where to store logs.')
    parser.add_argument('--data_dir', type=str, nargs=1, required=True,
                        help='Path to directory with eval data.')
    parser.add_argument('--model_dir', type=str, nargs=1, required=True,
                        help='Path to model checkpoints from training.')
    parser.add_argument('--batch_size', type=int, nargs=1, required=True,
                        help='Size of single batch.')
    parser.add_argument('--use_fold', type=int, nargs=1, required=True,
                        help='Index of fold to use.')
    parser.add_argument('--num_examples', type=int, nargs=1, required=True,
                        help='Index of fold to use.')
    
    args = parser.parse_args()
    eval_dir = args.eval_dir[0]
    data_dir = args.data_dir[0]
    model_dir = args.model_dir[0]
    batch_size = args.batch_size[0]
    use_fold = args.use_fold[0]
    num_examples = args.num_examples[0]

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    evaluate(eval_dir, model_dir, data_dir, batch_size, use_fold, num_examples)
