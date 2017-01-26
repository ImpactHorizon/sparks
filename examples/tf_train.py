import argparse
from datetime import datetime
import tensorflow as tf
from sparks import tf_core
import time

def train(data_dir, train_dir, batch_size, leave_idx, max_steps):    
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, labels = tf_core.distorted_inputs(data_dir, 
                                                    leave_idx, 
                                                    batch_size)

        total_tumors = tf.get_variable("total_tumors", 
                                    shape=[],
                                    dtype=tf.float32, 
                                    initializer=tf.constant_initializer(0.0)) 
        total_tumors = total_tumors.assign_add(tf.reduce_sum(
                                                tf.cast(labels, tf.float32)))
        tf.summary.scalar("total_tumors", total_tumors)
        logits = tf_core.inference(images, batch_size)
        loss = tf_core.loss(logits, labels)
        train_op = tf_core.train(loss, global_step, batch_size)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                ckpt = tf.train.get_checkpoint_state(train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self._step = int(ckpt.model_checkpoint_path.split('/')
                                        [-1].split('-')[-1])
                else:
                    self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(fetches=[loss, total_tumors])

            def after_run(self, run_context, run_values):                 
                duration = time.time() - self._start_time
                loss_value = run_values.results[0]

                if self._step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec;'
                                    ' %.3f sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                            examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                            checkpoint_dir=train_dir,
                            hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                                    tf.train.NanTensorHook(loss), 
                                    _LoggerHook()],
                            config=tf.ConfigProto(
                                    log_device_placement=False)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process in/out info.')
    parser.add_argument('--train_dir', type=str, nargs=1, required=True,
                        help='Path to directory where to store logs.')
    parser.add_argument('--data_dir', type=str, nargs=1, required=True,
                        help='Path to directory with training data.')
    parser.add_argument('--batch_size', type=int, nargs=1, required=True,
                        help='Size of single batch.')
    parser.add_argument('--leave_out_fold', type=int, nargs=1, required=True,
                        help='Index of fold to leave out.')
    parser.add_argument('--max_steps', type=int, nargs=1, required=True,
                        help='Number of batches to run.')
    args = parser.parse_args()
    train_dir = args.train_dir[0]
    data_dir = args.data_dir[0]
    batch_size = args.batch_size[0]
    leave_idx = args.leave_out_fold[0]
    max_steps = args.max_steps[0]

    if not tf.gfile.Exists(train_dir):        
        tf.gfile.MakeDirs(train_dir)

    train(data_dir, train_dir, batch_size, leave_idx, max_steps)