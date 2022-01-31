import startup
import numpy as np
import os
import time

import tensorflow as tf

from util.app_config import config as app_config
from util.system import setup_environment
from util.fs import mkdir_if_missing
from util.train import get_path
from util.data import tf_record_compression
from main.predict_eval import test_one_step
from models import models

def parse_tf_records(cfg, serialized):
    num_views = cfg.num_views
    image_size = cfg.image_size

    # A dictionary from TF-Example keys to tf.FixedLenFeature instance.
    features = {
        'image': tf.FixedLenFeature([num_views, image_size, image_size, 3], tf.float32),
        'mask': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32),
        'inpoints':tf.FixedLenFeature([num_views, cfg.gt_point_n, 2], tf.float32),
    }

    if cfg.saved_camera:
        features.update(
            {'extrinsic': tf.FixedLenFeature([num_views, 4, 4], tf.float32),
             'cam_pos': tf.FixedLenFeature([num_views, 3], tf.float32)})
    if cfg.saved_depth:
        features.update(
            {'depth': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32)})

    return tf.parse_single_example(serialized, features)

if __name__ == "__main__":
    cfg = app_config
    setup_environment(cfg)
    train_dir = get_path(cfg)
    train_dir = os.path.join(train_dir, "debug")
    mkdir_if_missing(train_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    split_name = 'train'
    dataset_file = os.path.join(cfg.inp_dir, f"{cfg['synth_set']}_{split_name}.tfrecords")
    print(dataset_file)

    dataset = tf.data.TFRecordDataset(dataset_file, compression_type=tf_record_compression(cfg))
    trainlen = 4733 - 1 #sum(1 for _ in dataset) - 1 -> 0 based indexing
    per_epoch_loss = tf.placeholder(dtype=tf.float64)

    if cfg.shuffle_dataset:
        dataset = dataset.shuffle(7000)
    dataset = dataset.map(lambda rec: parse_tf_records(cfg, rec), num_parallel_calls=4) \
        .batch(cfg.batch_size) \
        .prefetch(buffer_size=100) \
        .repeat()

    iterator = dataset.make_one_shot_iterator()
    train_data = iterator.get_next()
    model = models.ModelPointCloud(cfg)
    inputs = model.preprocess(train_data, cfg.step_size)
    

    session_config = tf.ConfigProto(
        log_device_placement=False)
    session_config.gpu_options.allow_growth = cfg.gpu_allow_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    with tf.Session(config=session_config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        count = 0
        print(cfg.max_number_of_steps)
        while count <= 10:
            gtpoints, gtimage = sess.run([inputs['inpoints'], inputs['images']])
            gtpoints = np.array(gtpoints[0])
            np.save(os.path.join(train_dir, "images_" + str(count) + "_.npy"), gtimage)
            np.save(os.path.join(train_dir, "points_" + str(count) + "_.npy"), gtpoints)
            count += 1