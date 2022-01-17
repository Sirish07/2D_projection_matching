#!/usr/bin/env python

import startup
import numpy as np
import os
import time

import tensorflow as tf

from models import models

from util.app_config import config as app_config
from util.system import setup_environment
from util.train import get_trainable_variables, get_learning_rate_origin, get_learning_rate, get_path
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression
import tensorflow.contrib.slim as slim
from scipy.io import loadmat
from main.predict_eval import test_one_step
tfsum = tf.contrib.summary


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

def train():
    cfg = app_config
    setup_environment(cfg)
    train_dir = get_path(cfg)
    train_dir = os.path.join(train_dir, str(cfg.vox_size))
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
    
    global_step = tf.train.get_or_create_global_step()
    model = models.ModelPointCloud(cfg, global_step)
    inputs = model.preprocess(train_data, cfg.step_size)
    model_fn = model.get_model_fn(
        is_training=True, reuse=False, run_projection=True)
    outputs = model_fn(inputs)
    # train_scopes
    train_scopes = ['encoder', 'decoder']
    # # loss
    task_loss, c_loss, k_loss, de_loss = model.get_loss(inputs, outputs)
    reg_loss = regularization_loss(train_scopes, cfg)
    loss = task_loss #+ reg_loss
    # optimizer
    learning_rate = get_learning_rate(cfg, global_step)
    tf.summary.scalar("Learning_Rate", learning_rate)
    ## Learning Rate Summary
    lr_summary_op = tf.summary.merge_all()
    var_list = get_trainable_variables(train_scopes)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step, var_list)

    # Epoch Loss summary
    tf.summary.scalar("Epoch_Loss", per_epoch_loss)
    loss_summary_op = tf.summary.merge_all()

    # saver
    max_to_keep = 120
    saver = tf.train.Saver(max_to_keep=max_to_keep)

    session_config = tf.ConfigProto(
        log_device_placement=False)
    session_config.gpu_options.allow_growth = cfg.gpu_allow_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    with tf.Session(config=session_config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(train_dir, flush_secs=10, graph=sess.graph)
        # # if you want restore model or finetune model, uncomment here.
        # checkpoint_file = os.path.join(train_dir, 'model-{}'.format(cfg.test_step))
        # saver.restore(sess, checkpoint_file)

        global_step_val = 0
        epoch_loss = 0
        t0 = time.perf_counter()
        while global_step_val <= cfg['max_number_of_steps']:
            
            _, loss_val, global_step_val, lr_summary, result = sess.run([train_op, loss, global_step, lr_summary_op, outputs])
            summary_writer.add_summary(lr_summary, global_step_val)
            temp = result['all_points']
            points3d = result['points3D']
            assert temp[0].all() == temp[1].all()
            assert temp[0].all() == points3d.all()
            is_nan = np.isnan(loss_val)
            assert(not np.any(is_nan))
            epoch_loss += loss_val
            
            if global_step_val % 50000 == 0 and global_step_val > 0:
                print("Checking Distributions")
                print("Decoder Output" + ": " + str(result['image2pc'].min()) + "," + str(result['image2pc'].max()) + "," + str(np.mean(result['image2pc'])) + "," + str(np.std(result['image2pc'])))
                print("Fused point clouds Output" + ": " + str(result['points3D'].min()) + "," + str(result['points3D'].max()) + "," + str(np.mean(result['points3D'])) + "," + str(np.std(result['points3D'])))
                print("2D Projections Output" + ": " + str(result['projs'].min()) + "," + str(result['projs'].max()) + "," + str(np.mean(result['projs'])) + "," + str(np.std(result['projs'])))

            if global_step_val % trainlen == 0 and global_step_val > 0:
                epoch_loss *= 100
                t1 = time.perf_counter()
                dt = t1 - t0
                print(f"Un-normalised Loss is = {epoch_loss}")
                print(f"step: {global_step_val}, loss = {epoch_loss/trainlen:.8f}, {dt:.6f} sec/epoch")
                loss_summary = sess.run(loss_summary_op, feed_dict = {per_epoch_loss:epoch_loss/trainlen})
                summary_writer.add_summary(loss_summary, global_step_val)
                epoch_loss = 0
                t0 = time.perf_counter()

            if global_step_val % 50000 == 0 and global_step_val > 0:
                saver.save(sess, f"{train_dir}/model", global_step=global_step_val)

            # if global_step_val % 50000 == 0 and global_step_val > 0:
            #     test_one_step(global_step_val)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
