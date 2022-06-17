import startup
import os
import numpy as np
import imageio
import scipy.io

import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim

from models import models
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from util.common import parse_lines
from util.app_config import config as app_config
from util.system import setup_environment
from util.train import get_path
from util.simple_dataset import Dataset3D
from util.fs import mkdir_if_missing
from util.data import tf_record_compression
from util.camera import get_full_camera, quaternion_from_campos
from util.visualise import vis_pc, merge_grid, mask4vis
from util.point_cloud import pointcloud2voxels, smoothen_voxels3d, pointcloud2voxels3d_fast, pointcloud_project_fast
from util.quaternion import as_rotation_matrix, quaternion_rotate


def parse_tf_records(cfg, serialized):
    num_views = cfg.num_views
    image_size = cfg.image_size

    # A dictionary from TF-Example keys to tf.FixedLenFeature instance.
    features = {
        'name': tf.FixedLenFeature([1], tf.string),
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

def gen_plot(data, title):
    """Create a pyplot plot and save to buffer."""
    x, y = data[:, 0], data[:, 1]
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
    
def compute_predictions():
    cfg = app_config
    setup_environment(cfg)

    exp_dir = get_path(cfg)
    exp_dir = os.path.join(exp_dir, str(cfg.vox_size))
    result_dir = os.path.join(exp_dir, "Tensorboard")
    mkdir_if_missing(result_dir)

    pc_num_points = cfg.pc_num_points
    vox_size = cfg.vox_size
    save_pred = cfg.save_predictions
    save_voxels = cfg.save_voxels
    fast_conversion = True
    
    num_views = 4
    plot_h = 4
    plot_w = 6
    grid = np.empty((plot_h, plot_w), dtype=object)
    vis_size = cfg.vis_size

    save_dir = os.path.join(exp_dir, '{}_vis_proj'.format(cfg.save_predictions_dir))
    mkdir_if_missing(save_dir)
    save_pred_dir = os.path.join(exp_dir, cfg.save_predictions_dir)
    mkdir_if_missing(save_pred_dir)

    split_name = 'test'
    dataset_file = os.path.join(cfg.inp_dir, f"{cfg.synth_set}_{split_name}.tfrecords")
    dataset = tf.data.TFRecordDataset(dataset_file, compression_type=tf_record_compression(cfg))
    dataset = dataset.map(lambda rec: parse_tf_records(cfg, rec), num_parallel_calls=4) \
        .batch(cfg.batch_size) \
        .prefetch(buffer_size=100)

    print(dataset_file)
    iterator = dataset.make_one_shot_iterator()
    train_data = iterator.get_next()

    model = models.ModelPointCloud(cfg)

    inputs = model.preprocess(train_data, cfg.step_size)
    model_fn = model.get_model_fn(
        is_training=True, reuse=False, run_projection=True)
    outputs = model_fn(inputs)

    plot_buf = tf.placeholder(tf.string)
    proj_image = tf.placeholder(dtype=tf.float64)

    image = tf.image.decode_png(plot_buf, channels=4)
    image = tf.expand_dims(image, 0)  # make it batched
    image_summary = tf.summary.image("2D_Points", image, max_outputs=1)
    proj_summary = tf.summary.image("2D_Projection", proj_image)


    session_config = tf.ConfigProto(
        log_device_placement=False)
    session_config.gpu_options.allow_growth = cfg.gpu_allow_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    with tf.Session(config=session_config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(result_dir, flush_secs=10, graph=sess.graph)
        variables_to_restore = slim.get_variables_to_restore(exclude=["meta"])
        restorer = tf.train.Saver(variables_to_restore)

        checkpoint_file = os.path.join("../Baseline Results/lr-0.0001_dataset-03001627_pointn-6400_gtp-5000/32", 'model-{}'.format(600000))
        restorer.restore(sess, checkpoint_file)
        count = 0
        while count < cfg.max_test_steps:
            result, images, gtmasks, gtpoints, byte_list = sess.run([outputs, inputs['images'], inputs['masks'], inputs['inpoints'], inputs['name'][0]])
            model_name = byte_list[0].decode('UTF-8')
            print("{}/{}: {}".format(count, cfg.max_test_steps, model_name))
            pred =  result['test_o'][0]
            gt = gtpoints[0]
            pred_buf = gen_plot(pred,"Predicted Image")
            gt_buf = gen_plot(gt, "Target Image")
            
            pred_image = sess.run(image_summary, feed_dict = {plot_buf: pred_buf.getvalue()})
            summary_writer.add_summary(pred_image, count)
            pred_mask = np.expand_dims(result['projs'][0], axis = 0)
            pred2D = sess.run(proj_summary, feed_dict = {proj_image: pred_mask})
            summary_writer.add_summary(pred2D, count)

            gt_image = sess.run(image_summary, feed_dict = {plot_buf: gt_buf.getvalue()})
            summary_writer.add_summary(gt_image, count)
            gt_mask = np.expand_dims(gtmasks[0], axis = 0)
            gt2D = sess.run(proj_summary, feed_dict = {proj_image: gt_mask})
            summary_writer.add_summary(gt2D, count)
            
            if save_pred:
                if cfg.save_as_mat:
                    save_dict = {"points": result['points_1']}
                    scipy.io.savemat("{}/{}_pc".format(save_pred_dir, model_name),
                                    mdict=save_dict)
                else:
                    np.savez("{}/{}_pc".format(save_pred_dir, model_name), result['points_1'])

                if save_voxels:
                    np.savez("{}/{}_vox".format(save_pred_dir, model_name), result['voxels'])

            for view_idx in range(num_views):
                input_image_np = images[[view_idx], :, :, :]
                gt_image = gtmasks[[view_idx], :, :, :]
                view_j = view_idx * 2 // plot_w
                view_i = view_idx * 2 % plot_w
                gt_image = np.squeeze(gt_image)
                grid[view_j, view_i] = mask4vis(cfg, gt_image, vis_size)

                curr_img = np.squeeze(result['projs'][view_idx])
                grid[view_j, view_i + 1] = mask4vis(cfg, curr_img, vis_size)
                
                if cfg.save_individual_images:
                    curr_dir = os.path.join(save_dir, model_name)
                    if not os.path.exists(curr_dir):
                        os.makedirs(curr_dir)

                if cfg.save_individual_images:
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'rgb_gt')),
                                    mask4vis(cfg, np.squeeze(input_image_np), vis_size))
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'mask_pred')),
                                    mask4vis(cfg, np.squeeze(result['projs'][view_idx]), vis_size))
            
            grid_merged = merge_grid(cfg, grid)
            imageio.imwrite("{}/{}_proj.png".format(save_dir, model_name), grid_merged)
            count += 1
       
    sess.close()
    print('over')
    return dataset

def main(_):
    compute_predictions()


if __name__ == '__main__':
    tf.app.run()
