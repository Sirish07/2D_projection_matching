import startup

import os

import numpy as np
import imageio
import scipy.io

import tensorflow as tf
import tensorflow.contrib.slim as slim

from models import models

from util.common import parse_lines
from util.app_config import config as app_config
from util.system import setup_environment
from util.train import get_path
from util.simple_dataset import Dataset3D
from util.fs import mkdir_if_missing
from util.camera import get_full_camera, quaternion_from_campos
from util.visualise import vis_pc, merge_grid, mask4vis
from util.point_cloud import pointcloud2voxels, smoothen_voxels3d, pointcloud2voxels3d_fast, pointcloud_project_fast
from util.quaternion import as_rotation_matrix, quaternion_rotate


def build_model(model):
    cfg = model.cfg()
    batch_size = cfg.batch_size
    inputs = tf.placeholder(dtype=tf.float32, shape=[cfg.step_size, cfg.image_size, cfg.image_size, 3])
    masks = tf.placeholder(dtype=tf.float32, shape=[cfg.step_size, cfg.image_size,cfg.image_size,1])
    cam_quaternion = tf.placeholder(dtype=tf.float32, shape=[cfg.step_size, 4])
    model_fn = model.get_model_fn(is_training=False, reuse=False)
    code = 'images'
    input = {code: inputs,
             'masks':masks,
             'camera_quaternion': cam_quaternion}
    outputs = model_fn(input)
    cam_transform = tf.no_op()
    outputs["inputs"] = inputs
    outputs["cam_quaternion"] = cam_quaternion
    outputs["cam_transform"] = cam_transform
    return outputs


def model_unrotate_points(cfg):
    """
    un_q = quat_gt^(-1) * predicted_quat
    pc_unrot = un_q * pc_np * un_q^(-1)
    """

    from util.quaternion import quaternion_normalise, quaternion_conjugate, \
        quaternion_rotate, quaternion_multiply
    input_pc = tf.placeholder(dtype=tf.float32, shape=[1, cfg.pc_num_points, 3])
    pred_quat = tf.placeholder(dtype=tf.float32, shape=[1, 4])
    gt_quat = tf.placeholder(dtype=tf.float32, shape=[1, 4])

    pred_quat_n = quaternion_normalise(pred_quat)
    gt_quat_n = quaternion_normalise(gt_quat)

    un_q = quaternion_multiply(quaternion_conjugate(gt_quat_n), pred_quat_n)
    pc_unrot = quaternion_rotate(input_pc, un_q)

    return input_pc, pred_quat, gt_quat, pc_unrot


def normalise_depthmap(depth_map):
    depth_map = np.clip(depth_map, 1.5, 2.5)
    depth_map -= 1.5
    return depth_map

def quaternion_from_campos_wrapper(campos):
            num = campos.shape[0]
            out = np.zeros([num, 4], dtype=np.float32)
            for k in range(num):
                out[k, :] = quaternion_from_campos(campos[k, :])
            return out

def compute_predictions():
    cfg = app_config

    setup_environment(cfg)

    exp_dir = get_path(cfg)
    exp_dir = os.path.join(exp_dir, str(cfg.vox_size))
     
    cfg.batch_size = 1
    cfg.step_size = 4

    pc_num_points = cfg['pc_num_points']
    vox_size = cfg.vox_size
    save_pred = cfg.save_predictions
    save_voxels = cfg.save_voxels
    fast_conversion = True

    pose_student = cfg.pose_predictor_student and cfg.predict_pose

    g = tf.Graph()
    with g.as_default():
        model = models.ModelPointCloud(cfg)

        out = build_model(model)
        input_image = out["inputs"]
        cam_quaternion = out["cam_quaternion"]
        point_cloud = out["points3D"]
        projs = out["projs"]
        projs_depth = out["projs_depth"]
        cam_transform = out["cam_transform"]

        input_pc = tf.placeholder(dtype = tf.float32, shape = [1, pc_num_points, 3])
        if save_voxels:
            if fast_conversion:
                # print(input_pc)
                voxels, _ = pointcloud2voxels3d_fast(cfg, input_pc, None)
                voxels = tf.expand_dims(voxels, axis=-1)
                voxels = smoothen_voxels3d(cfg, voxels, model.gauss_kernel())
            else:
                voxels = pointcloud2voxels(cfg, input_pc, model.gauss_sigma())

        q_inp = tf.placeholder(tf.float32, [1, 4])
        q_matrix = as_rotation_matrix(q_inp)

        input_pc_unrot, pred_quat, gt_quat, pc_unrot = model_unrotate_points(cfg)
        pc_rot = quaternion_rotate(input_pc_unrot, pred_quat)

        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
        config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        variables_to_restore = slim.get_variables_to_restore(exclude=["meta"])

    restorer = tf.train.Saver(variables_to_restore)
    checkpoint_file = tf.train.latest_checkpoint(exp_dir)

    print("restoring checkpoint", checkpoint_file)
    restorer.restore(sess, checkpoint_file)

    save_dir = os.path.join(exp_dir, '{}_vis_proj'.format(cfg.save_predictions_dir))
    mkdir_if_missing(save_dir)
    save_pred_dir = os.path.join(exp_dir, cfg.save_predictions_dir)
    mkdir_if_missing(save_pred_dir)

    vis_size = cfg.vis_size

    dataset = Dataset3D(cfg)
    pose_num_candidates = cfg.pose_predict_num_candidates
    num_views = cfg.step_size # only for saving
    plot_h = 4
    plot_w = 6
    num_views = int(min(num_views, plot_h * plot_w / 2))

    model_names = [sample.name for sample in dataset.data]

    num_models = len(model_names)
    modelCount = 0
    for k in range(num_models):
        if modelCount > 100:
            break
        modelCount += 1
        model_name = model_names[k]
        sample = dataset.sample_by_name(model_name)

        images = sample.image[:num_views]
        masks = sample.mask
        if cfg.saved_camera:
            cameras = sample.camera[:num_views]
            cam_pos = sample.cam_pos[:num_views]

        print("{}/{} {}".format(k, num_models, sample.name))
        grid = np.empty((plot_h, plot_w), dtype=object)
        cam_quaternions = []
        for view_idx in range(num_views):
            temp_camera = quaternion_from_campos(cam_pos[view_idx, :])
            cam_quaternions.append(temp_camera)
        
        cam_quaternions = np.stack(cam_quaternions)
        
        all_pcs = np.zeros((cfg.batch_size, pc_num_points, 3))
        all_voxels = np.zeros((cfg.batch_size, vox_size, vox_size, vox_size))
        (pc_np, proj_np, cam_transf_np) = sess.run([point_cloud,projs, cam_transform],
                                                               feed_dict={input_image: images,
                                                                          cam_quaternion: cam_quaternions})

        print("Checking projection distribution")
        print(proj_np.shape)
        print(proj_np.min(), proj_np.max(), np.mean(proj_np), np.std(proj_np))
        all_pcs = np.squeeze(pc_np)
        # multiplying by two is necessary because
        # pc->voxel conversion expects points in [-1, 1] range
        pc_np_range = pc_np
        if not fast_conversion:
            pc_np_range *= 2.0
        # print(pc_np_range)
        voxels_np = sess.run([voxels], feed_dict={input_pc: pc_np_range})
        all_voxels = np.squeeze(voxels_np)

        if save_pred:
            if cfg.save_as_mat:
                save_dict = {"points": all_pcs}
                scipy.io.savemat("{}/{}_pc".format(save_pred_dir, sample.name),
                                 mdict=save_dict)
            else:
                np.savez("{}/{}_pc".format(save_pred_dir, sample.name), all_pcs)

            if save_voxels:
                np.savez("{}/{}_vox".format(save_pred_dir, sample.name), all_voxels)

        for view_idx in range(num_views):
            input_image_np = images[[view_idx], :, :, :]
            gt_mask_np = masks[[view_idx], :, :, :]
            gt_image = gt_mask_np

            if pose_num_candidates == 1:
                view_j = view_idx * 2 // plot_w
                view_i = view_idx * 2 % plot_w

                gt_image = np.squeeze(gt_image)
                grid[view_j, view_i] = mask4vis(cfg, gt_image, vis_size)

                curr_img = np.squeeze(proj_np[view_idx])
                grid[view_j, view_i + 1] = mask4vis(cfg, curr_img, vis_size)

                if cfg.save_individual_images:
                    curr_dir = os.path.join(save_dir, sample.name)
                    if not os.path.exists(curr_dir):
                        os.makedirs(curr_dir)
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'rgb_gt')),
                                    mask4vis(cfg, np.squeeze(input_image_np), vis_size))
                    imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'mask_pred')),
                                    mask4vis(cfg, np.squeeze(proj_np[view_idx]), vis_size))

        grid_merged = merge_grid(cfg, grid)
        imageio.imwrite("{}/{}_proj.png".format(save_dir, sample.name), grid_merged)

    sess.close()
    print('over')
    return dataset

def main(_):
    compute_predictions()


if __name__ == '__main__':
    tf.app.run()
