import numpy as np
import scipy.io
import tensorflow.compat.v1 as tf
import io
from data_preprocessor.data_preprocessor import DataPreprocessor, pool_single_view

from util.losses import add_drc_loss, add_proj_rgb_loss, add_proj_depth_loss
from util.point_cloud import pointcloud_project, pointcloud_project_fast, \
    pc_point_dropout, chamfer_distance_topk, kl_distance, euclidean_distance_self, euclidean_distance_for_two_points, \
    another_euclidean_distance_for_fuzz_pc, euclidean_distance_for_fuzz_pc, chamfer_distance_self
from util.gauss_kernel import gauss_smoothen_image, smoothing_kernel
from util.quaternion import \
    quaternion_multiply as q_mul,\
    quaternion_normalise as q_norm,\
    quaternion_rotate as q_rotate,\
    quaternion_conjugate as q_conj

# from util.approxmatch import tf_approxmatch
# from util.nn_distance import tf_nndistance

from networks.net_factory import get_network
import imageio
import tensorflow.contrib.slim as slim


def tf_repeat_0(input, num):
    orig_shape = input.shape
    e = tf.expand_dims(input, axis=1)
    tiler = [1 for _ in range(len(orig_shape)+1)]
    tiler[1] = num
    tiled = tf.tile(e, tiler)
    new_shape = [-1]
    new_shape.extend(orig_shape[1:])
    final = tf.reshape(tiled, new_shape)
    return final


def get_smooth_sigma(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    diff = (cfg.pc_relative_sigma_end - cfg.pc_relative_sigma)
    sigma_rel = cfg.pc_relative_sigma + global_step / num_steps * diff
    sigma_rel = tf.cast(sigma_rel, tf.float32)
    return sigma_rel


def get_dropout_prob(cfg, global_step):
    if not cfg.pc_point_dropout_scheduled:
        return cfg.pc_point_dropout

    exp_schedule = cfg.pc_point_dropout_exponential_schedule
    num_steps = cfg.max_number_of_steps
    keep_prob_start = cfg.pc_point_dropout
    keep_prob_end = 1.0
    start_step = cfg.pc_point_dropout_start_step
    end_step = cfg.pc_point_dropout_end_step
    global_step = tf.cast(global_step, dtype=tf.float32)
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    if exp_schedule:
        alpha = tf.log(keep_prob_end / keep_prob_start)
        keep_prob = keep_prob_start * tf.exp(alpha * x)
    else:
        keep_prob = k * x + b
    keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
    keep_prob = tf.reshape(keep_prob, [])
    return tf.cast(keep_prob, tf.float32)


def get_st_global_scale(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    keep_prob_start = 0.0
    keep_prob_end = 1.0
    start_step = 0
    end_step = 0.1
    global_step = tf.cast(global_step, dtype=tf.float32)
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    keep_prob = k * x + b
    keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
    keep_prob = tf.reshape(keep_prob, [])
    return tf.cast(keep_prob, tf.float32)


def align_predictions(outputs, alignment):
    outputs["points_1"] = q_rotate(outputs["points_1"], alignment)
    outputs["poses"] = q_mul(outputs["poses"], q_conj(alignment))
    outputs["pose_student"] = q_mul(outputs["pose_student"], q_conj(alignment))
    return outputs


def predict_scaling_factor(cfg, input, is_training):
    if not cfg.pc_learn_occupancy_scaling:
        return None

    input = tf.nn.leaky_relu(input)
    init_stddev = 0.025
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=w_init,
            activation_fn=None):
        pred = slim.fully_connected(input, 1)
        pred = tf.sigmoid(pred) * cfg.pc_occupancy_scaling_maximum

    if is_training:
         tf.summary.scalar("pc_occupancy_scaling_factor", tf.reduce_mean(pred))

    return pred


def predict_focal_length(cfg, input, is_training):
    if not cfg.learn_focal_length:
        return None

    init_stddev = 0.025
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=w_init,
            activation_fn=None):
        pred = slim.fully_connected(input, 1)
        out = cfg.focal_length_mean + tf.sigmoid(pred) * cfg.focal_length_range

    if is_training:
        tf.summary.scalar("meta/focal_length", tf.reduce_mean(out))

    return out

def quaternionToRotMatrix(q):
    # Source: https://github.com/chenhsuanlin/3D-point-cloud-generation
	with tf.name_scope("quaternionToRotMatrix"):
		qa,qb,qc,qd = tf.unstack(q,axis=1)
		R = tf.transpose(tf.stack([[1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],
								   [2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],
								   [2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)]]),perm=[2,0,1])
	return R

def transParamsToHomMatrix(q,t):
    # Source: https://github.com/chenhsuanlin/3D-point-cloud-generation
	with tf.name_scope("transParamsToHomMatrix"):
		N = q.shape[0]
		R = quaternionToRotMatrix(q)
		Rt = tf.concat([R,tf.expand_dims(t,-1)],axis=2)
		hom_aug = tf.concat([tf.zeros([N,1,3]),tf.ones([N,1,1])],axis=2)
		RtHom = tf.concat([Rt,hom_aug],axis=1)
	return RtHom

def get3DhomCoord(XYZ,cfg):
    # Source: https://github.com/chenhsuanlin/3D-point-cloud-generation
	with tf.name_scope("get3DhomCoord"):
		ones = tf.ones([cfg.batch_size,cfg.step_size,cfg.image2pc_dim, cfg.image2pc_dim]) # [B, 4, H, W]
		XYZhom = tf.transpose(tf.reshape(tf.concat([XYZ,ones],axis=1),[cfg.batch_size,4,cfg.step_size,-1]),perm=[0,2,1,3])
	return XYZhom # [B,V,4,HW]

def fuseallimages(fuseTrans, fusedimages, cfg):
    # fusedimages = [B, H, W, 3V]
    # Source: https://github.com/chenhsuanlin/3D-point-cloud-generation
    # Code: https://github.com/chenhsuanlin/3D-point-cloud-generation/blob/master/transform.py#L6
    num_views = cfg.step_size
    height = fusedimages.shape[1]
    width = fusedimages.shape[2]
    renderDepth = 1.0
    with tf.name_scope("transform_fuse3D"):
        XYZ = tf.transpose(fusedimages,perm=[0, 3, 1, 2]) # [B, 3V, H, W]
        Khom2Dto3Dmatrix = np.array([[int(width),0 ,0,int(width)/2],
							   [0,-int(height),0,int(height)/2],
							   [0,0,-1,0],
							   [0,0, 0,1]],dtype=np.float32)
        
        invKhom = np.linalg.inv(Khom2Dto3Dmatrix)
        invKhomTile = np.tile(invKhom,[cfg.batch_size,num_views,1,1])
        q_view = tf.nn.l2_normalize(fuseTrans,dim=1)
        t_view = np.tile([0,0,-renderDepth],[num_views, 1]).astype(np.float32) # took renderDepth=1 similar to EPCG
        RtHom_view = transParamsToHomMatrix(q_view,t_view)
        RtHomTile_view = tf.tile(tf.expand_dims(RtHom_view,0),[cfg.batch_size,1,1,1])
        with tf.device("/cpu:0"):
            invRtHomTile_view = tf.matrix_inverse(RtHomTile_view)
        RtHomTile = tf.matmul(invRtHomTile_view,invKhomTile) # [B,V,4,4]
        RtTile = RtHomTile[:,:,:3,:] # [B,V,3,4]
        XYZhom = get3DhomCoord(XYZ, cfg) # [B,V,4,HW]
        XYZid = tf.matmul(RtTile,XYZhom) # [B, V, 3, HW]
        XYZid = tf.reshape(tf.transpose(XYZid,perm=[0,2,1,3]),[cfg.batch_size,3,-1]) # [B,3,VHW]
    return XYZid

class ModelPointCloud(DataPreprocessor):  # pylint:disable=invalid-name
    """Inherits the generic Im2Vox model class and implements the functions."""

    def __init__(self, cfg, global_step=0):
        super(ModelPointCloud, self).__init__(cfg)
        self._gauss_sigma = None
        self._gauss_kernel = None
        self._sigma_rel = None
        self._global_step = global_step
        self.setup_sigma()
        self.setup_misc()
        self._alignment_to_canonical = None

    def setup_sigma(self):
        cfg = self.cfg()
        sigma_rel = get_smooth_sigma(cfg, self._global_step)

        tf.summary.scalar("meta/gauss_sigma_rel", sigma_rel)
        self._sigma_rel = sigma_rel
        self._gauss_sigma = sigma_rel / cfg.vox_size
        self._gauss_kernel = smoothing_kernel(cfg, sigma_rel)

    def gauss_sigma(self):
        return self._gauss_sigmaq

    def gauss_kernel(self):
        return self._gauss_kernel

    def setup_misc(self):
        if self.cfg().pose_student_align_loss:
            num_points = 2000
            sigma = 1.0
            values = np.random.normal(loc=0.0, scale=sigma, size=(num_points, 3))
            values = np.clip(values, -3*sigma, +3*sigma)
            self._pc_for_alignloss = tf.Variable(values, name="point_cloud_for_align_loss",
                                                 dtype=tf.float32)

    def model_predict(self, images, is_training=False, reuse=False, predict_for_all=True, alignment=None):
        outputs = {}
        #images shape = [cfg.step_size, cfg.image_size, cfg.image_size, 3]
        cfg = self._params
        # First, build the encoder
        encoder_fn = get_network(cfg.encoder_name)
        with tf.variable_scope('encoder', reuse=reuse):
            # Produces id/pose units
            enc_outputs = encoder_fn(images, cfg, is_training)
            ids = enc_outputs['ids']
            outputs['conv_features'] = enc_outputs['conv_features'] 
            outputs['ids'] = ids # [B, 1024]
            outputs['z_latent'] = enc_outputs['z_latent'] # [B, 1024]
            outputs['encoderOut'] = enc_outputs['encoderOut'] # [B, 512]

        # Second, build the decoder and projector
        decoder_fn = get_network(cfg.decoder_name)
        with tf.variable_scope('decoder', reuse=reuse):
            key = 'encoderOut'
            decoder_out = decoder_fn(outputs[key], cfg, is_training)
            image2pc = decoder_out['xyz']
            outputs['image2pc'] = image2pc

        if self._alignment_to_canonical is not None:
            outputs = align_predictions(outputs, self._alignment_to_canonical)

        return outputs

    def get_dropout_keep_prob(self):
        cfg = self.cfg()
        return get_dropout_prob(cfg, self._global_step)

    def compute_projection(self, inputs, outputs, is_training):
        cfg = self.cfg()
        all_points = outputs['all_points']
        
        all_rgb = None
        
        if cfg.predict_pose:
            camera_pose = outputs['poses']
        else:
            if cfg.pose_quaternion:
                camera_pose = inputs['camera_quaternion']
            else:
                camera_pose = inputs['matrices']

        if is_training and cfg.pc_point_dropout != 1:
            dropout_prob = self.get_dropout_keep_prob()
            if is_training:
                tf.summary.scalar("meta/pc_point_dropout_prob", dropout_prob)
            all_points, all_rgb = pc_point_dropout(all_points, all_rgb, cfg.sample_scale / cfg.pc_num_points)
        if cfg.pc_fast:
            predicted_translation = outputs["predicted_translation"] if cfg.predict_translation else None
            proj_out = pointcloud_project_fast(cfg, all_points, camera_pose, predicted_translation,
                                               all_rgb, self.gauss_kernel(),
                                               scaling_factor=outputs['all_scaling_factors'],
                                               focal_length=None)
            
            proj = proj_out["proj"]
            outputs['tr_pc'] = proj_out['tr_pc']
            outputs["drc_probs"] = proj_out["drc_probs"]
            outputs["projs_depth"] = proj_out["proj_depth"]
            outputs["coord"] = proj_out["coord"]
            outputs["voxels"] = proj_out["voxels"]
        else:
            proj, voxels = pointcloud_project(cfg, all_points, camera_pose, self.gauss_sigma())
            outputs["projs_rgb"] = None
            outputs["projs_depth"] = None

        outputs['projs'] = proj
        return outputs

    def replicate_for_multiview(self, tensor):
        cfg = self.cfg()
        new_tensor = tf_repeat_0(tensor, cfg.step_size)
        return new_tensor

    def get_model_fn(self, is_training=True, reuse=False, run_projection=True):
        cfg = self._params

        def model(inputs):
            code = 'images'
            num_views = cfg.step_size
            outputs = self.model_predict(inputs[code], is_training, reuse)
            features = tf.concat([outputs['encoderOut'][0], outputs['encoderOut'][1], outputs['encoderOut'][2], outputs['encoderOut'][3]], axis = -1)
            points3D = tf.concat([outputs['image2pc'][0], outputs['image2pc'][1], outputs['image2pc'][2], outputs['image2pc'][3]], axis = -1)
            features = tf.expand_dims(features, axis = 0)
            points3D = tf.expand_dims(points3D, axis = 0)
            #points3D = [B, H, W, 3V]
            if run_projection:
                points3D = fuseallimages(inputs['camera_quaternion'], points3D, cfg) # [B, 3, VHW]
                points3D = tf.transpose(points3D, perm=[0, 2, 1]) # [B, VHW, 3]
                points3D = tf.tanh(points3D)
                if cfg.pc_unit_cube:
                    points3D = points3D / 2.0 
                outputs['points3D'] = points3D
                scaling_factor = predict_scaling_factor(cfg, features, is_training)
                all_points = self.replicate_for_multiview(points3D) # [4, VHW, 3]
                num_candidates = cfg.pose_predict_num_candidates
                outputs['all_points'] = all_points
                if cfg.pc_learn_occupancy_scaling:
                    all_scaling_factors = self.replicate_for_multiview(scaling_factor)
                    if num_candidates > 1:
                        all_scaling_factors = tf_repeat_0(all_scaling_factors, num_candidates)
                else:
                    all_scaling_factors = None
                outputs['all_scaling_factors'] = all_scaling_factors
                outputs = self.compute_projection(inputs, outputs, is_training)
                scale = 128 / cfg.vox_size
                outputs['test_o'] = outputs['coord'] * scale
            return outputs

        return model

    def bilinear_sampler(self,coords, imgs, n=1):
        """Construct a new image by bilinear sampling from the input image.

        Points falling outside the source image boundary have value 0.

        Args:
            imgs: source image to be sampled from [batch, height_s, width_s, channels]
            coords: coordinates of source pixels to sample from [batch, height_t,
              width_t, 2]. height_t/width_t correspond to the dimensions of the output
              image (don't need to be the same as height_s/width_s). The two channels
              correspond to x and y coordinates respectively.
        Returns:
        A new sampled image [batch, height_t, width_t, channels]
        """
        def _repeat(x, n_repeats):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([
                    n_repeats,
                ])), 1), [1, 0])
            rep = tf.cast(rep, 'float32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

        with tf.name_scope('encoder'):
            cfg = self.cfg()
            num_candidates = cfg.pose_predict_num_candidates
            imgs = tf_repeat_0(imgs, num_candidates)
            coords_x, coords_y = tf.split(coords, [1, 1], axis=2)
        
            if cfg.bicubic_gt_downsampling:
                interp_method = tf.image.ResizeMethod.BICUBIC
            else:
                interp_method = tf.image.ResizeMethod.BILINEAR
            imgs = tf.image.resize_images(imgs, [cfg.vox_size, cfg.vox_size], interp_method)
            
            imgs = tf.image.flip_up_down(imgs)

            ns = tf.constant(n)

            inp_size = imgs.get_shape()
            coord_size = coords.get_shape()
            out_size = coords.get_shape().as_list()
            out_size[2] = ns

            range_ns = tf.cast(tf.range(ns), 'float32')
            coords_x = tf.cast(coords_x, 'float32')
            coord_x = tf.tile(coords_x, [1,1,ns])
            coords_x_1 = coord_x - range_ns
            coords_x_2 = coord_x + range_ns

            coords_y = tf.cast(coords_y, 'float32')
            coord_y = tf.tile(coords_y, [1,1,ns])
            coords_y_1 = coord_y - range_ns
            coords_y_2 = coord_y + range_ns
            
            y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
            x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
            zero = tf.zeros([], dtype='float32') 
            one = tf.ones([], dtype='float32')
            coords_x_1 = tf.clip_by_value(coords_x_1, zero, x_max)
            coords_x_2 = tf.clip_by_value(coords_x_2, zero, x_max)
            coords_y_1 = tf.clip_by_value(coords_y_1, zero, y_max)
            coords_y_2 = tf.clip_by_value(coords_y_2, zero, y_max)
            
            x0 = tf.floor(coords_x_1)
            x1 = tf.floor(coords_x_2) + 1
            y0 = tf.floor(coords_y_1)
            y1 = tf.floor(coords_y_2) + 1

            x0_safe = tf.clip_by_value(x0, zero, x_max)
            y0_safe = tf.clip_by_value(y0, zero, y_max)
            x1_safe = tf.clip_by_value(x1, zero, x_max)
            y1_safe = tf.clip_by_value(y1, zero, y_max)

            wt_x0 = x1_safe - coords_x_2
            wt_y0 = y1_safe - coords_y_2
            wt_x1 = coords_x_1 - x0_safe
            wt_y1 = coords_y_1 - y0_safe
            
            dim2 = tf.cast(inp_size[2], 'float32') 
            dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
            base = tf.reshape(
                _repeat(
                    tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                    coord_size[1]),
                [out_size[0], out_size[1], 1])
            
            base_y0 = base + y0_safe * dim2
            base_y1 = base + y1_safe * dim2
            idx00 = tf.reshape(x0_safe + base_y0, [-1])
            idx01 = x0_safe + base_y1
            idx10 = x1_safe + base_y0
            idx11 = x1_safe + base_y1

            top_m = cfg.batch_size * cfg.vox_size * cfg.vox_size * cfg.step_size
            imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
            imgs_flat = tf.cast(imgs_flat, 'float32')
            int_idx00 = tf.clip_by_value(tf.cast(idx00, 'int32'), 0, top_m)
            im00 = tf.reshape(tf.gather(imgs_flat, int_idx00), out_size) 
            int_idx01 = tf.clip_by_value(tf.cast(idx01, 'int32'), 0, top_m)
            im01 = tf.reshape(tf.gather(imgs_flat, int_idx01), out_size)
            int_idx10 = tf.clip_by_value(tf.cast(idx10, 'int32'), 0, top_m)
            im10 = tf.reshape(tf.gather(imgs_flat, int_idx10), out_size)
            int_idx11 = tf.clip_by_value(tf.cast(idx11, 'int32'), 0, top_m)
            im11 = tf.reshape(tf.gather(imgs_flat, int_idx11), out_size)

            w00 = wt_x0 * wt_y0
            w01 = wt_x0 * wt_y1
            w10 = wt_x1 * wt_y0
            w11 = wt_x1 * wt_y1

            output = tf.add_n([
                w00 * im00, w01 * im01,
                w10 * im10, w11 * im11
            ])
            return output

    def proj_loss_pose_candidates(self, gt, pred, inputs):
        """
        :param gt: [BATCH*VIEWS, IM_SIZE, IM_SIZE, 1]
        :param pred: [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        :return: [], [BATCH*VIEWS]
        """
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates
        gt = tf_repeat_0(gt, num_candidates) # [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        sq_diff = tf.square(gt - pred)
        all_loss = tf.reduce_sum(sq_diff, [1, 2, 3]) # [BATCH*VIEWS*CANDIDATES]
        all_loss = tf.reshape(all_loss, [-1, num_candidates]) # [BATCH*VIEWS, CANDIDATES]
        min_loss = tf.argmin(all_loss, axis=1) # [BATCH*VIEWS]
        tf.summary.histogram("winning_pose_candidates", min_loss)

        min_loss_mask = tf.one_hot(min_loss, num_candidates) # [BATCH*VIEWS, CANDIDATES]
        num_samples = min_loss_mask.shape[0]

        min_loss_mask_flat = tf.reshape(min_loss_mask, [-1]) # [BATCH*VIEWS*CANDIDATES]
        min_loss_mask_final = tf.reshape(min_loss_mask_flat, [-1, 1, 1, 1]) # [BATCH*VIEWS*CANDIDATES, 1, 1, 1]
        loss_tensor = (gt - pred) * min_loss_mask_final
        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
            weights = tf_repeat_0(weights, num_candidates)
            weights = tf.reshape(weights, [weights.shape[0], 1, 1, 1])
            loss_tensor *= weights
        proj_loss = tf.nn.l2_loss(loss_tensor)
        proj_loss /= tf.to_float(num_samples)

        return proj_loss, min_loss

    def add_student_loss(self, inputs, outputs, min_loss, add_summary):
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates

        student = outputs["pose_student"]
        teachers = outputs["poses"]
        teachers = tf.reshape(teachers, [-1, num_candidates, 4])

        indices = min_loss
        indices = tf.expand_dims(indices, axis=-1)
        batch_size = teachers.shape[0]
        batch_indices = tf.range(0, batch_size, 1, dtype=tf.int64)
        batch_indices = tf.expand_dims(batch_indices, -1)
        indices = tf.concat([batch_indices, indices], axis=1)
        teachers = tf.gather_nd(teachers, indices)
        # use teachers only as ground truth
        teachers = tf.stop_gradient(teachers)

        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
        else:
            weights = 1.0

        if cfg.pose_student_align_loss:
            ref_pc = self._pc_for_alignloss
            num_ref_points = ref_pc.shape.as_list()[0]
            ref_pc_all = tf.tile(tf.expand_dims(ref_pc, axis=0), [teachers.shape[0], 1, 1])
            pc_1 = q_rotate(ref_pc_all, teachers)
            pc_2 = q_rotate(ref_pc_all, student)
            student_loss = tf.nn.l2_loss(pc_1 - pc_2) / num_ref_points
        else:
            q_diff = q_norm(q_mul(teachers, q_conj(student)))
            angle_diff = q_diff[:, 0]
            student_loss = tf.reduce_sum((1.0 - tf.square(angle_diff)) * weights)

        num_samples = min_loss.shape[0]
        student_loss /= tf.to_float(num_samples)

        if add_summary:
            tf.summary.scalar("losses/pose_predictor_student_loss", student_loss)
        student_loss *= cfg.pose_predictor_student_loss_weight

        return student_loss

    def add_proj_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        gt = inputs['masks']
        pred = outputs['projs']
        num_samples = pred.shape[0]

        gt_size = gt.shape[1]
        pred_size = pred.shape[1]
        assert gt_size >= pred_size, "GT size should not be higher than prediction size"
        if gt_size > pred_size:
            if cfg.bicubic_gt_downsampling:
                interp_method = tf.image.ResizeMethod.BICUBIC
            else:
                interp_method = tf.image.ResizeMethod.BILINEAR
            gt = tf.image.resize_images(gt, [pred_size, pred_size], interp_method)
        if cfg.pc_gauss_filter_gt:
            sigma_rel = self._sigma_rel
            smoothed = gauss_smoothen_image(cfg, gt, sigma_rel)
            if cfg.pc_gauss_filter_gt_switch_off:
                gt = tf.where(tf.less(sigma_rel, 1.0), gt, smoothed)
            else:
                gt = smoothed

        total_loss = 0
        num_candidates = cfg.pose_predict_num_candidates
        if num_candidates > 1:
            proj_loss, min_loss = self.proj_loss_pose_candidates(gt, pred, inputs)
            if cfg.pose_predictor_student:
                student_loss = self.add_student_loss(inputs, outputs, min_loss, add_summary)
                total_loss += student_loss
        else:
            proj_loss = tf.nn.l2_loss(gt - pred)
            proj_loss /= tf.to_float(num_samples)

        total_loss += proj_loss

        if add_summary:
            tf.summary.scalar("losses/proj_loss", proj_loss)

        total_loss *= weight_scale
        return total_loss
        
    def add_sum_entropy_distance_loss(self, inputs, outputs, weight_scale, weight_scale2, add_summary = True):
        cfg = self.cfg()
        pred = outputs['coord']
        pixels_weight = tf.squeeze(self.bilinear_sampler(pred, inputs['masks']),2)
        pixels_weight_tile = tf.tile(tf.expand_dims(pixels_weight,1),[1,cfg.pc_num_points,1])

        num_samples = pred.shape[0]
        pred_liner = pred
        _, en_distance = chamfer_distance_self(pred_liner, 2 * cfg.vox_size * cfg.vox_size)
        en_distance /= tf.to_float(cfg.vox_size)
        en_distance = tf.multiply(en_distance, pixels_weight_tile)
        en_distance = tf.reduce_mean(en_distance, 2)
        en_distance = tf.multiply(pixels_weight, en_distance)
        sum_distance_loss = tf.reduce_mean(en_distance)
     
        en_distance = tf.clip_by_value(en_distance, 0.00001, tf.to_float(cfg.vox_size))            
        dis_sum = tf.reduce_sum(en_distance, 1)
        dis_sum = tf.expand_dims(dis_sum, 1)
        dis_sum = tf.tile(dis_sum, [1, cfg.pc_num_points])
        en_distance = tf.divide(en_distance, dis_sum)
        test = tf.reduce_sum(en_distance, 1)
        entropy_distance_losses = -tf.reduce_sum(en_distance * tf.log(en_distance), 1)
        
        entropy_distance_loss = -weight_scale2 * tf.reduce_mean(entropy_distance_losses)
        loss = -sum_distance_loss * weight_scale
        return loss,entropy_distance_loss, en_distance, test
	
    def add_thermodynamic_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        pred = outputs['coord']
        num_samples = pred.shape[0]
        _, en_distance = chamfer_distance_self(pred, 2 * cfg.vox_size * cfg.vox_size)
        en_distance /= tf.to_float(cfg.vox_size)
        en_distance_min, en_distance = chamfer_distance_self(pred, 2 * cfg.vox_size * cfg.vox_size)
        en_distance = tf.clip_by_value(en_distance_min, 0.000001, tf.to_float(cfg.vox_size*cfg.vox_size))
        en_distance = tf.sqrt(en_distance)		
        ro = 0.412
        mr = tf.pow(tf.divide(ro, en_distance + 0.35), 1)
        en_loss = tf.multiply(mr, mr - 1)
        loss = weight_scale * tf.reduce_mean(en_loss)
        return loss

    def add_classify_loss(self, inputs, outputs,weight_scale, add_summary=True):
        real_points_type = inputs['points'][:,:,2]
        generate_points_type = outputs['point_type']
        loss = tf.reduce_mean(tf.square(generate_points_type - real_points_type))
        loss = weight_scale * loss
        return loss
    
    def add_opt_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        real_points_type = inputs['points'][:,:,2]
        
        temp = -0.1 * (tf.cast(tf.range(2), 'float32') + 1.0)
        temp = np.array([-0.05,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.7,-0.8,-0.85])
        temp = tf.cast(tf.tile(temp, [200]), 'float32')
        real_points_type = tf.expand_dims(temp, 0)
        generate_points_type = outputs['point_type']
        loss = tf.reduce_mean(tf.square(generate_points_type - real_points_type))
        loss = weight_scale * loss
        return loss

    def add_flowing_loss(self,inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        n = 4
        masks = inputs['masks']
        coords = outputs['test_o']
        mask_shape = masks.shape[1]
        coord_shape = coords.get_shape().as_list()
        lens = int(mask_shape) / n
        lens = int(lens)
        res = np.zeros(shape=[1,cfg.batch_size * cfg.step_size])
        for i in range(n):
            for j in range(n):
                masks_split = masks[:,i*lens:(i+1)*lens,j*lens:(j+1)*lens]
                mask_shapes = tf.cast(tf.reduce_sum(masks_split,[1,2,3]), 'float32') + 0.0001
                shape1 = lens * lens - mask_shapes
                middle_len = (2*i+1)/lens
                temp = np.expand_dims(np.array([i * lens, j * lens]),0)
                temp = np.expand_dims(np.tile(temp, [coord_shape[1], 1]),0)
                down_ = np.tile(temp, [coord_shape[0], 1,1])

                temp = np.expand_dims(np.array([(i+1) * lens, (j+1) * lens]),0)
                temp = np.expand_dims(np.tile(temp, [coord_shape[1], 1]),0)
                up_ = np.tile(temp, [coord_shape[0], 1,1])
                points = tf.where(tf.logical_and(coords >= down_, coords < up_),coords, -tf.ones(coord_shape))
                sample_2d_points = self.bilinear_sampler(points, masks, 1)
                shape2 = tf.reduce_sum(sample_2d_points, [1,2])
                result = tf.expand_dims(shape2 / shape1, 0)
                if i == 0 and j == 0:
                    res = result
                else:
                    res = tf.concat([res, result], 0)

        res = tf.reshape(res, [n,n,cfg.batch_size*cfg.step_size])

        fenbu = res * 1000
        l, h, batch_size = fenbu.shape
        nongdu = 0
        for i in range(l):
            for j in range(h):
                if i > 0:
                    nongdu += fenbu[i][j] + fenbu[i-1][j]
                elif i < l - 1:
                    nongdu += fenbu[i][j] + fenbu[i+1][j]
                elif j > 0:
                    nongdu += fenbu[i][j] + fenbu[i][j-1]
                elif j < h - 1:
                    nongdu += fenbu[i][j] + fenbu[i][j+1]
        loss = tf.reduce_mean(nongdu) * weight_scale 
        return loss

    def add_cd_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        n = cfg.pc_num_points
        pred = outputs['test_o']
        gt = inputs['inpoints'][:,:n,:2]
        
        zeros = tf.zeros([cfg.batch_size*cfg.step_size,n,1])
        pred = tf.concat([pred, zeros], 2)
        gt = tf.concat([gt, zeros], 2)
        
        a,b,c,d = tf_nndistance.nn_distance(pred, gt)
        loss = tf.reduce_mean(a) + tf.reduce_mean(c)
        return loss * weight_scale

    def add_emd_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        n = cfg.pc_num_points
        zero = tf.zeros([cfg.step_size * cfg.batch_size,n,1])
        pred = outputs['test_o']
        gt = inputs['inpoints'][:,:n,:2]
        pred = tf.concat([pred, zero], 2)
        gt = tf.concat([gt, zero], 2)
        match = tf_approxmatch.approx_match(gt, pred)
        cost = tf_approxmatch.match_cost(gt, pred, match)
        cost = cost / n
        loss = tf.reduce_mean(cost)
        return loss * weight_scale

    def add_kl_divergence_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        n = cfg.pc_num_points
        pred = outputs['test_o']
        gt = inputs['inpoints'][:,:n,:2]
        pred = pred / 128.0
        gt = gt / 128.0
        loss = kl_distance(gt, pred)
        loss = tf.reduce_mean(loss)
        return loss * weight_scale

    def add_exp_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pred = outputs['test_o']
        gt = inputs['inpoints'][:,:2000,:2]
        pred = pred / 128.0
        gt = gt / 128.0
        dis = cd_distance_2(gt, pred)
        exp_dis = tf.exp(-dis/10000000)
        mean_dis = tf.reduce_mean(exp_dis, 2)
        loss = tf.reduce_mean(mean_dis)
        rr = tf.ones([cfg.batch_size*cfg.step_size,2000]) / 2000.0
        exp_dis_sum = tf.reduce_sum(mean_dis,1)
        exp_dis_sum = tf.tile(tf.expand_dims(exp_dis_sum,1), [1,2000])
        mean_dis = tf.divide(mean_dis, exp_dis_sum)
        kl = tf.reduce_sum(mean_dis * (tf.log(mean_dis) - tf.log(rr)), 1)
        res = tf.reduce_mean(kl)
        return -loss, res

    def add_another_kl_divergence_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        n = cfg.pc_num_points
        pred = outputs['test_o']
        gt = inputs['inpoints'][:,:n,:2]
        pred = pred / 128.0
        gt = gt / 128.0
        pred_x = pred[:,:,0]
        pred_y = pred[:,:,1]
        gt_x = gt[:,:,0]
        gt_y = gt[:,:,1]
        scale_pred = tf.reduce_mean(pred_x)
        scale_gt = tf.reduce_mean(gt_x)
          
        pred_x = tf.concat([pred_x, n - tf.expand_dims(tf.reduce_sum(pred_x, 1), 1)], 1)
        pred_y = tf.concat([pred_y, n - tf.expand_dims(tf.reduce_sum(pred_y, 1), 1)], 1)
        gt_x = tf.concat([gt_x, n - tf.expand_dims(tf.reduce_sum(gt_x, 1), 1)], 1)
        gt_y = tf.concat([gt_y, n - tf.expand_dims(tf.reduce_sum(gt_y, 1), 1)], 1)

        pred_x = tf.divide(pred_x, tf.reduce_sum(pred_x, 1))
        pred_y = tf.divide(pred_y, tf.reduce_sum(pred_y, 1))
        gt_x = tf.divide(gt_x, tf.reduce_sum(gt_x, 1))
        gt_y = tf.divide(gt_y, tf.reduce_sum(gt_y, 1))
     
        _x = (pred_x + gt_x) / 2.0
        _y = (pred_y + gt_y) / 2.0
        kl_x = pred_x * (tf.log(pred_x + 1e-8) - tf.log(_x)) + gt_x * (tf.log(gt_x) - tf.log(_x))
        kl_y = pred_y * (tf.log(pred_y + 1e-8) - tf.log(_y)) + gt_y * (tf.log(gt_y) - tf.log(_y))
        kl = kl_x + kl_y
      
        loss = tf.reduce_mean(kl)
        return loss

    def add_cosine_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pred = outputs['test_o'] 
        gt = inputs['inpoints'][:,:2000,:2]
        pred = pred / 128.0
        gt = gt / 128.0
        pooled_len_1 = tf.sqrt(tf.reduce_sum(pred*pred,2)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(gt*gt, 2))
        pooled_mul_12 = tf.reduce_sum(pred * gt, 2) 
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
        loss = tf.reduce_mean(score)
        return loss
 
    def add_df_loss(self, inputs, outputs, weight_scale, add_summary=True):
        def normalization(matrix):
            mat_sum = tf.expand_dims(tf.reduce_sum(matrix, 2), 2)
            return tf.divide(matrix, mat_sum)

        cfg = self.cfg()
        n = cfg.pc_num_points
        pred = outputs['test_o'][:,:,:]
        gt = inputs['inpoints'][:,:n,:2] 
        pred = pred / 128.0 
        gt = gt / 128.0
        A = euclidean_distance_self(pred)
        outputs['A'] = A
        B = euclidean_distance_for_two_points(gt, pred)
        B_exp = tf.exp(-tf.square(B))
        BI = normalization(B_exp)
        C = euclidean_distance_self(gt)
        C_exp = tf.exp(-tf.square(C))
        CI = normalization(C_exp)
        D = BI * A
        F = CI * B
        D_sum = tf.reduce_sum(D, 2)
        F_sum = tf.reduce_sum(F, 2)
        loss = tf.reduce_mean(D_sum-F_sum)
        
        return loss * weight_scale
    
    def add_fuzz_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        n = 4000
        tile_num = 10
        pred = outputs['test_o']
        gt = inputs['inpoints'][:,:n,:]
        pred /= 128
        gt /= 128
        generate_pc = tf.tile(outputs['tr_pc'][:,:,None,:],(1,1,tile_num,1))
        gt_pc = inputs['fuzz_pc'][:,:n,:tile_num,:]
        pred_gt_dis = euclidean_distance2(pred, gt)
        pred_gt_min = tf.reduce_min(pred_gt_dis, 2)
        pred_gt_min = tf.tile(pred_gt_min[:,:,None], [1,1,n])
        pred_gt_D = 1 - pred_gt_dis + pred_gt_min
  
        pred_gt_temp = tf.eye(4000)
        pred_gt_temp = tf.expand_dims(pred_gt_temp, 0)
        pred_gt_E = tf.squeeze(euclidean_distance_for_fuzz_pc(generate_pc, gt_pc), 3)
        pred_gt_F = pred_gt_temp * pred_gt_E
        pred_gt_F = tf.matmul(pred_gt_temp, pred_gt_E)
        pred_gt_F = tf.reduce_min(pred_gt_F, 1)
        pred_gt_F = tf.reduce_mean(pred_gt_F, 1)
    
        gt_pred_dis = tf.transpose(pred_gt_dis,[0,2,1])
        gt_pred_min = tf.reduce_min(gt_pred_dis, 2)
        gt_pred_min = tf.tile(gt_pred_min[:,:,None], [1,1,n])
        gt_pred_D = 1 - gt_pred_dis + gt_pred_min
        gt_pred_temp = tf.transpose(pred_gt_temp, [0, 2, 1])
        gt_pred_E = tf.transpose(pred_gt_E, [0,2,1])
        gt_pred_F = gt_pred_temp * gt_pred_E
        gt_pred_F = tf.matmul(gt_pred_temp, gt_pred_E)
        gt_pred_F = tf.reduce_min(gt_pred_F, 1)
        gt_pred_F = tf.reduce_mean(gt_pred_F,1)
    
        loss = tf.reduce_mean(pred_gt_F) + tf.reduce_mean(gt_pred_F)
        return loss

    def add_another_fuzz_loss(self,inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pred = outputs['test_o']
        gt = inputs['inpoints']
        pred /= 128.0
        gt /= 128.0
        generate_pc = outputs['tr_pc']
        generate_pc = tf.tile(generate_pc[:,:,None,:],(1,1,20,1))
        gt_pc = inputs['fuzz_pc'][:,:4000,0:1,:]
        gt_pc = tf.tile(gt_pc, (1,1,20,1))
        dis1,nm = another_euclidean_distance_for_fuzz_pc(generate_pc, gt_pc)
        dis2 = tf.transpose(dis1,[0,2,1,3])
        res1 = tf.reduce_min(dis1, 3)
        res2 = tf.reduce_min(dis2, 3)
        res1 = tf.reduce_min(res1, 2)
        res2 = tf.reduce_min(res2, 2)
        outputs['loss1'] = tf.reduce_mean(res1)
        outputs['loss2'] = tf.reduce_mean(res2)
        loss = outputs['loss1'] + outputs['loss2']
        loss = tf.cast(loss, 'float32')
        return loss * weight_scale   

    def add_interpolation_kl_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pixel = outputs['coord_pixels']
        pred = outputs['test_o']
        gt = inputs['inpoints']  
        pred = pred / 128.0
        gt = gt / 128.0
        dis = euclidean_distance_for_two_points(gt, pred)
        min_dis = tf.reduce_min(dis, 2)
        exp_dis = tf.exp(-dis/0.1)
        guiyihua_dis = self.guiyihua(exp_dis)
        gt_pixel = tf.matmul(guiyihua_dis, pixel)
        res = tf.reduce_mean(1-gt_pixel, 1)
        
        ones = tf.ones_like(guiyihua_dis) / cfg.pc_num_points
        mat_sum = tf.expand_dims(tf.reduce_sum(gt_pixel, 1), 1)
        gt_pixel_gyh = tf.divide(gt_pixel, mat_sum)
        kl = tf.reduce_sum(gt_pixel_gyh * (tf.log(gt_pixel_gyh + 1e-8) - tf.log(ones)), 1)
        loss_1 = 1 * tf.reduce_mean(res)
        loss_3 = 1000 * tf.reduce_mean(min_dis)
        loss_2 = 100 * tf.reduce_mean(1-pixel)
        outputs['loss2'] = 1-pixel
        return loss_1, loss_2, loss_3
   
    def add_another_cd_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pred = outputs['test_o']
        gt = inputs['inpoints']
        pred = pred / 128.0
        gt = gt / 128.0
        dis = euclidean_distance_for_two_points(pred, gt)
        res1 = tf.reduce_min(dis, 1)
        res2 = tf.reduce_min(dis, 2)
        loss1 = 1 * tf.reduce_mean(res1) 
        loss2 = 1 * tf.reduce_mean(res2)
        return loss1, loss2
    
    def add_topk_cd_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pred = outputs['test_o']
        gt = inputs['inpoints']
        print(pred.shape, gt.shape)
        pred = pred / 128.0
        gt = gt / 128.0
        mindis1 = chamfer_distance_topk(gt, pred, cfg.gt_topk)
        mindis2 = chamfer_distance_topk(pred, gt, cfg.pred_topk)
        loss1 = tf.reduce_mean(mindis1)
        loss2 = tf.reduce_mean(mindis2)
        return loss1, loss2

    def get_loss(self, inputs, outputs, add_summary=True):
        cfg = self.cfg()
        g_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.proj_weight:
            proj_loss = self.add_proj_loss(inputs, outputs, cfg.proj_weight, add_summary)
            g_loss += proj_loss
        else:
            proj_loss = tf.zeros(dtype=tf.float32, shape=[])
        
        if cfg.sum_weight:
            sum_loss = self.add_sum_entropy_distance_loss(inputs, outputs, cfg.sum_weight, add_summary)
            g_loss += sum_loss
        else:
            sum_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.thermodynamic_weight:
            thermodynamic_loss = self.add_thermodynamic_loss(inputs, outputs, cfg.thermodynamic_weight, add_summary)
            g_loss += thermodynamic_loss
        else:
            thermodynamic_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.classify_weight:
            class_loss = self.add_classify_loss(inputs, outputs, cfg.classify_weight, add_summary)
            g_loss += class_loss
        else:
            class_loss = tf.zeros(dtype=tf.float32, shape=[])
        
        if cfg.opt_weight:
            opt_loss = self.add_opt_loss(inputs, outputs, cfg.opt_weight, add_summary)
            g_loss += opt_loss
        else:
            opt_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.flow_weight:
           flow_loss = self.add_flowing_loss(inputs, outputs, cfg.flow_weight, add_summary)
           g_loss += flow_loss
        else:
           flow_loss = tf.zeros(dtype=tf.float32, shape=[])
 
        if cfg.cd_weight:
            cd1_loss, cd2_loss = self.add_another_cd_loss(inputs, outputs, cfg.cd_weight, add_summary)
            g_loss += cd1_loss * 100
            g_loss += cd2_loss * 100
        else:
            cd1_loss = tf.zeros(dtype=tf.float32, shape=[])
            cd2_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.emd_weight:
            emd_loss = self.add_emd_loss(inputs, outputs, cfg.emd_weight, add_summary)        
            g_loss += emd_loss
        else:
            emd_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.topk_cd_weight:
            topk_cd_loss = self.add_topk_cd_loss(inputs, outputs, cfg.topk_cd_weight, add_summary)
            g_loss += topk_cd_loss
        else:
            topk_cd_loss = tf.zeros(dtype=tf.float32, shape=[])
       
        if cfg.cos_weight:
            cos_loss = self.add_cosine_loss(inputs, outputs, cfg.cos_weight, add_summary)
            g_loss += cos_loss
        else:
            cos_loss = tf.zeros(dtype=tf.float32, shape=[])
        
        if cfg.exp_weight:
            e_loss = self.add_exp_loss(inputs, outputs, cfg.exp_weight, add_summary)
            g_loss += e_loss
        else:
            e_loss = tf.zeros(dtype=tf.float32, shape=[])
        
        if cfg.df_weight:
            df_loss = self.add_df_loss(inputs, outputs, cfg.df_weight, add_summary)
            g_loss += df_loss
        else:
            df_loss = tf.zeros([])

        if cfg.fuzz_weight:
            fuzz_loss = self.add_fuzz_loss(inputs, outputs, cfg.fuzz_weight, add_summary)
            g_loss += fuzz_loss
        else:
            fuzz_loss = tf.zeros(dtype=tf.float32, shape=[])
        
        if cfg.kl_weight:
            kl_loss = self.add_kl_divergence_loss(inputs, outputs, cfg.kl_weight, add_summary)
            g_loss += kl_loss
        else:
            kl_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.interpolation_kl_weight:
            inter_loss1, inter_loss2, inter_loss3 = self.add_interpolation_kl_loss(inputs, outputs, cfg.interpolation_kl_weight, add_summary)
            g_loss += inter_loss1
            g_loss += inter_loss2
            g_loss += inter_loss3
        else:
            inter_loss1 = tf.zeros(dtype=tf.float32, shape=[])
            inter_loss2 = tf.zeros(dtype=tf.float32, shape=[])
            inter_loss3 = tf.zeros(dtype=tf.float32, shape=[])
        
        return g_loss, cd1_loss, cd1_loss, cd2_loss
