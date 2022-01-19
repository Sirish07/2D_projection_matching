import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1


def model(images, cfg, is_training):
    """Model encoding the images into view-invariant embedding."""
    image_size = images.get_shape().as_list()[1]
    target_spatial_size = 4

    f_dim = cfg.f_dim
    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim
    encout_dim = cfg.encout_dim # 512
    outputs = dict()

    act_func = tf.nn.relu

    images = _preprocess(images)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={'training': is_training, 'momentum':0.95},
            activation_fn=act_func):
        
        batch_size = images.shape[0]
        hf = slim.conv2d(images, f_dim, [5, 5], stride=2)
        num_blocks = int(np.log2(image_size / target_spatial_size) - 1)

        for k in range(num_blocks):
            f_dim = f_dim * 2
            hf = slim.conv2d(hf, f_dim, [3, 3], stride=2)
            hf = slim.conv2d(hf, f_dim, [3, 3], stride=1)

        # Reshape layer
        rshp0 = tf.reshape(hf, [batch_size, -1])
        outputs["conv_features"] = rshp0
        fc1 = slim.fully_connected(rshp0, fc_dim)
        fc2 = slim.fully_connected(fc1, fc_dim)
        fc3 = slim.fully_connected(fc2, z_dim)
        fc4 = slim.fully_connected(fc3, encout_dim, normalizer_fn=None, activation_fn=None) # encOut 512 dim a/c to EPCG, No activation func for final layer 

        outputs["z_latent"] = fc1 # [B, 1024]
        outputs['ids'] = fc3 # [B, 1024]
        outputs["encoderOut"] = fc4 # [B, 512]
        if cfg.predict_pose:
            outputs['poses'] = slim.fully_connected(fc2, z_dim)
    return outputs

