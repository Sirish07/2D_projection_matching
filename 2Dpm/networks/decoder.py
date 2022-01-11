import tensorflow as tf
import tensorflow.contrib.slim as slim


def model(inputs, cfg, is_training):
        # Source : https://github.com/chenhsuanlin/3D-point-cloud-generation/blob/master/graph.py
        act_fn = tf.nn.relu
        with slim.arg_scope(
                [slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                normalizer_fn=slim.batch_norm, 
                normalizer_params={'is_training': is_training},
                activation_fn=act_fn):

                hf = slim.fully_connected(inputs, 1024)
                hf = slim.fully_connected(hf, 2048)
                hf = slim.fully_connected(hf, 4096)
                hf = tf.reshape(hf,[cfg.batch_size,4,4,-1]) # [B, 4, 4, 256]
                feat = slim.conv2d_transpose(hf, 192, 3, 2) # [B, 8, 8, 192]
                feat = slim.conv2d_transpose(feat, 128, 3, 2) # [B, 16, 16, 128]
                feat = slim.conv2d_transpose(feat, 96, 3, 2) # [B, 32, 32, 96]
                feat = slim.conv2d_transpose(feat, 64, 17, 1, padding = 'VALID') # [B, 48, 48, 64]
                feat = slim.conv2d(feat, 3, 3, 1, padding = "SAME") # [B, 48, 48, 3]

        out = dict()
        out["xyz"] = feat
        out["rgb"] = None
        return out