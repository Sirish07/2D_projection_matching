import tensorflow as tf
import tensorflow.contrib.slim as slim


def model(inputs, outputs_all, cfg, is_training):
        # Source : https://github.com/chenhsuanlin/3D-point-cloud-generation/blob/master/graph.py
        act_fn = tf.nn.leaky_relu
        # print("Decoder Outputs")
        with slim.arg_scope(
                [slim.conv2d_transpose, slim.fully_connected],
                weights_initializer=tf.contrib.layers.variance_scaling_initializer()):

                hf = slim.fully_connected(inputs, 1024, activation_fn=act_fn)
                hf = slim.fully_connected(hf, 2048, activation_fn=act_fn)
                hf = slim.fully_connected(hf, 4096, activation_fn=act_fn)
                hf = tf.reshape(hf,[cfg.batch_size,4,4,-1])
                feat = slim.conv2d_transpose(hf, 128, 3, 2, activation_fn=act_fn)
                feat = slim.conv2d_transpose(feat, 64, 3, 2, activation_fn=act_fn)
                feat = slim.conv2d_transpose(feat, 32, 3, 2, activation_fn=act_fn)
                feat = slim.conv2d_transpose(feat, 3, 3, 2, activation_fn=act_fn)

        out = dict()
        out["xyz"] = feat
        out["rgb"] = None
        return out