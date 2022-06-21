import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def model(inputs, cfg, is_training):
        # Source : https://github.com/chenhsuanlin/3D-point-cloud-generation/blob/master/graph.py
        act_fn = tf.nn.relu
        view_per_image = 1
        with slim.arg_scope(
                [slim.conv2d_transpose, slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                normalizer_fn=tf.layers.batch_normalization, 
                normalizer_params={'training': is_training, 'momentum': 0.95},
                activation_fn=act_fn):

                hf = act_fn(inputs)
                hf = slim.fully_connected(hf, 1024)
                hf = slim.fully_connected(hf, 2048)
                hf = slim.fully_connected(hf, 4096)
                hf = tf.reshape(hf,[cfg.step_size,4,4,-1]) # [B, 4, 4, 256]
                feat = slim.conv2d_transpose(hf, 192, 3, 2) # [B, 8, 8, 192]
                feat = slim.conv2d_transpose(feat, 128, 3, 2) # [B, 16, 16, 128]
                feat = slim.conv2d_transpose(feat, 96, 3, 2) # [B, 32, 32, 96]
                feat = slim.conv2d_transpose(feat, 64, 9, 1, padding = 'VALID') # [B, 40, 40, 64]
                with tf.variable_scope("pixelconv"):
                        feat = pixelconv2Layer(cfg, feat, view_per_image*4) # [B, 40, 40, 4]
                XYZ,_ = tf.split(feat,[view_per_image*3,view_per_image],axis=3) # [B,H,W,3V],[B,H,W,V]
                XYZ = tf.reshape(XYZ, [cfg.batch_size, cfg.image2pc_dim * cfg.image2pc_dim * cfg.step_size, 3])
                
                XYZ = tf.tanh(XYZ)
                if cfg.pc_unit_cube:
                    XYZ = XYZ / 2.0 

        out = dict()
        out["xyz"] = XYZ
        out["rgb"] = None
        return out

def pixelconv2Layer(cfg,feat,outDim):
		weight,bias = createVariable(cfg,[1,1,int(feat.shape[-1]),outDim],gridInit=True)
		conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
		return conv

def createVariable(cfg,weightShape,biasShape=None,stddev=None,gridInit=False):
	view_per_image = 1
	renderDepth = 1.0
	if biasShape is None:
		biasShape = [weightShape[-1]]
	weight = tf.Variable(tf.random_normal(weightShape,stddev=0.1),dtype=np.float32,name="weight")
	if gridInit:
		X,Y = np.meshgrid(range(cfg.image2pc_dim),range(cfg.image2pc_dim),indexing="xy") # [H,W]
		X,Y = X.astype(np.float32),Y.astype(np.float32)
		initTile = np.concatenate([np.tile(X,[view_per_image,1,1]),
								   np.tile(Y,[view_per_image,1,1]),
								   np.ones([view_per_image,cfg.image2pc_dim,cfg.image2pc_dim],dtype=np.float32)*renderDepth,
								   np.zeros([view_per_image,cfg.image2pc_dim,cfg.image2pc_dim],dtype=np.float32)],axis=0) # [4V,H,W]
		biasInit = np.expand_dims(np.transpose(initTile,axes=[1,2,0]),axis=0) # [1,H,W,4V]
	else:
		biasInit = tf.constant(0.0,shape=biasShape)
	bias = tf.Variable(biasInit,dtype=np.float32,name="bias")
	return weight,bias