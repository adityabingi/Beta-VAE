import tensorflow as tf
from config import Config
from utils import adjust_data_range

class VAE:

	def __init__(self, sess, training=True, restore=False, training_iterator=None):


		self.sess = sess
		self.training = training
		self.restore = restore

		self.latent_dim = Config.latent_dim
		self.filters = Config.filters
		self.last_convdim = Config.last_convdim  # images are downsampled to (8*8*512) for celeba before dense layer
		self.kl_weight = Config.kl_weight

		if training_iterator:
			input_imgs = training_iterator.get_next()
			cast_imgs = tf.cast(input_imgs, tf.float32)
			self.real_imgs = adjust_data_range(cast_imgs, drange_in=[0,255], drange_out=[0,1])

		else:
			self.real_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None,)+Config.img_shape)

		self.recon_imgs, self.mean_var_summary = self.forward_pass(inputs=self.real_imgs)
		# for visualizing training
		self.real_img, self.recon_img = self.real_imgs[0], self.recon_imgs[0]
		self.loss_values= self.compute_loss()
		total_loss = self.loss_values[0]
		self.learning_rate_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate_ph,\
													beta1=Config.beta1, beta2=Config.beta2)
		optimize_op = optimizer.minimize(total_loss)
		update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
		self.train_op = tf.group(optimize_op, update_ops)

		self.loss_summary = self.loss_summaries()

		self.global_step=0
		self.init = tf.compat.v1.global_variables_initializer()
		self.saver = tf.compat.v1.train.Saver(max_to_keep=3)

		if self.restore:
			latestSnapshot = tf.train.latest_checkpoint(Config.modelDir)
			if not latestSnapshot:
				raise Exception('No saved model found in: ' + Config.modelDir)
			self.saver.restore(self.sess, latestSnapshot)
			print("Restored saved model from latest snapshot")
			self.global_step = int(latestSnapshot.split('-')[1].split('.')[0])

	def _batch_norm(self, inputs):
		"""Performs a batch normalization using a standard set of parameters."""
		# Set fused=True for a significant performance boost. See
		# https://www.tensorflow.org/performance/performance_guide#common_fused_ops
		
		return tf.compat.v1.layers.batch_normalization(
			inputs=inputs, axis=-1, momentum=Config.momentum, epsilon=Config.epsilon, center=True,
			scale=True, training=self.training, fused=True)

	def _conv_block(self, inputs, n_filters):

		w_init = tf.compat.v1.initializers.he_normal(seed=None)
		b_init = tf.constant_initializer(0.0)
		with tf.name_scope('conv_block'):
			conv = tf.compat.v1.layers.conv2d(inputs,filters=n_filters,kernel_size=(5,5),strides=(2, 2),
						bias_initializer=b_init,kernel_initializer=w_init)
			batch_norm = self._batch_norm(conv)
			outputs = tf.nn.relu(batch_norm)
		return outputs

	def _deconv_block(self, inputs, n_filters):

		w_init = tf.compat.v1.initializers.he_normal(seed=None)
		b_init = tf.constant_initializer(0.0)
		with tf.name_scope('deconv_block'):
			deconv = tf.compat.v1.layers.conv2d_transpose(inputs,filters=n_filters,kernel_size=(5,5),
						strides=(2,2),padding = 'same')
			batch_norm = self._batch_norm(deconv)
			outputs = tf.nn.relu(batch_norm)
		return outputs

	def encode(self, inputs):

		with tf.compat.v1.variable_scope('Encoder', reuse=tf.compat.v1.AUTO_REUSE):
			for i in range(len(self.filters)):
				inputs = self._conv_block(inputs, self.filters[i])
			flatten  = tf.compat.v1.layers.flatten(inputs)
			dense1  = tf.compat.v1.layers.dense(flatten, units=self.latent_dim)
			z_mean  = self._batch_norm(dense1)
			dense2 = tf.compat.v1.layers.dense(flatten, units=self.latent_dim)
			z_logvar = self._batch_norm(dense2)

		return z_mean, z_logvar

	def decode(self, inputs):

		with tf.compat.v1.variable_scope('Decoder', reuse=tf.compat.v1.AUTO_REUSE):

			inputs = tf.compat.v1.layers.dense(inputs, units=self.last_convdim*self.last_convdim*self.filters[-1])
			inputs = self._batch_norm(inputs)
			inputs = tf.reshape(inputs, [-1, self.last_convdim, self.last_convdim, self.filters[-1]])
			for i in range(len(self.filters)-1, -1, -1):
				inputs = self._deconv_block(inputs, self.filters[i])
			output = tf.compat.v1.layers.conv2d_transpose(inputs, filters=3, kernel_size=(5,5),
						strides=(1,1), padding='same')
			output = tf.nn.sigmoid(output)

		return output

	def sample(self, mean, logvar):

		"""Sample from Gaussian Distribution """
		with tf.name_scope('reparameterize'):
			eps = tf.random.normal(shape =tf.shape(mean))
			z = mean + tf.exp(0.5*logvar) * eps
		return z

	def forward_pass(self, inputs):

		with tf.name_scope('VAE'):

			self.z_mean, self.z_logvar = self.encode(inputs)
			mean_summary = tf.compat.v1.summary.histogram('z_mean_summary', self.z_mean)
			variance_summary = tf.compat.v1.summary.histogram('z_logvar_summary', self.z_logvar)
			mean_and_var_summary = tf.compat.v1.summary.merge([mean_summary, variance_summary])
			z = self.sample(self.z_mean, self.z_logvar)	
			output = self.decode(z)  # z - sampled latent vector

		return output, mean_and_var_summary

	def compute_loss(self):

		N = Config.img_shape[0] * Config.img_shape[1] * Config.img_shape[2]
		m = self.latent_dim
		with tf.name_scope('Loss'):

			rec_loss =  tf.reduce_sum(tf.square(self.real_imgs - self.recon_imgs))
			rec_loss = rec_loss/N
			rec_los  = rec_loss/Config.batch_size
			kl_loss  =  tf.reduce_sum(- 0.5 * tf.reduce_sum(1 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar),1))
			kl_loss = kl_loss/m
			kl_loss = kl_loss/Config.batch_size
			total_loss = rec_loss + self.kl_weight * kl_loss

		return [total_loss, rec_loss, kl_loss]

	#summary functions
	def loss_summaries(self):

		self.rec_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
		self.kl_loss_ph  = tf.compat.v1.placeholder(tf.float32, shape=None)
		self.total_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
		rec_loss_summ = tf.compat.v1.summary.scalar('reconstruction_loss', self.rec_loss_ph)
		kl_loss_summ = tf.compat.v1.summary.scalar('KL loss', self.kl_loss_ph)
		total_loss_summ = tf.compat.v1.summary.scalar('total loss', self.total_loss_ph)
		loss_summary=tf.compat.v1.summary.merge([rec_loss_summ, kl_loss_summ, total_loss_summ])
		return loss_summary