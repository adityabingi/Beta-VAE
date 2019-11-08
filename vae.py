import argparse
import numpy as np
import tensorflow as tf
from config import Config
from dataset import prepare_tfrecords, prepare_dataset

from utils import *

class VAE:

	def __init__(self, training=True):

		self.latent_dim = Config.latent_dim
		self.filters = Config.filters
		self.last_convdim = Config.last_convdim  # images are downsampled to (4*4*1024) for celeba before dense layer
		self.training = training
		self.kl_weight = Config.kl_weight

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

	def _encode(self, inputs):

		with tf.compat.v1.variable_scope('Encoder'):
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

	def _sample(self, mean, logvar):

		"""Sample from Gaussian Distribution """
		with tf.name_scope('reparameterize'):
			eps = tf.random.normal(shape =tf.shape(mean))
			z = mean + tf.exp(0.5*logvar) * eps
		return z

	def forward_pass(self, inputs):

		with tf.name_scope('VAE'):
			self.z_mean, self.z_logvar = self._encode(inputs)
			mean_summary = tf.compat.v1.summary.histogram('z_mean_summary', self.z_mean)
			variance_summary = tf.compat.v1.summary.histogram('z_logvar_summary', self.z_logvar)
			mean_and_var_summary = tf.compat.v1.summary.merge([mean_summary, variance_summary])
			z = self._sample(self.z_mean, self.z_logvar)	
			output = self.decode(z)  # z - sampled latent vector

		return output, mean_and_var_summary

	def compute_loss(self, reals, recon_imgs):
		N = Config.img_shape[0] * Config.img_shape[1] * Config.img_shape[2]
		m = self.latent_dim
		with tf.name_scope('Loss'):
			rec_loss =  tf.reduce_sum(tf.square(reals - recon_imgs))
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

def lr_schedule(epoch, previous_lr):
	if (epoch % 8)==0:
		new_lr = previous_lr/ 4
	else:
		new_lr = previous_lr
	return new_lr
def train(sess, vae, dataset):

	iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
	input_imgs = iterator.get_next()
	cast_imgs = tf.cast(input_imgs, tf.float32)
	processed_imgs = adjust_data_range(cast_imgs, drange_in=[0,255], drange_out=[0,1])

	real_img = processed_imgs[0]  # for visualizing training 
	recon_imgs, mean_var_summary = vae.forward_pass(inputs=processed_imgs)
	recon_img = recon_imgs[0] # for visualizing training
	loss_values= vae.compute_loss(reals=processed_imgs , recon_imgs=recon_imgs)
	total_loss = loss_values[0]
	learning_rate_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate_ph, beta1=Config.beta1, beta2=Config.beta2)
	optimize_op = optimizer.minimize(total_loss)
	update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
	train_op = tf.group(optimize_op, update_ops)

	loss_summary = vae.loss_summaries()
	random_vector = tf.random.normal(shape=[Config.num_gen_imgs, vae.latent_dim])

	init  = tf.compat.v1.global_variables_initializer()
	saver = tf.compat.v1.train.Saver(max_to_keep=3)
	sess.run(init)
	print("Initialized with new values")
	train_writer = tf.compat.v1.summary.FileWriter(Config.summaryDir+'train', sess.graph)

	num_epochs = Config.num_epochs
	batches_per_epoch =int(Config.total_training_imgs / Config.batch_size)
	batch_count = 0
	learning_rate = Config.initial_learning_rate
	for epoch in range(1, num_epochs+1):
		#learning_rate = lr_schedule(epoch, previous_lr = learning_rate)
		reals, reconstructed = [], []
		print("At Epoch {}".format(epoch))
		print("------------------------------------------")
		
		for batch in range(batches_per_epoch):
			batch_count+=1
			_, loss_vals, real, recon, mean_var_summ = sess.run([train_op, loss_values, real_img, recon_img, mean_var_summary], \
															feed_dict={learning_rate_ph:learning_rate})

			loss_summ = sess.run(loss_summary, feed_dict={vae.total_loss_ph:loss_vals[0], \
									vae.rec_loss_ph:loss_vals[1], vae.kl_loss_ph:loss_vals[2]})
			train_writer.add_summary(loss_summ, batch_count)
			train_writer.add_summary(mean_var_summ, batch_count)

			print("Total loss at {}/{} is {:0.3f}".format(batch+1, batches_per_epoch, loss_vals[0]))
			if(batch<32):
				reals.append(real)
				reconstructed.append(recon)

			if(batch % Config.image_snapshot_freq==0):
				vae.training = False # for batch-norm layer during genertation of fakes
				fakes = generate_fake_images(sess, vae, random_vector)
				vae.training = True
				gen_filename = Config.results_dir + 'fakes_epoch{:02d}_batch{:05d}.jpg'.format(epoch, batch)
				save_image_grid(fakes, gen_filename, drange=[0,1], grid_size=Config.grid_size)

		saver.save(sess, Config.modelDir+'snapshot', global_step = epoch)
		rec_filename = Config.results_dir + 'reconstructed_epoch{:03d}.jpg'.format(epoch)
		save_image_grid(np.array(reconstructed), rec_filename, drange=[0,1], grid_size=Config.grid_size)

	input_filename = Config.results_dir + 'input_images.jpg'
	save_image_grid(np.array(reals), input_filename, drange=[0,1], grid_size=Config.grid_size)
	make_training_gif(Config.results_dir)

def generate_fake_images(sess, vae, random_vector, restore=False):

	gen_fakes= vae.decode(inputs = random_vector)
	if restore:
		saver = tf.compat.v1.train.Saver()
		latestSnapshot = tf.train.latest_checkpoint(Config.modelDir)
		if not latestSnapshot:
			raise Exception('No saved model found in: ' + Config.modelDir)
		saver.restore(sess, latestSnapshot)
		print("Restored saved model from latest snapshot")

	fake_images = sess.run(gen_fakes)

	return fake_images

def make_session():

	config = tf.compat.v1.ConfigProto(allow_soft_placement=True) #log_device_placement=True)
	config.gpu_options.allow_growth = True
	sess  = tf.compat.v1.Session(config=config)
	return sess

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train variational autoencoder', action='store_true')
	parser.add_argument('--generate',help='generate images', action='store_true')
	args = parser.parse_args()

	sess = make_session()

	if(args.train):
		vae = VAE(training=True)
		#prepare_tfrecords()
		training_dataset = prepare_dataset(Config.tfrecord_dir+'train.tfrecord')
		train(sess, vae, training_dataset)
	if(args.generate):
		vae = VAE(training=False)
		random_vector = tf.random.normal(shape=[Config.num_gen_imgs, vae.latent_dim])
		gen_imgs = generate_fake_images(sess, vae, random_vector, restore=True)
		for i, img in enumerate(gen_imgs):
			filename = Config.results_dir + 'randomFake_{:03d}'.format(i)
			save_image(img, filename, drange=[0,1])
		filename = Config.results_dir + 'randomFakeGrid'
		save_image_grid(gen_imgs, filename, drange=[0,1], grid_size=Config.grid_size)


if __name__=='__main__':
	main()
