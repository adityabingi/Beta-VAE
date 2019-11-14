import argparse
import numpy as np
import tensorflow as tf

from utils import *
from model import VAE
from config import Config
from dataset import prepare_tfrecords, prepare_dataset, load_and_preprocess

def lr_schedule(epoch, previous_lr):
	if (epoch % 8)==0:
		new_lr = previous_lr/4
	else:
		new_lr = previous_lr
	return new_lr

def train(sess, vae):

	random_vector = tf.random.normal(shape=[Config.num_gen_imgs, vae.latent_dim], seed=82)

	if not vae.restore:
		sess.run(vae.init)
		print("Initialized with new values")

	train_writer = tf.compat.v1.summary.FileWriter(Config.summaryDir+'train', sess.graph)

	num_epochs = Config.num_epochs
	batches_per_epoch =int(Config.total_training_imgs / Config.batch_size)
	batch_count = 0
	lr = Config.initial_learning_rate

	for epoch in range(1, num_epochs+1):
		#lr = lr_schedule(epoch, previous_lr = lr)
		reals, reconstructed = [], []
		print("At Epoch {}".format(epoch))
		print("------------------------------------------")
		
		for batch in range(batches_per_epoch):
			batch_count+=1
			_, loss_vals, real, recon, mean_var_summ = sess.run([vae.train_op, vae.loss_values,\
														vae.real_img, vae.recon_img, vae.mean_var_summary],\
														feed_dict={vae.learning_rate_ph:lr})

			loss_summ = sess.run(vae.loss_summary, feed_dict={vae.total_loss_ph:loss_vals[0], \
									vae.rec_loss_ph:loss_vals[1], vae.kl_loss_ph:loss_vals[2]})
			train_writer.add_summary(loss_summ, batch_count)
			train_writer.add_summary(mean_var_summ, batch_count)

			print("Total loss at {}/{} is {:0.3f}".format(batch+1, batches_per_epoch, loss_vals[0]))
			if(batch<32):
				reals.append(real)
				reconstructed.append(recon)

			if(batch % Config.image_snapshot_freq==0):
				vae.training = False # for batch-norm layer during genertation of fakes
				gen_filename = Config.results_dir + 'fakes_epoch{:02d}_batch{:05d}.jpg'.format(epoch, batch)
				generate_fake_images(sess, vae, random_vector, gen_filename)
				vae.training = True
				
	
		vae.global_step+=epoch
		vae.saver.save(sess, Config.modelDir+'snapshot', global_step = vae.global_step)

		rec_filename = Config.results_dir + 'reconstructed_epoch{:03d}.jpg'.format(epoch)
		save_image_grid(np.array(reconstructed), rec_filename, drange=[0,1], grid_size=Config.grid_size)

	input_filename = Config.results_dir + 'input_images.jpg'
	save_image_grid(np.array(reals), input_filename, drange=[0,1], grid_size=Config.grid_size)

	make_training_gif()

def generate_fake_images(sess, vae, random_vector, filename):

	gen_fakes= vae.decode(inputs = random_vector)
	fake_images = sess.run(gen_fakes)
	save_image_grid(fake_images, filename, drange=[0,1], grid_size=Config.grid_size)

def traverse_latents(sess, vae, target_dim, filename):

	np.random.seed(183)
	random_img_z = np.random.normal(size=[1, vae.latent_dim])
	seed_z = np.float32(random_img_z)

	"""seed_img = load_and_preprocess(Config.data_dir_path+'000008.jpg')
	seed_img = tf.expand_dims(seed_img, axis=0)
	seed_img = tf.cast(seed_img, tf.float32)
	inputs = adjust_data_range(seed_img, drange_in=[0,255], drange_out=[0,1])

	seed_img_z, seed_img_z_logvar = vae.encode(inputs)
	#seed_sample_z = vae.sample(seed_img_z, seed_img_z_logvar)
	seed_z = sess.run(seed_img_z)"""

	interpolation = np.linspace(start=-3.0, stop=3.0, num=10)
	z = seed_z.copy()
	traversal_vector = np.zeros([interpolation.shape[0], vae.latent_dim])
	for i in range(interpolation.shape[0]):
		z_mean = z.copy()
		z_mean[:,target_dim] = interpolation[i]
		#print(z_mean)
		traversal_vector[i] = z_mean[0]

	print(traversal_vector)
	traversal_vector=np.float32(traversal_vector)
	latent_img = vae.decode(inputs=traversal_vector)
	traverse_imgs = sess.run(latent_img)

	save_image_grid(traverse_imgs, filename, grid_size=(1, 10))

def make_session():

	config = tf.compat.v1.ConfigProto(allow_soft_placement=True) #log_device_placement=True)
	config.gpu_options.allow_growth = True
	sess  = tf.compat.v1.Session(config=config)
	return sess

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train variational autoencoder', action='store_true')
	parser.add_argument('--generate',help='generate images', action='store_true')
	parser.add_argument('--traverse', nargs=1, help='traverse latent space in specified dimension', type=int)
	args = parser.parse_args()

	sess = make_session()

	if(args.train):

		#prepare_tfrecords()
		training_dataset = prepare_dataset(Config.tfrecord_dir+'train.tfrecord')
		iterator = tf.compat.v1.data.make_one_shot_iterator(training_dataset)
		vae = VAE(sess, training=True, training_iterator = iterator)
		train(sess, vae)

	if(args.generate) or (args.traverse):
		
		vae = VAE(sess, training=False, restore=True)

		if(args.generate):
			random_vector = tf.random.normal(shape=[Config.num_gen_imgs, vae.latent_dim])
			filename = Config.results_dir + 'randomFakeGrid.jpg'
			gen_imgs = generate_fake_images(sess, vae, random_vector, filename)
			

		if(args.traverse):
			filename = Config.results_dir + 'traverse_latent.jpg'
			traverse_latents(sess, vae, target_dim=args.traverse[0], filename =filename)

if __name__=='__main__':
	main()
