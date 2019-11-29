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

def traverse_latents(sess, vae):

	random_seeds = [42, 62, 1024, 72, 92] # These random seeds are chosen after visual inspection 
					      # Seeds can also be chosen randomly
	seed_zs =[]

	for seed in random_seeds:
		np.random.seed(seed)
		random_img_z = np.random.normal(size=[vae.latent_dim])
		seed_z = np.float32(random_img_z)
		seed_zs.append(seed_z)

	seed_zs = np.array(seed_zs)
	
	for dim in range(vae.latent_dim):
		interpolation = np.linspace(start=-3.0, stop=3.0, num=10)
		traversal_vectors = np.zeros([seed_zs.shape[0]*interpolation.shape[0], seed_zs.shape[1]])
		for i in range(seed_zs.shape[0]):
			z_s = seed_zs.copy()
			one_seed_z = z_s[i]
			traversal_vectors[i*interpolation.shape[0]:(i+1)*interpolation.shape[0]] = one_seed_z
			traversal_vectors[i*interpolation.shape[0]:(i+1)*interpolation.shape[0], dim] = interpolation
			
		traversal_vectors =np.float32(traversal_vectors)
		latent_imgs = vae.decode(inputs=traversal_vectors)
		traverse_imgs = sess.run(latent_imgs)
		save_image_grid(traverse_imgs, Config.results_dir+'traverse_latentdim{}.jpg'.format(dim), grid_size=(5, 10))

def make_session():

	config = tf.compat.v1.ConfigProto(allow_soft_placement=True) #log_device_placement=True)
	config.gpu_options.allow_growth = True
	sess  = tf.compat.v1.Session(config=config)
	return sess

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train variational autoencoder', action='store_true')
	parser.add_argument('--generate',help='generate images', action='store_true')
	parser.add_argument('--traverse',help='traverse latent space', action='store_true')
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
			traverse_latents(sess, vae)

if __name__=='__main__':
	main()
