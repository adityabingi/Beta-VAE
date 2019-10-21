class Config:

	# dataset preparation config for CelebA dataset
	data_dir_path = 'celebA/img_align_celeba/'
	tfrecord_dir = 'tfrecords/'  # dir where tfrecords are saved
	data_crop = [57, 21, 128, 128] # [crop-y(top-left y), crop-x(top-left x), crop-height, crop-width] (128 x 128 resolution images are taken)
	img_shape = (128, 128, 3) # image shape for Variational Autoencoder


	# batch_norm layer parameters
	momentum = 0.99
	epsilon  = 1e-5

	# VAE architecture 
	latent_dim = 128
	filters = [64, 128, 256, 512]
	last_convdim = int(img_shape[0]/(2**len(filters))) # images are downsampled to (8*8*512) before dense layer

	#training parameters
	initial_learning_rate = 0.0005

	kl_weight = 0.01

	total_training_imgs = 202599  # Total uncorrupted images for CelebA
	batch_size = 32     # Configure it based on available GPU memory
	num_epochs = 30
	image_snapshot_freq = 500  # Number of batches shown in between image_grid snapshots


	#results
	modelDir = 'model/'
	summaryDir = 'summaries/run1/'
	num_gen_imgs = 32   # number of images to generate
	grid_size = (4, 8)  # results are saved to an image grid of this size 
	results_dir = 'results/'
