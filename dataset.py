import os
import time
import random
import numpy as np
import tensorflow as tf
from config import Config
from sklearn.model_selection import train_test_split

def load_imgpaths(dir_path, split=False):

	files = os.listdir(dir_path)
	paths = [dir_path + file for file in files]
	random.shuffle(paths)
	if split:
		train_paths, eval_paths = \
		train_test_split(paths, test_size=0.05, random_state=42)
		return train_paths, eval_paths
	return paths

def load_and_preprocess(img_path):

    img_str = tf.io.read_file(img_path)
    img = tf.image.decode_and_crop_jpeg(img_str, Config.data_crop, channels=3)
    return img

def write_tfrecord(img_paths, tfrec_path):

	path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
	image_ds = path_ds.map(load_and_preprocess, num_parallel_calls =8) # Num_parallel_calls = no.of.cpu_cores
	proto_ds = image_ds.map(tf.io.serialize_tensor)
	tfrec = tf.data.experimental.TFRecordWriter(tfrec_path)
	tfrec_op = tfrec.write(proto_ds)

	with tf.compat.v1.Session() as sess:
		sess.run(tfrec_op)

def prepare_tfrecords():

	print("Preparing tfrecords")
	paths = load_imgpaths(Config.data_dir_path)
	if len(paths)==2:
		write_tfrecord(paths[0], Config.tfrecord_dir+'train.tfrecord')
		write_tfrecord(paths[1], Config.tfrecord_dir+'test.tfrecord')
	else:
		write_tfrecord(paths, Config.tfrecord_dir+'train.tfrecord')

def parse_fn(tfrecord):

	result_img = tf.io.parse_tensor(tfrecord, out_type=tf.uint8)
	result_img = tf.reshape(result_img, Config.img_shape)
	return result_img

def prepare_dataset(tfrecord_file):

	dataset = tf.data.TFRecordDataset(tfrecord_file)
	dataset = dataset.map(map_func=parse_fn, num_parallel_calls=8)
	#dataset = dataset.shuffle(buffer_size=Config.total_training_imgs, reshuffle_each_iteration=False)
	dataset = dataset.batch(batch_size=Config.batch_size, drop_remainder=True)
	dataset = dataset.repeat(Config.num_epochs)
	dataset = dataset.prefetch(buffer_size = 1)
	return dataset

def main():
	start_time = time.time()
	prepare_tfrecords()
	end_time = time.time()
	print(end_time-start_time)
if __name__ == '__main__':
	main()
