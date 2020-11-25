import re
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras import backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from classifier import plot_loss

def convolutionalAutoencoder(filter_pixel=3, filters=256, layers_=5):
	print("[INFO] building autoencoder...")
	counter = 0
	kernel_s = (filter_pixel, filter_pixel)

	input_img = keras.Input(shape=(28, 28, 1))
	filters_list = []
	filters_list.append(filters)
	for i in range(0, layers_ - 1):
		filters_list.append(filters_list[i] // 2)

	x = input_img
	for f in filters_list[::-1]:
		counter += 1
		x = layers.Conv2D(f, kernel_s, activation='relu', padding='same')(x)
		x = layers.BatchNormalization()(x)
		if counter%2 == 0:
			x = layers.MaxPooling2D((2,2), padding='same')(x)

	counter = 0

	# loop over our number of filters again, but this time in
	# reverse order
	for f in filters_list:
		counter += 1
		x = layers.Conv2DTranspose(f, kernel_s, activation='relu', padding='same')(x)
		x = layers.BatchNormalization()(x)
		if counter%2 == 0:
			x = layers.UpSampling2D((2,2))(x)

	# apply a single CONV_TRANSPOSE layer used to recover the
	x = layers.Conv2DTranspose(1, kernel_s, activation="sigmoid", padding="same")(x)
	autoencoder = Model(input_img, x, name="autoencoder")
	return autoencoder

# training 
def training(autoencoder, all_images, epochs=10, batch_size=128):
	print("[INFO] training started...")
	df = pd.DataFrame.from_dict(all_images, orient='index')
	# now in df we have a dataframe with size: dimensions x number_of_images with all of our images
	opt = keras.optimizers.Adam(lr=1e-3)
	autoencoder.compile(loss="mse", optimizer=opt)

	# To train it we will need the data
	train, test = train_test_split(df, test_size=0.33, random_state=42)

	train = train.astype('float32') / 255.
	test = test.astype('float32') / 255.

	train = train.to_numpy()
	test = test.to_numpy()

	train = np.reshape(train, (len(train), 28, 28, 1))
	test = np.reshape(test, (len(test), 28, 28, 1))

	history = autoencoder.fit(train, train,
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True,
					validation_data=(test, test),
					callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

	decoded_imgs=autoencoder.predict(test)
	return (history, decoded_imgs, test)

def plotting(history, decoded_imgs, test, epochs=10):
	# Plot 1
	n = 10
	plt.figure(figsize=(20, 4))
	for i in range(1, n + 1):
		# Display original
		ax = plt.subplot(2, n, i)
		plt.imshow(test[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# Display reconstruction
		ax = plt.subplot(2, n, i + n)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

	# Plot 2
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# Plot 3
	plot_name = "./logs/autoencoder_train_val_loss.png"
	plot_loss(history, plot_name)

	return 0
