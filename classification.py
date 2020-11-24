import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from keras import models, layers
from tensorflow.keras.models import Model
from reader import handleInput, read_all_files
from classifier import classification_training, plot_figures, plot_metrics 

def main(argv):
	training_set, training_labels, test_set, test_labels, model = handleInput(argv)
	if not training_set:
		sys.exit("You need to provide a training_set file path")
	elif not training_labels:
		sys.exit("You need to provide a training_labels file path")
	elif not test_set:
		sys.exit("You need to provide a test_set file path")
	elif not test_labels:
		sys.exit("You need to provide a test_labels file path")
	elif not model:
		sys.exit("You need to provide a model file path")

	train_images, train_labels, test_images, test_labels = read_all_files(training_set, training_labels, test_set, test_labels)
	# Create the classifier model
	# unpack autoencoder to get encoder (N1)
	autoencoder = models.load_model(model)
	classifier = models.Sequential(name="encoder")

	for i in range(0, len(autoencoder.layers) // 2):
		classifier.add(autoencoder.layers[i])

	for i in range(0, len(classifier.layers)):
		classifier.layers[i].trainable = False
	
	classifier.add(layers.Flatten())
	classifier.add(layers.Dense(512, activation='relu'))
	classifier.add(layers.Dropout(0.3))
	classifier.add(layers.Dense(10, activation='sigmoid'))
    # finish classifier model
	batch_size = 128
	epochs = 10
	(classify_train, predicted_classes, test_data, test_eval) = classification_training(classifier, train_images, test_images, train_labels, test_labels, epochs, batch_size)

	# re-execution 
	while True:
		option = input('''
		******************************************
		*	Press 1 for re-execution         *
		*	Press 2 to display metrics       *
		*	Press 3 to display figures       *
		******************************************
		''')

		if option == "1":
			epochs, batch_size = [int(i) for i in input("Insert epochs, batch size: ").split(",")]
			(classify_train, predicted_classes, test_data, test_eval) = classification_training(classifier, train_images, test_images, train_labels, test_labels, int(epochs), int(batch_size))
		elif option == "2":
			plot_metrics(classify_train, test_eval, test_labels, predicted_classes, int(batch_size), int(epochs))
		elif option == "3":
			plot_figures(predicted_classes, test_labels, test_data, int(batch_size))
		else:
			print("Wrong input")

if __name__ == "__main__":
	main(sys.argv[1:])
