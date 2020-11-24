import pandas as pd 
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

def classification_training(classifier, train_images, test_images, train_labels, test_labels, epochs=10, batch_size=128):
	train_data = pd.DataFrame.from_dict(train_images, orient='index')
	test_data = pd.DataFrame.from_dict(test_images, orient='index')

	train_data = train_data.astype('float32') / 255.
	test_data = test_data.astype('float32') / 255.

	train_data = train_data.to_numpy()
	test_data = test_data.to_numpy()

	train_data = np.reshape(train_data, (len(train_data), 28, 28, 1))
	test_data = np.reshape(test_data, (len(test_data), 28, 28, 1))

	# Change the labels from categorical to one-hot encoding
	train_Y_one_hot = keras.utils.to_categorical(train_labels)
	test_Y_one_hot = keras.utils.to_categorical(test_labels)

	train_X, valid_X, train_label, valid_label = train_test_split(train_data, train_Y_one_hot, test_size=0.2, random_state=13)

	classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	classify_train = classifier.fit(train_X, 
							train_label, 
							batch_size=batch_size,
							epochs=epochs,
							validation_data=(valid_X, valid_label), 
							callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
	
	# print the test loss and accuracy
	test_eval = classifier.evaluate(test_data, test_Y_one_hot, verbose=0)
	
	predicted_classes = classifier.predict(test_data)
	predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

	return (classify_train, predicted_classes, test_data, test_eval)


def plot_figures(predicted_classes, test_labels, test_data, batch_size):
	# print the pictures found correctly
	correct = np.where(predicted_classes == test_labels)[0]
	print("Found", len(correct), "correct labels")
	plt.figure(figsize=(7, 7))
	for i, correct in enumerate(correct[:9]):
		plt.subplot(3, 3, i+1)
		plt.imshow(test_data[correct].reshape(28,28), cmap='gray')
		plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
		plt.tight_layout()
	plot_name = "./logs/found_correct_fc"+str(batch_size)+".png"
	plt.savefig(plot_name, bbox_inches='tight')
	plt.show()
	
	# print the pictures found incorrectly
	incorrect = np.where(predicted_classes != test_labels)[0]
	print("Found", len(incorrect), "incorrect labels")
	plt.figure(figsize=(7, 7))
	for i, incorrect in enumerate(incorrect[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(test_data[incorrect].reshape(28,28), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
		plt.tight_layout()
	plot_name = "./logs/found_incorrect_fc"+str(batch_size)+".png"
	plt.savefig(plot_name, bbox_inches='tight')
	plt.show()


def plot_metrics(classify_train, test_eval, test_labels, predicted_classes, batch_size, epochs):
	# print loss
	correct = np.where(predicted_classes == test_labels)[0]
	incorrect = np.where(predicted_classes != test_labels)[0]
	plot_name = "./logs/fc" + str(batch_size) + "_complete_train_val_loss.png"
	plot_loss(classify_train, plot_name)
	# write txt with classification_report
	report = "./logs/classification_report_fc" + str(batch_size) + ".txt"
	file1 = open(report, "w")
	# write loss, acc, correct and incorrect labels
	string = "Test loss: " + str(test_eval[0]) + "\n" + "Test accuracy: " + str(test_eval[1]) + "\n\n"
	file1.writelines(string)
	string = "Found " + str(len(correct)) + " correct labels" + "\n" + "Found " + str(len(incorrect)) + " incorrect labels" + "\n\n"
	file1.writelines(string)
	# write metrics
	target_names = ["Class {}".format(i) for i in range(10)]
	print(classification_report(test_labels, predicted_classes, target_names=target_names))
	file1.writelines(classification_report(test_labels, predicted_classes, target_names=target_names))
	file1.close()

	# print accuracy
	accuracy = classify_train.history['accuracy']
	val_accuracy = classify_train.history['val_accuracy']
	epochs = range(len(accuracy))
	plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
	plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
	plt.title("Training and Validation accuracy")
	plt.legend()
	plot_name = "./logs/training_validation_acc_fc" + str(batch_size) + ".png"
	plt.savefig(plot_name, bbox_inches='tight')
	plt.show()


def plot_loss(fitted_model, plot_name):
	loss = fitted_model.history['loss']
	val_loss = fitted_model.history['val_loss']
	epochs = range(len(loss))
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and Validation loss')
	plt.legend()
	plt.savefig(plot_name, bbox_inches='tight')
	plt.show()
