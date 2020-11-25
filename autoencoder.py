import sys
from reader import handleInput, reader
from convolutional import convolutionalAutoencoder, training, plotting

def main(argv):
	inputfile = handleInput(argv)
	if not inputfile:
		sys.exit("You need to provide a dataset file path")
	epochs = 10
	autoencoder = convolutionalAutoencoder()
	all_images = reader(inputfile)
	(history, decoded_imgs, test) = training(autoencoder, all_images)
	
	while True:
		option = input('''
		******************************************
		*	Press 1 for re-execution         *
		*	Press 2 to display figures       *
		*	Press 3 to save the model        *
		******************************************
		''')

		if option == "1":
			epochs, batch_size, filter_pixel, filters, layers = [int(i) for i in input("Insert epochs, batch size, filter_pixel, filters, layers: ").split(",")]
			if layers > 5:
				print("You need to provide less than or equal to 5 layers")
				continue
			if layers < 1:
				print("Please provide a positive number of layers")
				continue
			autoencoder = convolutionalAutoencoder(int(filter_pixel), int(filters), int(layers))
			(history, decoded_imgs, test) = training(autoencoder, all_images, int(epochs), int(batch_size))
		elif option == "2":
			plotting(history, decoded_imgs, test, int(epochs))
		elif option == "3":
			path = input("Insert the path to save model: ")
			autoencoder.save(path+"/model.h5")
		else:
			print("Wrong input")

	return 0

if __name__ == "__main__":
	main(sys.argv[1:])
