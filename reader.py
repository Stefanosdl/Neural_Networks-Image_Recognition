import getopt, sys

def handleInput(argv):
	training_set = ''
	training_labels = ''
	test_set = ''
	test_labels = ''
	model = ''

	for i in range(0, len(argv)):
		if argv[i] == "-d":
			try:
				training_set = argv[i+1]
			except IndexError:
				sys.exit("You need to provide a training_set file path")
		elif argv[i] == "-dl":
			try:
				training_labels = argv[i+1]
			except IndexError:
				sys.exit("You need to provide a training_labels file path")
		elif argv[i] == "-t":
			try:
				test_set = argv[i+1]
			except IndexError:
				sys.exit("You need to provide a test_set file path")
		elif argv[i] == "-tl":
			try:
				test_labels = argv[i+1]
			except IndexError:
				sys.exit("You need to provide a test_labels file path")
		elif argv[i] == "-model":
			try:
				model = argv[i+1]
			except IndexError:
				sys.exit("You need to provide a model file path")
	if len(argv) == 3:
		return training_set
	return (training_set, training_labels, test_set, test_labels, model)

def reader(dataset):
	# create a dictionary of lists to store all images
	all_images = {}
	all_images[0] = []
	with open(dataset, "rb") as f:
		counter = 0
		# read metadata
		while (byte:= f.read(4)):
			if counter == 0:
				magic_number = int.from_bytes(byte, "big")
			elif counter == 1:
				number_of_images = int.from_bytes(byte, "big")
			elif counter == 2:
				rows = int.from_bytes(byte, "big")
			elif counter == 3:
				cols = int.from_bytes(byte, "big")
				break
			counter += 1
		# start reading the images 
		byte_counter = 0
		image_counter = 0
		dimensions = rows * cols
		while (byte:= f.read(1)):
			# store byte in the 
			all_images[image_counter].append(int.from_bytes(byte, "big"))
			byte_counter += 1
			if (byte_counter == dimensions):
				# next image
				image_counter += 1
				byte_counter = 0
				# initialize the list for this image
				all_images[image_counter] = []
	# finished with reading of file
	# remove last item number_of_images index that is anyway an empty list
	all_images.popitem()
	# return all_images dict
	return all_images

def reader_labels(dataset):
	# create a dictionary of lists to store all images
	all_labels = []
	with open(dataset, "rb") as f:
		counter = 0
		# read metadata
		while (byte:= f.read(4)):
			if counter == 0:
				magic_number = int.from_bytes(byte, "big")
			elif counter == 1:
				number_of_items = int.from_bytes(byte, "big")
				break
			counter += 1
		# start reading the labels 
		while (byte:= f.read(1)):
			# store byte in the 
			all_labels.append(int.from_bytes(byte, "big"))
	# finished with reading of file
	return all_labels


def read_all_files(training_set, training_labels, test_set, test_labels):
	train_images = reader(training_set)
	train_labels = reader_labels(training_labels) 
	test_images = reader(test_set)
	test_labels = reader_labels(test_labels)
	# labels are lists of 10000 items
	return (train_images, train_labels, test_images, test_labels)
