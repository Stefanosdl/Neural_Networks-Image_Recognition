import sys
from reader import handleInput, read_all_files

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
	

if __name__ == "__main__":
	main(sys.argv[1:])
