In this folder have the training of all the classification model:
	- MobileNetV3
	- EfficientNetV2
	- DinoV2
	
Params:
	Epochs: 100
	
For this train, is used the dataset for classification, with and without classification

All the loop for training is done by hand, with pytorch and transform library. It used the default transform for ImageNet for all the process
For the train, was applied:

	- MobileNetV3
		- With Base Dataset
		- With Data Augmented Dataset
	- EfficientNetV2
		- With Base Dataset
		- With Data Augmented Dataset
		- With Data Augmented Dataset, but with 150 epochs
	- DinoV2
		- With Base Dataset
		- With Data Augmented Dataset
		- With Data Augmented Dataset, but only train the classification layer