Need to have the original dataset, in the format Dataset/AllData, and the scripts need to be run by order

Folders
	Dataset - Base Folder, with the multiclass and uniclass. Created with scripts 1
	DatasetOrganized - Correctly organized dataset, based with Dataset. Also have the outliers removed dataset's
	DatasetWithoutTrainTest - Final Dataset for unimodel object detection + classification
	DatasetObjectDetectionOnly - Dataset for object detection, in YOLO + Classifier aproach
	DatasetClassification - Dataset for classification, in YOLO + Classifier aproach
	DatasetClassificationWithDataAugment - Dataset for classification, with data augmentation, in YOLO + Classifier aproach


Scripts

1 - setlabels_at_zero (Jupyter Notebook)
	Run throught an dataset, and set the labels to zero. All txt use YOLOv1 format (label x y width height).
	Used for Esca Uniclass and LeafBlight Uniclass dataset
	Generate: Dataset/....
	
2 - reorganized_dataset (Jupyter Notebook)
	Some of the labels original dataset data are in the wrong folder. This jupyter notebook copy the images and move the wrong anotation to the new folder, keeping the same name 
	Generate: DatasetOrganized/
	
3 - remove_outliers (Jupyter Notebook)
	For the box plot used on the last folder, is created dataset without annotation with areas bigger than 20%/50% of the upper fence
	Generate: DatasetOrganized/....

4 - mix_train_test (Jupyter Notebook)
	After all the code before, this notebook mix the train/ and test/ images and labels, with the same structure
	Generate: DatasetWithoutTrainTest/
	
5 - create_object_detection_only_dataset (Jupyter Notebook)
	Create an dataset for object detection, detecting only if is an disease. Used for YOLO + Classifier
	Generate: DatasetObjectDetectionOnly/
	
6 - create_CNN_dataset (Jupyter Notebook)
	This notebook create the dataset used for training the Classification Models by cropping the original images with the annotation. Used for YOLO + Classifier
	Generate: DatasetClassification/
	
7 - data_augment (Jupyter Notebook)
	Apply Data Augmentation to classifier dataset, increasing each annotation to 40000 per label
	Generate: DatasetClassificationWithDataAugment/