In this folder have the structure and code used to train YOLOs for unimodel object detection + classification.
Is used Ultralytic python library to train all of the models, using an small script to get an dataset divided between train, test and val, with an fixed seed (42).

In the first training step, was trained with the Ultralytic's default values (Yolov12n_ObjectDetection_Classification)
On the second, was training without data modification, to validate if is better (Yolov12n_ObjectDetection_ClassificationWithoutDataModification)
On the last folder, was tried to train all YOLOs by removing the same number of outliers detected of the lower areas annotation

In the validation_model_OLD have the old validation code used.