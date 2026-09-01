import os, shutil


original_dataset = "../../../../DatasetOriginal/"
training_dataset = "Dataset"
val_dataset = "DatasetVal"

divided_by_label_folder = True

copy_image = True

## Create Folders

if not val_dataset in os.listdir("./"):
    os.mkdir(val_dataset)
    os.mkdir(os.path.join(val_dataset, "images"))
    os.mkdir(os.path.join(val_dataset, "labels"))
    os.mkdir(os.path.join(val_dataset, "images", "test"))
    os.mkdir(os.path.join(val_dataset, "labels", "test"))

## Copy images
#  

test_img_original_dataset = os.path.join(original_dataset, "images") 
test_img_dataset_train_path = os.path.join(training_dataset, "images", "test")
test_img_dataset_val_path = os.path.join(val_dataset, "images", "test")

if copy_image:
    
    img_list = [] 

    for folder in os.listdir(test_img_dataset_train_path):

        if "cache" in folder:
            continue

        for im in os.listdir(os.path.join(test_img_dataset_train_path, folder)):

            old_img = os.path.join(test_img_original_dataset, im)
            new_img = os.path.join(test_img_dataset_val_path, im)

            shutil.copy(old_img, new_img)

            img_list.append(new_img)

    print(f"Image Number: {len(img_list)}")

# Copy labels with original_data

test_labels_original_dataset = os.path.join(original_dataset, "labels") 
test_labels_dataset_train_path = os.path.join(training_dataset, "labels", "test")
test_labels_dataset_val_path = os.path.join(val_dataset, "labels", "test")

labels_path = []

for folder in os.listdir(test_labels_dataset_train_path):   

    if "cache" in folder:
        continue

    for txt in os.listdir(os.path.join(test_labels_dataset_train_path, folder)):
        
        old_txt = os.path.join(test_labels_original_dataset, txt)
        new_txt = os.path.join(test_labels_dataset_val_path, txt)

        shutil.copy(old_txt, new_txt)

        labels_path.append(new_txt)

print(f"Label Number: {len(labels_path)}")