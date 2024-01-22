import os
import shutil

# run json_to_dataset.bat first
curr_dir = os.getcwd()
img_dir = "../mydata/pic"
json_dir = "../mydata/json"
label_dir = "../mydata/cv2_mask"
labelme_dir = "../mydata/labelme_json"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(json_dir):
    os.makedirs(json_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(labelme_dir):
    os.makedirs(labelme_dir)

file_list = os.listdir(curr_dir)
for file in file_list:
    file_path = os.path.join(curr_dir, file)
    if os.path.isdir(file_path):
        new_labelme_dir = os.path.join(labelme_dir, file)
        shutil.copytree(file_path, new_labelme_dir)
        label_file_path = os.path.join(file_path, 'label.png')
        if os.path.isfile(label_file_path):
            # new file name is the file directory name
            new_file_name = file + '.png'
            new_file_path = os.path.join(label_dir, new_file_name)
            shutil.copy2(label_file_path, new_file_path)
            print(f"'label.png' from {label_file_path} has been copied and renamed to {new_file_path}")
        else:
            print("error")
    else:
        if file.endswith(".jpg") or file.endswith(".png"):
            new_file_path = os.path.join(img_dir, file)
            shutil.copy2(file_path, new_file_path)
        elif file.endswith(".json"):
            new_file_path = os.path.join(json_dir, file)
            shutil.copy2(file_path, new_file_path)
