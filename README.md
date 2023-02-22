# ImageSearch

This phase of the project delves into concepts of various image feature descriptors and
how they transform a given set of images into features without any loss of important
information. Thereby the project aims to build an application leveraging important
information to recognize the query image using models Histogram of oriented gradients
(HOG), Extended Local Binary Pattern (ELBP), Color Moments and distance metrics
Manhattan Distance, Euclidean distance, from various image datasets.


Command to store all features in file:
python3 /Users/vishnuvardhan/Desktop/CSE-515/task_2.py
"/Users/vishnuvardhan/Desktop"
python3 /Users/vishnuvardhan/Desktop/CSE-515/task_2.py root_path

Command to run for task-3:
python3 /Users/vishnuvardhan/Desktop/CSE-515/task_3.py
"/Users/vishnuvardhan/Desktop" "set2" "image-0.png" "cm" 4 "manhattan"

It returns k nearest neighbours
python3 file_name code_directory set_name image_name model_name k
Distance_metric_name is either manhattan or euclidean

Command to run for task-4:
python3 /Users/vishnuvardhan/Desktop/CSE-515/task_4.py
"/Users/vishnuvardhan/Desktop" "set2" "image-0.png" 4 "manhattan"
