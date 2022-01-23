import numpy as np
import tensorflow as tf
from MoveNetPreprocessor import MoveNetPreprocessor
import os
from load_pose_landmarks import load_pose_landmarks

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="pose_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)


images_in_train_folder = os.path.join("yoga_cg", 'pred')
images_out_train_folder = 'poses_images_out_pred'
csvs_out_train_path = 'prediction.csv'

preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    images_out_folder=images_out_train_folder,
    csvs_out_path=csvs_out_train_path,
)

preprocessor.process(per_pose_class_limit=None)

X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

row = X
row = row.astype("float32")
row = row[0:1]
image_categories = ["Chair", "Left", "Right", "Tree"]


interpreter.set_tensor(input_details[0]['index'], row)


interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
