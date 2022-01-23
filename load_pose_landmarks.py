import pandas as pd
from tensorflow import keras


def load_pose_landmarks(csv_path):
    """Loads a CSV created by MoveNetPreprocessor.

    Returns:
      X: Detected landmark coordinates and scores of shape (N, 17 * 3)
      y: Ground truth labels of shape (N, label_count)
      classes: The list of all class names found in the dataset
      dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
        truth labels (y) to use later to train a pose classification model.
    """
    # Load the CSV file
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()
    # Drop the file_name columns as you don't need it during training.
    df_to_process.drop(columns=['file_name'], inplace=True)
    # Extract the list of class names
    classes = df_to_process.pop('class_name').unique()
    # Extract the labels
    y = df_to_process.pop('class_no')
    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype('float64')
    y = keras.utils.to_categorical(y)
    return X, y, classes, dataframe
