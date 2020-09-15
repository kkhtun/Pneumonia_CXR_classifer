import cv2
import numpy as np

def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image/255.0
    image = np.reshape(image, (224,224,1))
    return image
    
def process_dataset(dataframe):
    X_data = []
    Y_labels = []
    
    for image_path, label in dataframe.values:
        X_data.append(process_image(image_path))
        Y_labels.append(label)
    X = np.array(X_data)
    Y = np.array(Y_labels)
    return X, Y