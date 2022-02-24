
# Brain Tumor Detection

Build a model using Convolutional Neural Network for image classification.
Images

CNNs are widely used in image recognition, image classification, object detection, face recognition etc.

Convolutional Neural Network have various layers of processing which provides more accurate results.

A Convolutional Neural Network is a special type of an Artificial Intelligence implementation which uses a special mathematical matrix manipulation called the convolution operation to process data from images.

# Data Source
train = contains the training data/images for teaching our model.

val = contains images which we will use to validate our model. The purpose of this data set is to prevent our model from Overfitting. Overfitting is when your model gets a little too comfortable with the training data and can't handle data it hasn't see....too well.

test = this contains the data that we use to test the model once it has learned the relationships between the images and their label (Tumor/Not-Tumor)

# Model training

Epochs = 9
Steps per Epochs = 8
Validatiob Steps = 20

Result accuracy depends on Epochs, more the number of epochs , more accurate results
