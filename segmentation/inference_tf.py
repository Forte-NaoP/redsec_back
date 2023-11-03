from dataloader_tf import *
import numpy as np
import config
from model_tf import unet_model

from tensorflow import keras as K
import os
saved_model_dir = config.OUTPUT_PATH+config.MODEL_NAME
model_filename = os.path.join(saved_model_dir)

unet = unet_model()
model = unet.load_model(model_filename)

model.summary()


def get_test_image_and_annotation_arrays():
    '''
    Unpacks the test dataset and returns the input images and segmentation masks
    '''

    ds = test_batches.unbatch()
    ds = ds.batch(info.splits['test'].num_examples)

    images = []
    y_true_segments = []

    for image, annotation in ds.take(1):
        y_true_segments = annotation.numpy()
        images = image.numpy()

    y_true_segments = y_true_segments[
                      :(info.splits['test'].num_examples - (info.splits['test'].num_examples % BATCH_SIZE))]

    return images[
           :(info.splits['test'].num_examples - (info.splits['test'].num_examples % BATCH_SIZE))], y_true_segments


def create_mask(pred_mask):
    '''
    Creates the segmentation mask by getting the channel with the highest probability. Remember that we
    have 3 channels in the output of the UNet. For each pixel, the predicition will be the channel with the
    highest probability.
    '''

    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0].numpy()


def make_predictions(image, mask, num=1):
    '''
    Feeds an image to a model and returns the predicted mask
    '''

    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    pred_mask = model.predict(image)
    pred_mask = create_mask(pred_mask)

    return pred_mask


def class_wise_metrics(y_true, y_pred):
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(3):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)

        dice_score = 2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


y_true_images, y_true_segments = get_test_image_and_annotation_arrays()

# feed the test set to th emodel to get the predicted masks
results = model.predict(test_batches, steps=info.splits['test'].num_examples // BATCH_SIZE)
results = np.argmax(results, axis=3)
results = results[..., tf.newaxis]

# compute the class wise metrics
cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_true_segments, results)

# show the IOU for each class
for idx, iou in enumerate(cls_wise_iou):
    spaces = ' ' * (10-len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, iou))


# show the Dice Score for each class
for idx, dice_score in enumerate(cls_wise_dice_score):
    spaces = ' ' * (10-len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, dice_score))

# Please input a number between 0 to 3647 to pick an image from the dataset
import random as rand
for _ in range(10):

    integer_slider = rand.randint(0, 3647)

    # Get the prediction mask
    y_pred_mask = make_predictions(y_true_images[integer_slider], y_true_segments[integer_slider])

    # Compute the class wise metrics
    iou, dice_score = class_wise_metrics(y_true_segments[integer_slider], y_pred_mask)

    # Overlay the metrics with the images
    display_with_metrics([y_true_images[integer_slider], y_pred_mask, y_true_segments[integer_slider]], iou, dice_score)
