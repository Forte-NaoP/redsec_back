import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import config


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint, resize=config.RESIZE):
    input_image = tf.image.resize(datapoint['image'], resize)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], resize)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples

print(info)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

# class list of the mask pixels
class_names = ['pet', 'background', 'outline']


def display_with_metrics(display_list, iou_list, dice_score_list):
    ''' displays a list of images/masks and overlays a list of IoU and Dice Score '''
    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list))]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sort in place

    display_string_list = [f"{class_names[idx]}: IoU: {iou} Dice Score: {dice_score}" for idx, iou, dice_score in
                           metrics_by_id]
    display_string = "\n\n".join(display_string_list)

    display(display_list, ["Image", "Predicted Mask", "True Mask"], display_string=display_string)


def display(display_list, titles=[], display_string=None):
    ''' displays list of images/masks'''
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if display_string and i == 1:
            plt.xlabel(display_string, fontsize=12)
        img_arr = tf.keras.preprocessing.image.array_to_img(display_list[i])
        plt.imshow(img_arr)

    plt.show()


def show_image_from_dataset(dataset):
    for image, mask in dataset.take(1):
        sample_image, sample_mask = image, mask
        print(type(sample_mask))
        b_mask = tf.cast(tf.equal(sample_mask, 1), tf.uint8)
        print(sample_image.shape, sample_mask.shape)
    display([sample_image, b_mask], titles=['Image', 'True Maks'])


# np.set_printoptions(threshold=np.inf)
#
# def tensor_to_2d(tensor):
#     return np.squeeze(tensor, axis=-1)


if __name__ == '__main__':
    # display an image from the train set
    show_image_from_dataset(train_images)

    # display an image from the test set
    show_image_from_dataset(test_images)