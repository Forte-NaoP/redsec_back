import tensorflow as tf
from tensorflow import keras as K
import config
import os

class unet_model:
    def __init__(self,
                 fms=config.FEATURE_MAPS,
                 output_path=config.OUTPUT_PATH,
                 model_name=config.MODEL_NAME,
                 learning_rate=config.LEARNING_RATE,
                 weight_dice_loss=config.WEIGHT_DICE_LOSS,
                 num_threads=config.NUM_INTRA_THREADS,
                 use_dropout=config.USE_DROPOUT,
                 print_model=config.PRINT_MODEL):

        self.fms = fms
        self.output_path = output_path
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_dice_loss = weight_dice_loss
        self.num_threads = num_threads
        self.use_dropout = use_dropout
        self.print_model = print_model

        self.optimizer = K.optimizers.Adam(learning_rate=self.learning_rate)

        self.concat_axis = -1
        K.backend.set_image_data_format('channels_last')

    def unet_model(self, image_shape, mask_shape, depth, drop_out=0.2):

        num_chan_in = 3
        num_chan_out = 3

        self.input_shape = image_shape
        self.num_input_channels = num_chan_in

        inputs = K.layers.Input(shape=image_shape, name='input_image')

        params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

        params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                            padding="same")

        encoder_out, encode_layers = encoder(inputs, self.fms, params, depth=depth)
        bottom = conv2d_block(encoder_out, self.fms*(2**depth), params)
        decoder_out = decoder(bottom, encode_layers, self.fms, params, params_trans, self.concat_axis)

        prediction = K.layers.Conv2D(name='output',
                                     filters=num_chan_out, kernel_size=(1, 1),
                                     activation='sigmoid')(decoder_out)

        model = K.models.Model(inputs=[inputs], outputs=[prediction], name="pet_unet")
        optimizer = self.optimizer

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

        if self.print_model:
            model.summary()

        return model

    def get_callbacks(self):
        """
        Define any callbacks for the training
        """

        model_filename = os.path.join(
            self.output_path, self.model_name)

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = K.callbacks.ModelCheckpoint(model_filename,
                                                       verbose=1,
                                                       monitor="val_loss",
                                                       save_best_only=True)

        directoryName = "unet_inter{}".format(self.num_threads)

        # Tensorboard callbacks

        tensorboard_filename = os.path.join(self.output_path,
                                            "keras_tensorboard_transposed/{}".format(
                                                directoryName))

        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=tensorboard_filename,
            write_graph=True, write_images=True)

        early_stopping = K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        return model_filename, [model_checkpoint, early_stopping, tensorboard_checkpoint]

    def evaluate_model(self, model_filename, ds_test, steps):
        """
        Evaluate the best model on the validation dataset
        """

        model = K.models.load_model(
            model_filename)#, custom_objects=self.custom_objects)

        print("Evaluating model on test dataset. Please wait...")
        metrics = model.evaluate(
            ds_test,
            steps=steps,
            verbose=1)

        for idx, metric in enumerate(metrics):
            print("Test dataset {} = {:.4f}".format(
                model.metrics_names[idx], metric))

    def load_model(self, model_filename) -> K.models.Model:
        """
        Load a model from Keras file
        """

        return K.models.load_model(model_filename)#, custom_objects=self.custom_objects)


def conv2d_block(input_tensor, filters, params):
    x = input_tensor
    for _ in range(2):
        x = K.layers.Conv2D(filters=filters, **params)(x)

    return x


def encode_block(inputs, filters, params, pool_size=(2, 2), drop_out=0.2):
    cb = conv2d_block(inputs, filters=filters, params=params)
    p = K.layers.AveragePooling2D(pool_size=pool_size)(cb)
    p = K.layers.SpatialDropout2D(drop_out)(p)

    return cb, p


def encoder(inputs, filters, params, depth):
    encode_layers = []
    p = inputs
    for i in range(depth):
        f, p = encode_block(p, filters*(2**i), params)
        encode_layers.append(f)

    return p, encode_layers


def decode_block(inputs, conv_concat, filters, params, param_trans, axis):
    up = K.layers.Conv2DTranspose(filters=filters, **param_trans)(inputs)
    concat = K.layers.concatenate([up, conv_concat], axis=axis)
    cb = conv2d_block(concat, filters=filters, params=params)

    return cb


def decoder(inputs, encode_layers, filters, params, param_trans, axis):
    db = inputs
    for i in reversed(range(len(encode_layers))):
        db = decode_block(db, encode_layers[i], filters*(2**i), params, param_trans, axis)
    return db


