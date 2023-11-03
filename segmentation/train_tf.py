import dataloader_tf
import config

if __name__ == '__main__':
    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    from model_tf import unet_model
    shape = config.RESIZE
    unet = unet_model()
    model = unet.unet_model((*shape, 3), (*shape, 1), config.DEPTH)

    model_filename, model_callbacks = unet.get_callbacks()
    print(model_filename)

    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    model.fit(
        dataloader_tf.train_batches,
        epochs=config.EPOCHS,
        validation_data=dataloader_tf.test_batches,
        verbose=1,
        callbacks=model_callbacks,
        use_multiprocessing=True,
        workers=config.NUM_INTRA_THREADS,
        steps_per_epoch=dataloader_tf.STEPS_PER_EPOCH
    )

    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet.evaluate_model(model_filename, dataloader_tf.test_batches, dataloader_tf.STEPS_PER_EPOCH)

