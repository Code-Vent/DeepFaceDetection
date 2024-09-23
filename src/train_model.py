import dataset
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from model import Facetracker, localization_loss


def main():
    train, _ , val = dataset.get()
    
    batches_per_epoch = len(train)
    lr_decay = (1./0.75 - 1)/batches_per_epoch
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)
    
    model = Facetracker()
    classification_loss = tf.keras.losses.BinaryCrossentropy()
    #regression_loss = localization_loss
    model.compile(opt, classification_loss, localization_loss)
    logdir = 'logs'
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
    model.save('facetracker.h5')
    return 0
    

if __name__ == '__main__':
    exit(main())
    





