import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16


def build_model():
    input_layer = Input(shape=(120,120,3))
    vgg = VGG16(include_top=False)(input_layer)
    
    f1 = GlobalMaxPooling2D()(vgg)
    c1 = Dense(2048, activation='relu')(f1)
    c2 = Dense(1, activation='sigmoid')(c1)
    
    f2 = GlobalMaxPooling2D()(vgg)
    r1 = Dense(2048, activation='relu')(f2)
    r2 = Dense(4, activation='sigmoid')(r1)
    
    return Model(inputs=input_layer, outputs=[c2, r2])

def localization_loss(y_true, y_pred):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - y_pred[:,:2]))
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]
    
    h_pred = y_pred[:,3] - y_pred[:,1]
    w_pred = y_pred[:,2] - y_pred[:,0]
    
    delta_size = tf.reduce_sum(tf.square(h_true - h_pred) + tf.square(w_true - w_pred))
    
    return delta_size + delta_coord


class Facetracker(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = build_model()
        
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localization_loss
        self.opt = opt
        
    def train_step(self, batch, **kwargs):
        x, y = batch
        with tf.GradientTape() as tape:
            classes, coords = self.model(x, training=True)
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss + 0.5*batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))        
        return {'total':total_loss, 'class':batch_classloss, 'regress':batch_localizationloss} 
    
    def test_step(self, batch, **kwargs):
        x, y = batch  
        classes, coords = self.model(x, training=False)
        closs = self.closs(y[0], classes)
        rloss = self.lloss(tf.cast(y[1], tf.float32), coords)            
        total_loss = rloss + 0.5*closs
        return {'val_total':total_loss, 'val_class':closs, 'val_regress':rloss}
    
    def call(self, X, **kwargs):
        return self.model(X, **kwargs)
    
    def save(self, filename):
        self.model.save(filename)