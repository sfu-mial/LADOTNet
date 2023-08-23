from keras.layers import *
from keras.models import Model,model_from_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import Utils_TL_new as a #Utils as a # dmyinput ata_inputall as a
import matplotlib.pyplot as plt
from keras import optimizers
import keras  as keras
from keras import backend as K
import os
from keras import losses
#from keras.models import model_from_json
# from keras.utils import multi_gpu_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
K = keras.backend

#from keras.models import load_model
#from keras.callbacks import EarlyStopping
from keras.callbacks import History
#from adabound import AdaBound

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')
logger.setLevel(logging.INFO)
#history = History()
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # The GPU to choose, either "0" or "1"
#normal=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)
import tensorflow as tf

current_directory = os.getcwd()
# final_directory = os.path.join(current_directory, 'testcase')
Clinical_directory = os.path.join(current_directory, 'TL_results')
# if not os.path.exists(final_directory):
#    os.makedirs(final_directory)
if not os.path.exists(Clinical_directory):
   os.makedirs(Clinical_directory)

def huber_loss(y_true, y_pred):
    h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    return h(y_true,y_pred)
class MyCallback(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch >100 and K.get_value(self.beta)<=0.5:
            K.set_value(self.beta, K.get_value(self.beta) +0.0001)
            if  K.get_value(self.alpha)<0.5:
                 K.set_value(self.alpha, K.get_value(self.alpha) +0.00002)
#            K.set_value(self.alpha, max(0.75, K.get_value(self.alpha) -0.0001))
#                  K.set_value(self.beta,  min(0.7, K.get_value(self.beta) -0.0001))
        logger.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))
def squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
#K.square(y_pred - y_true)+K.std(y_true)

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

# kullback_leibler_divergence = keras.losses.kullback_leibler_divergence

def kl_divergence_regularizer(inputs):
    means = K.mean(inputs, axis=0)
    return 0.00000001 * (kullback_leibler_divergence(K.cast(0.05,'float32'), means)
                 + kullback_leibler_divergence(K.cast(1 - 0.05,'float32'), 1 - means))

def normalized_intensity(actualscan):
    
    max_actuali = np.max(actualscan[:,0:128], axis=1); #max value of each row where a row is source (i/ii) measurmenent for a given frequency
    max_actualii = np.max(actualscan[:,128:], axis=1); #max value of each row where a row is source (i/ii) measurmenent for a given frequency
    
    min_actuali = np.min(actualscan[:,0:128], axis=1); ##min value of each row
    min_actualii = np.min(actualscan[:,128:], axis=1); ##min value of each row
    
    minmaxi=(max_actuali - min_actuali).reshape((actualscan[:,0:128] .shape[0],1))
    minmaxii=(max_actualii - min_actualii).reshape((actualscan[:,128:].shape[0],1))

    normalized_intensityi =np.divide((actualscan[:,0:128] - min_actuali.reshape((actualscan[:,0:128] .shape[0],1))) ,minmaxi)
    normalized_intensityii =np.divide((actualscan[:,128:] - min_actualii.reshape((actualscan[:,0:128] .shape[0],1))) ,minmaxii)
    normalized_intensity= np.concatenate((normalized_intensityi, normalized_intensityii), axis=1)  #normalized_intensityi+ normalized_intensityii

    return normalized_intensity

def kl_divergence(rho, rho_hat):
    return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

def gaussian_kernel(x1, x2, beta = 1.0):
    r = tf.transpose(x1)
    r = tf.expand_dims(r, 2)
    return K.tf.reduce_sum(K.exp( -beta * K.square(r - x2)), axis=-1)
def MMD(x1, x2, beta):
#     ##"""
#     maximum mean discrepancy (MMD) based on Gaussian kernel
#     function for keras models (theano or tensorflow backend)
    
#     - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
#     Advances in neural information processing systems. 2007.
#    ## """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2)
    return diff


def weight_vector(x):
 
    for i in range (x.shape[0]):
           for j in range (255):
              if j>128:
                 x[i,j]= x[i,j]*(1+ (j)/128)
              else: 
                x[i,j]= x[i,j]*(1+ (256-j)/128)
    return x
def normalize_data(x):
    epsilon=0.001
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    return ((x - mvec)/stdvec)# s,mvec,stdvec


input_shape = Input(shape=(128,))
# input_ = GaussianNoise(0.02)(input_shape)
input_ =(input_shape)

x= Dense( 256, activation = 'sigmoid',  kernel_initializer='random_uniform',kernel_regularizer=tf.keras.regularizers.L2(0.001))(input_)#0000001 

x= Dense(128, activation = 'sigmoid', kernel_initializer='random_normal')(x) #he_normal #activity_regularizer=kl_divergence_regularizer
#x= Dense(output_dim = 16*16, activation = 'sigmoid', kernel_initializer='he_normal')(x)
#x= Dense(output_dim = 128, activation = 'sigmoid', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.1))(input_shape)
def windowedmae(y_true,y_pred,k=4):
    ms = 0
    count =0
    n,m = np.shape(y_pred)
    for i in range (k,m):
        #  i=1+k
         x= y_true[:,i-k:i+k]
         z= y_pred[:,i-k:i+k]
        #  print ("x",np.shape (x))
        #  print ("z",np.shape (z))
         L1_Distance= K.mean (tf.keras.losses.mean_absolute_error(x, z)) #K.mean(K.sum(K.abs(x - z), axis=-1), axis=-1)#
         ms += L1_Distance
        #  print ("ms",np.shape (ms))
         count +=1
    return ms/count

def gm(xs):
    return K.exp(K.mean(K.log(xs))) ## Make sure never negative or zero

def flatness(xs):
    return gm(xs)/K.mean(xs)

def flatnesserror(xs, ys):
    zs=K.abs(xs - ys)
    return flatness(1 + zs)
def loss(alpha):
    def custom_loss_func(y_true, y_pred):
        return custom_loss(y_true, y_pred,beta, alpha)
    return custom_loss_func
alpha = K.variable(0.2)#(0.1)
beta = K.variable(0.2)#(0.1)

def custom_loss(y_true, y_pred,alpha, beta):
    # loss =kullback_leibler_divergence(y_true, y_pred) #alpha*losses.mean_absolute_error(y_true, y_pred)# was 1*   mean_squared_error
    # loss +=huber_loss(y_true, y_pred) 
    L_mae= tf.keras.losses.mean_absolute_error(y_true, y_pred)
    L_wind = windowedmae(y_true, y_pred)
    L_flatnesserror= flatnesserror(y_true, y_pred)
    L_MMD= MMD(y_true, y_pred,1)
    loss= L_MMD+L_mae+ alpha*L_wind #+beta*L_flatnesserror+L_mae
    return  loss/2
model_loss= loss(alpha) 

model = Model(inputs=input_shape, outputs=x)
#opt = AdaBound(lr=1e-03,
#                        final_lr=0.1,
#                        gamma=1e-03,
#                        weight_decay=0.,
#                        amsbound=False)
opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer="Adam", loss=model_loss)#'mae')#'logcosh' ) #keras.losses.kullback_leibler_divergence#'mse' , huber_loss, mean_absolute_error #_logarithmic

model.summary()

x_train_i, y_train_i, x_test_i,  y_test_i = a.load_data()


  
# ####-0-128##
# x_train= normalize_data(x_train_i[:,0:128])
# x_test = normalize_data(x_test_i[:,0:128])
# y_train= (y_train_i[:,0:128]/y_train_i[:,0:128].max())
# y_test = (y_test_i[:,0:128]/y_test_i[:,0:128].max())


# # # # ##128:256 ###
x_train= normalize_data(x_train_i[:,128:256])
x_test = normalize_data(x_test_i[:,128:256])
y_train= (y_train_i[:,128:256]/y_train_i[:,128:256].max())
y_test = (y_test_i[:,128:256]/y_test_i[:,128:256].max())



keras.callbacks.Callback()

#early_stopping= keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=500, verbose=0, mode='auto')
keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=500, batch_size=128, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
TensorBoard= keras.callbacks.TensorBoard(log_dir='/home/hanenby/ML_dataset/log/batch', histogram_freq=10, batch_size=128, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
callbacks = [MyCallback(alpha, beta)]
history =model.fit(x_train,y_train,
                epochs=1000,
                batch_size=8,
                shuffle=True,
                verbose=1,
                validation_split=0.1,
                callbacks=[MyCallback(alpha, beta)]
                   )
model.save('./TL_results/TL-MMD_L/TL.h5')
import timeit

#beg_ts = time.time()
start_time = timeit.default_timer()
#decoded_imgs = loaded_model.predict(nn)
#elapsed = timeit.default_timer() - start_time
#end_ts = time.time()
decoded_imgs_test = model.predict(x_test)
#beg_ts = time.time()
start_time = timeit.default_timer()
#decoded_imgs = loaded_model.predict(nn)
elapsed = timeit.default_timer() - start_time
#end_ts = time.time()
print(elapsed)
#test section
for i in range (0,len(y_test)):
	 tes=y_test[i]#.reshape(128, 128)
	 plt.plot(tes.T,'b-')
	 plt.plot((decoded_imgs_test[i]).T,'g')
	 labels = ["GT", "Result"]
	 plt.legend(labels, loc=(1, 0))
	 plt.savefig('./TL_results/TL-MMD_L/result_imag '+ str(i)+'.png', bbox_inches=None)
	 # # tes = y_test[i]
	 # # im = plt.imshow(tes.reshape(128, 128))
    #  # # cbar= plt.colorbar(im,orientation='vertical')
	 # # plt.colorbar()
	 # # plt.savefig('./outsig/test_image' + str(i) + '.png', bbox_inches=None, frameon=None)
	 plt.close('all')
decoded_imgs = model.predict(x_train)

for i in range (0,len(y_train)):
	 tes=y_train[i]#.reshape(128, 128)
	 plt.plot(tes.T,'b-')
	 plt.plot((decoded_imgs[i]).T,'g')
	 labels = ["GT", "Result"]
	 plt.legend(labels, loc=(1, 0))
	 plt.savefig('./TL_results/TL-MMD_L/result_train_imag '+ str(i)+'.png', bbox_inches=None)
	 # # tes = y_test[i]
	 # # im = plt.imshow(tes.reshape(128, 128))
    #  # # cbar= plt.colorbar(im,orientation='vertical')
	 # # plt.colorbar()
	 # # plt.savefig('./outsig/test_image' + str(i) + '.png', bbox_inches=None, frameon=None)
	 plt.close('all')     

#decoded_1 = model.predict(x_train)
decoded= np.concatenate((decoded_imgs,decoded_imgs_test), axis=0)
x_all= np.concatenate((x_train,x_test ), axis=0)
y_all= np.concatenate((y_train,y_test ), axis=0)
plt.plot(x_all.T, '-m')
plt.plot(y_all.T, '-g')
# plt.plot(decoded.T, '-y')
labels=[ "g-synth","m- real"]
plt.legend( labels, loc=(1,0))
plt.savefig('./TL_results/TL-MMD_L/result_all.png', bbox_inches=None)
plt.close('all')


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
#plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./TL_results/TL-MMD_L/model loss.png", bbox_inches='tight')
plt.close("all")

