
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint

import tensorflow as tf
from tensorflow import keras
from keras.layers import Lambda, Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn import preprocessing, metrics

np.random.seed=42
tf.random.set_seed(42)

from plot_utils import plot_time_series, plot_percent_change

class stocks:
    """
    class to organize stock data info
    """
    def __init__(self):
        self.exxon=self.get_data('data/Exxon.csv')
        self.nasdaq=self.get_data('data/NASDAQ.csv')

    def get_data(self, loc):
        """
        takes: location of data
        returns: dict with (pd, training data, testing data) of that stock
        """
        data=pd.read_csv(loc)
        train,test=self.train_test_split(data)

        normalized_train=preprocessing.normalize(train.reshape(1,-1),norm='max',axis=1)
        normalized_test=preprocessing.normalize(test.reshape(1,-1),norm='max',axis=1)

        info={
            'data':data,
            'raw train':train,
            'train': normalized_train.reshape(normalized_train.shape[1]),
            'raw test':test,
            'test': normalized_test.reshape(normalized_test.shape[1]),
        }

        return info

    def train_test_split(self,pd):
        """
        takes: pandas dataframe of stock info
        returns: training, testing time-series data split
        """
        date=np.array(pd['date'])
        close=np.array(pd['close'])

        # in this case, vae only takes closing price
        x_data=close
        halfway=len(date)/2

        # time-series train-test split
        train, test = [], []
        cutoff_reached=False
        for row, day in enumerate(date):

            # check if reached cutoff date; raise flag
            if day=='2018-01-02':
                cutoff_reached=True

            if cutoff_reached==False:
                train.append(x_data[row])
            elif cutoff_reached==True:
                test.append(x_data[row])
        
        train=np.array(train)
        test=np.array(test)

        return [train,test]

def generate_samples(data,sample_size):
    num_samples=data.shape[0]//sample_size
    results=np.empty((num_samples,sample_size))
    for i in range(num_samples):
        results[i]=data[i*sample_size:i*sample_size+sample_size]
    return results

class sampling(keras.layers.Layer):
    def call(self,inputs):
        mean,log_var=inputs
        return K.exp(log_var/2)*K.random_normal(tf.shape(log_var))+mean

def vae(x_train,x_test,sample_size):

    latent_dim=2

    inputs=Input(shape=sample_size)
    z=Dense(198, activation='sigmoid',name='encoder_dense1')(inputs)
    z=keras.layers.LeakyReLU(128,name='encoder_LReLU2')(z)
    z=Dense(96,activation='tanh', name='encoder_dense2')(z)
    z=Dense(32,activation='sigmoid', name='encoder_dense3')(z)

    latent_mean=Dense(latent_dim,name='latent_mean')(z)
    latent_log_var=Dense(latent_dim,name='latent_log_var')(z)
    z=sampling()([latent_mean, latent_log_var])

    encoder=Model(
                    inputs=[inputs],
                    outputs=[latent_mean,latent_log_var,z],
                    name='encoder'
                 )
    encoder.summary()
    
    decoder_inputs=Input(shape=(latent_dim,))
    x=Dense(32,activation='sigmoid',name='decoder_dense1')(decoder_inputs)
    x=Dense(96,activation='tanh',name='decoder_dense2')(x)
    x=keras.layers.LeakyReLU(128,name='decoder_LReLU2')(x)
    x=Dense(198,activation='sigmoid',name='decoder_dense3')(x)

    outputs=Dense(sample_size,activation='sigmoid',name='decoder_output')(x)
    decoder=Model(
                    inputs=[decoder_inputs],
                    outputs=[outputs],
                    name='decoder'
                 )
    decoder.summary()
    
    _,_,sample=encoder(inputs)
    vae_outputs=decoder(sample)

    vae=Model(
                inputs=[inputs],
                outputs=[vae_outputs],
                name='vae'
             )

    def vae_loss(x,x_decoded):

        reconstruction_loss=keras.losses.mean_absolute_error(x,x_decoded)

        z_mean,z_log_var,_=encoder(x)
        kl_loss=-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
        kl_loss=tf.reduce_mean(tf.reduce_sum(kl_loss,axis=1))

        total_loss=reconstruction_loss+kl_loss

        return total_loss

    vae.compile(
                    loss=vae_loss,
                    optimizer=keras.optimizers.Adam(),
                    metrics=keras.metrics.mean_absolute_error
                )
    
    vae.fit(
                x_train,x_train,
                epochs=1000,
                batch_size=32,
                validation_data=(x_test,x_test)
            )

    return vae
    
if __name__ == '__main__':

    stock_data=stocks()

    # exxon mobil
    sample_size=len(stock_data.exxon['test'])
    
    x_train_exxon=generate_samples(stock_data.exxon['train'],sample_size)
    x_test_exxon=generate_samples(stock_data.exxon['test'],sample_size)
    exxon_vae=vae(x_train_exxon,x_test_exxon,sample_size)
    
    simulated_metric=np.empty((len(x_test_exxon),sample_size))
    for i in range(x_test_exxon.shape[0]):
        simulated_metric[i]=exxon_vae.predict(x_test_exxon[[i]])
    plt.style.use('ggplot')
    fig = plt.figure()

    axes = fig.add_subplot(111)
    axes.plot(stock_data.nasdaq['raw test']*x_test_exxon.flatten(), 'o-', label="actual")
    axes.plot(stock_data.nasdaq['raw test']*simulated_metric.flatten()+500, 'x-', label="predicted")
    axes.set_title('Exxon')

    plt.xlabel('Date')
    plt.ylabel('Prices($)')
    plt.legend()
    plt.show()

    # nasdaq
    sample_size=len(stock_data.nasdaq['test'])

    x_train_nasdaq=generate_samples(stock_data.nasdaq['train'],sample_size)
    x_test_nasdaq=generate_samples(stock_data.nasdaq['test'],sample_size)

    nasdaq_vae=vae(x_train_nasdaq,x_test_nasdaq,sample_size)

    simulated_metric=np.empty((len(x_test_nasdaq),sample_size))
    for i in range(x_test_nasdaq.shape[0]):
        print(i)
        simulated_metric[i]=nasdaq_vae.predict(x_test_nasdaq[[i]])

    plt.style.use('ggplot')
    fig = plt.figure()

    axes = fig.add_subplot(111)
    axes.plot(stock_data.nasdaq['raw test']*x_test_nasdaq.flatten(), 'o-', label="actual")
    axes.plot(stock_data.nasdaq['raw test']*simulated_metric.flatten(), 'x-', label="predicted")
    axes.set_title('Nasdaq')

    plt.xlabel('Date')
    plt.ylabel('Prices($)')
    plt.legend()
    plt.show()
        