import numpy as np
from pypuf.simulation import *
import pypuf.metrics as pm
from pypuf.io import random_inputs
import pypuf.io, pypuf.simulation
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model


def get_parity_vectors(C):
    n = C.shape[1]
    m = C.shape[0]
    C[C == 0] = -1
    parityVec = np.zeros((m, n+1))
    parityVec[:, 0:1] = np.ones((m, 1))
    for i in range(2, n+2):
        parityVec[:, i -
                  1: i] = np.prod(C[:, 0: i-1], axis=1).reshape((m, 1))
    return parityVec
def get_parity_vectors2(C):
    # C = 2. * C - 1
    C = np.fliplr(C)
    C = np.cumprod(C, axis=1,  dtype=np.float)
    return C

class AccuracyStop(tf.keras.callbacks.Callback):
        def __init__(self, stop_validation_accuracy: float) -> None:
            super().__init__()
            self.stop_validation_accuracy = stop_validation_accuracy

        def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
            if float(logs.get('val_accuracy')) > self.stop_validation_accuracy:
                self.model.stop_training = True
            if float(logs.get('val_accuracy')) > 0.93 and float(logs.get('accuracy')) > 0.99:
                self.model.stop_training = True

class CustomSoftmaxThre(tf.keras.layers.Layer):
    def __init__(self, threshold=0.2, **kwargs):
        super(CustomSoftmaxThre, self).__init__(**kwargs)
        self.threshold = K.variable(threshold)
        
    def call(self, inputs):
        # softmax
        softmax_x = tf.nn.softmax(inputs)

        # set value below threshold to 0
        custom_output = tf.where(softmax_x < self.threshold, 0.0, softmax_x)
        return custom_output
    
class CustomSoftmaxExperts(tf.keras.layers.Layer):
    def __init__(self, experts_needed=5,threshold=0.2,**kwargs):
        super(CustomSoftmaxExperts, self).__init__(**kwargs)
        self.experts_needed = tf.Variable(experts_needed, trainable=False, dtype=tf.int32)
        self.threshold =  K.variable(threshold)
    # def call(self, inputs):
    #     # softmax
    #     softmax_x = tf.nn.softmax(inputs)
        
    #     # Find the values to keep
    #     top_k_values, _ = tf.nn.top_k(softmax_x, k=self.experts_needed)
    #     threshold_value = top_k_values[..., -1]  # Get the smallest value among the top-k values
    #     mask = tf.greater_equal(softmax_x, threshold_value)

    #     # Apply the mask, only keeping the top k activations
    #     custom_output = tf.where(mask, softmax_x, 0.0)

    #     return custom_output
    def call(self, inputs):
        # softmax
        softmax_x = tf.nn.softmax(inputs)
        
        # Find the values to keep based on top k
        top_k_values, _ = tf.nn.top_k(softmax_x, k=self.experts_needed)
        threshold_value_topk = tf.expand_dims(top_k_values[..., -1], -1)  # Expand dimensions to match softmax_x
        mask_topk = tf.greater_equal(softmax_x, threshold_value_topk)

        # Mask based on the threshold
        threshold_expanded = tf.expand_dims(self.threshold, 0)  # Make threshold's shape compatible with softmax_x
        mask_threshold = tf.greater_equal(softmax_x, threshold_expanded)
        
        # Combine the two masks
        combined_mask = tf.logical_and(mask_topk, mask_threshold)

        # Apply the combined mask, only keeping activations that meet both conditions
        custom_output = tf.where(combined_mask, softmax_x, 0.0)

        return custom_output
    
class Expert_customize(Model):
    def __init__(self,hidden_shape_list,activation = 'relu',kernel_init = 'random_normal',drop_out_rate=0.0):
        super(Expert_customize, self).__init__()
        # self.hidden_shape_list = hidden_shape_list
        self.activation = activation
        self.kernel_init = kernel_init
        self.drop_out_rate = drop_out_rate
        self.dense_list = []
        for hidden_shape in hidden_shape_list:
            self.dense_list.append(Dense(units=hidden_shape,activation=self.activation,kernel_initializer=self.kernel_init))

    def call(self, inputs):
        for layer in self.dense_list:
            inputs = layer(inputs)
            inputs = tf.nn.dropout(inputs,self.drop_out_rate)
        # output = tf.nn.dropout(inputs,self.drop_out_rate)
        return inputs
    
class PUFs():
    def __init__(self, stages=64, similarity=2):
        self.pufs = []
        self.responses = []
        self.stages = stages
        self.seed = 1
        self.similarity = similarity
        self.challenges = None
        self.n_pufs = None
        self.hereroFF_PUF = []

    def add_XOR_PUF(self,k,num):
        for _ in range(num):
            # puf = XORArbiterPUF(n=self.stages,k=k,seed=self.seed,scale2=self.similarity)
            puf = XORArbiterPUF(n=self.stages,k=k,seed=self.seed)
            # seed increase
            self.seed += 10
            self.pufs.append(puf)

    def add_FF_PUF(self,ff,num):
        for _ in range(num):
            puf = FeedForwardArbiterPUF(n=self.stages,ff=ff,seed=self.seed)
            self.seed +=10
            self.pufs.append(puf)

    def add_XORFF_PUF(self,k,ff,num):
         for _ in range(num):
            puf = XORFeedForwardArbiterPUF(n=self.stages,k=k,ff=ff,seed=self.seed)
            self.seed +=10
            self.pufs.append(puf)

    def add_herero_XORFF_PUFs(self,k,ff,num):
        for i in range(num):
            FF_PUF = []
            for j in range(k):
                puf = FeedForwardArbiterPUF(n=self.stages,ff=ff[j],seed=self.seed)
                self.seed += 10
                FF_PUF.append(puf)
            self.hereroFF_PUF.append(FF_PUF)

    def add_interpose_PUFs(self, ks_up, ks_down, num):
        for i in range(num):
            k_up,k_down = ks_up, ks_down
            puf = InterposePUF(self.stages,k_down=k_down,k_up=k_up)
            self.seed += 10
            self.pufs.append(puf)

    def generate_crps(self, c_seed=2,N=[3000]):
        self.challenges = random_inputs(n=self.stages,N=N,seed=c_seed)
        self.n_pufs = len(self.pufs)+len(self.hereroFF_PUF)
        for puf in self.pufs:
            r = puf.eval(self.challenges)
            r = (1-r) // 2
            self.responses.append(r)
        for ff_pufs in self.hereroFF_PUF:
            rs = []
            for ff_puf in ff_pufs:
                r = ff_puf.eval(self.challenges)
                r = (1-r) // 2
                rs.append(r)
            rs = np.array(rs)
            rs = np.bitwise_xor.reduce(rs,axis=0)
            self.responses.append(rs)

    # def mask(self,mask)

        return self.challenges, self.responses