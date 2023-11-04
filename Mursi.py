import numpy as np
from pypuf.simulation import *
# import pypuf.metrics as pm
from pypuf.io import random_inputs
import pypuf.io, pypuf.simulation
from sklearn.model_selection import train_test_split
# import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras.metrics as metrics
import random
from mmoe import MMoE
import time
import psutil
process = psutil.Process()
memory_info_start = process.memory_info()
class PUFs():
    def __init__(self, stages=64, similarity=2):
        self.pufs = []
        self.responses = []
        self.stages = stages
        self.seed = 1
        self.similarity = similarity
        self.challenges = None
        self.n_pufs = None

    def add_XOR_PUF(self,k,num):
        for _ in range(num):
            # puf = RXORArbiterPUF(n=self.stages,k=k,seed=self.seed,scale2=self.similarity)
            puf = XORArbiterPUF(n=self.stages,k=k,seed=self.seed)
            # seed increase
            self.seed += 1
            self.pufs.append(puf)

    def add_FF_puf(self,ff,num):
        for _ in range(num):
            puf = FeedForwardArbiterPUF(n=self.stages,ff=ff,seed=self.seed)
            self.seed +=1
            self.pufs.append(puf)

    def add_XORFF_PUF(self,k,ff,num):
         for _ in range(num):
            puf = XORFeedForwardArbiterPUF(n=self.stages,k=k,ff=ff,seed=self.seed)
            self.seed +=1
            self.pufs.append(puf)

    def generate_crps(self, c_seed=2,N=2000):
        self.challenges = random_inputs(n=self.stages,N=N,seed=c_seed)
        self.n_pufs = len(self.pufs)
        for puf in self.pufs:
            r = puf.eval(self.challenges)
            r = (1-r) // 2
            self.responses.append(r)
        return self.challenges, self.responses
    
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


PUF_list = PUFs(stages=64, similarity=2)
PUF_list.seed = random.randint(1,100)
print("random seed",PUF_list.seed)
# PUF_list.seed = 69
k = 7
epochs = 20
N_crp = [10000, # 2
         30000, # 3
         100000, # 4
         400000, # 5
         1000000, # 6
         5000000 #7
         ]
## k=6 seed=69 N=1M fail

PUF_list.add_XOR_PUF(k=k,num=2)
# PUF_list.add_XORFF_PUF(4,[(20,50)],1)
# PUF_list.add_FF_puf([(32,50)],3)
# PUF_list.add_XORFF_PUF(2,[(20,50)],2)
# PUF_list.add_FF_puf([(20,50)],8)
[c, responses] = PUF_list.generate_crps(10,N_crp[k-2])
c = get_parity_vectors2(c)
# c = XORArbiterPUF.transform_atf(c, k=1)[:, 0, :]
# c = np.concatenate([c,get_parity_vectors2(c)],axis=1)
# c = np.concatenate([c,XORArbiterPUF.transform_atf(c, k=1)[:, 0, :]],axis=1)
responses = np.array(responses)
# responses = np.array(responses,dtype=np.float)
# c = np.array(c,dtype=np.float)
train_c,test_c,train_r,test_r = train_test_split(c,responses.T,test_size=0.2, random_state=42)
train_r_groups = [train_r[:,i].reshape(-1,1) for i in range(PUF_list.n_pufs)]
test_r_groups = [test_r[:,i].reshape(-1,1) for i in range(PUF_list.n_pufs)]

from tensorflow.keras.optimizers import RMSprop
def lr_schedule(epoch, lr):
    if epoch < 30:
        return 0.01
    elif epoch < 60:
        return 0.005
    elif epoch < 90:
        return 0.001
    else:
        return 0.0001
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, min_delta=0.01, mode='max')

def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.binary_crossentropy(.5 - .5 * y_true, .5 - .5 * y_pred)

def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.keras.metrics.binary_accuracy(.5 - .5 * y_true, .5 - .5 * y_pred)

input_dim = 64
# activation = 'relu'
activation = 'tanh'
kernel_init = 'random_normal'
# kernel_init = 'glorot_uniform'
n_n = min(k,5)
input_layer = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(2**(n_n-1), activation=activation,kernel_initializer=kernel_init)(input_layer)
x = tf.keras.layers.Dense(2**(n_n), activation=activation,kernel_initializer=kernel_init)(x)
x = tf.keras.layers.Dense(2**(n_n-1), activation=activation,kernel_initializer=kernel_init)(x)
# x = tf.keras.layers.Dense(h_n, activation=activation,kernel_initializer=kernel_init)(x)
output_layer1 = tf.keras.layers.Dense(1, activation='sigmoid',kernel_initializer=kernel_init)(x)
# output_layer2 = tf.keras.layers.Dense(1, activation='sigmoid',kernel_initializer=kernel_init)(x)

# Build
model = tf.keras.Model(inputs=input_layer, outputs=output_layer1)
optimizer1 = RMSprop(learning_rate=0.01,rho=0.9)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer3 = 'adam'
# Compile
model.compile(optimizer=optimizer3,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
model.summary()
# Train
# batch_size = N_crp[k-2]//100
batch_size = [500,500,500,1000,1000,10000]


start = time.time()
# model.fit(train_c, train_r_groups[0], validation_data=(test_c, test_r_groups[0]), batch_size=batch_size, epochs=epochs)
model.fit(train_c, train_r_groups[0], validation_data=(test_c, test_r_groups[0]), batch_size=batch_size[k-2], epochs=epochs, callbacks=[lr_scheduler])
end = time.time()
run_time = end-start
print("random seed",PUF_list.seed)
print(f"Run time {run_time} s")

process = psutil.Process()
memory_info = process.memory_info()

# Memory cost
print(f"开始内存使用量: {(memory_info.rss-memory_info_start.rss) / (1024 ** 3):.2f} GiB")