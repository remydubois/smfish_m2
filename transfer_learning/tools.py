#tools
import numpy as np
from transfer_learning.blurr import add_gaussian_noise
import tensorflow as tf

class Multilabel_dataset():
#Takes tensorflow-like dataset as argument

  def __init__(self, dataset):
    self.train_data = dataset.train.images
    self.eval_data = np.apply_along_axis(add_gaussian_noise,1,dataset.test.images).astype(np.float32)
    self.train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
    self.eval_labels = np.asarray(dataset.test.labels, dtype=np.int32)

  def Multilabel(self, ratio=0.33):
    #Pick up indices of natural images
    train_natural_inds = np.random.randint(0,self.train_data.shape[0],round(self.train_data.shape[0]*ratio))

    #Create labels for the second task
    domain_label_train = np.asarray([1 if i in train_natural_inds else 0 for i in range(self.train_data.shape[0])])

    #Blurr the natural indices in the dataset
    for (i,m) in enumerate(self.train_data):
      self.train_data[i] = add_gaussian_noise(self.train_data[i]) if i in train_natural_inds else self.train_data[i]

    domain_label_eval = np.asarray([1 for i in range(self.eval_data.shape[0])])

    #Then build labels
    self.train_labels = {'digits':np.asarray(self.train_labels, dtype=np.int32), 'domain':domain_label_train}
    self.eval_labels = {'digits':np.asarray(self.eval_labels, dtype=np.int32), 'domain':domain_label_eval}


    
def parse_kwargs(**kwargs):
  params = {}
  param_default = {
  'optimizer':tf.train.GradientDescentOptimizer,
  'BN':False,
  'trainable':True,
  'train_steps':1600,
  'lr_decay':False
  }
  #parse if optimizer argument is passed
  for n in param_default:
    if n in kwargs:
      try:
        params[n] = getattr(tf.train,kwargs[n])
      except:
        params[n] = kwargs[n]
    else:
      params[n] = param_default[n]

  return params
