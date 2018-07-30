# Run file for different experiments

from transfer_learning.train import *
from transfer_learning.evaluate import *

if __name__ == '__main__':
    """
  
    Following "Unsupervised Domain Adaptation by Backpropagation, Ganin Y Lempitsky V", this script runs the three following experiments,
    performed on the MNIST dataset included in the tensorflow builtin library.
  
    - Experiment 1:
    A stantard CNN is trained on synthetic images (base images included in tensorflow) and tested on natural images (base images to which was
    added a white gaussian noise of a reasonable amplitude (half of each image variance)).
  
    - Experiment 3:
    A multitask CNN was trained on a mix dataset (containing both natural and synthetic images), following the paper's idea: One task was to
    identify whether an individual (image) was synthetic or natural, the other task was to classify digits the standard way. Including a gradient
    reversal layer (placed just before the feature extractor), the network therefore learns features which are invariant per change of domain
    (i.e. features useful for both synthetic and natural images). The trained CNN is then tested on natural images
  
    - Experiment 2:
    A standard CNN is pretrained on natural images. The feature extractor is then frozen and just the final classifier (fully connected simplex)
    is re-trained on the natural dataset. The network is then tested on natural images.
  
    The network was tuned using AdamOptimizer without learning rate decay. Batch Normalization was tested but showed no improvement. Batch size is 50.
    The dataset is composed of 40.000 synthetic images for train, 20.000 natural images for train, 10.000 natural images for test.
  
  
    Experiment should be run separately though, for a better control over saving directories etc
    """

    # Load dataset
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mixed_mnist = Multilabel_dataset(mnist)

    ###################################
    ########   Experiment 1 ###########
    ###################################
    train_model(
        dataset=mixed_mnist,
        model='standard_cnn',
        name='Exp1',
        optimizer='AdamOptimizer',
        BN=False,
        trainable=True,
        train_steps=2400)

    # Make sure to be in the right directory
    eval_model(
        dataset=mixed_mnist,
        name='Exp1')

    """
    ###################################
    ########   Experiment 2 ###########
    ###################################
    #Prepare data, not very neat so far.
    train_model(
      dataset=mixed_mnist, 
      model='standard_cnn',
      name='Exp2', 
      optimizer='AdamOptimizer', 
      BN=False, 
      trainable=True)
  
    pretrain_dataset = mixed_mnist
    pretrain_dataset.train_data = np.apply_along_axis(add_gaussian_noise,1,pretrain_dataset.train_data).astype(np.float32)[:10000]
    pretrain_dataset.train_labels = pretrain_dataset.train_labels[:10000]
    
    train_model(
      dataset=pretrain_dataset, 
      model='standard_cnn',
      name='Exp2', #make sure to pickup the previous model
      optimizer='AdamOptimizer', 
      BN=True, 
      trainable=False,
      train_steps=800) #This freezes the features extractor  
  
    eval_model(
      dataset=mixed_mnist,
      name='Exp2')
    """

    ###################################
    ########   Experiment 3 ###########
    ###################################
    # Multilabel the dataset for multitasking
    mixed_mnist.Multilabel()
    train_model(
        dataset=mixed_mnist,
        model='bitask_cnn',
        name='Exp3',
        optimizer='AdamOptimizer',
        BN=False,
        train_steps=2400)

    # Make sure to be in the right directory
    eval_model(
        dataset=mixed_mnist,
        name='Exp3')
