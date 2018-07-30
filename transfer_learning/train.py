from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from transfer_learning.tools import *
from transfer_learning import model_fns
import tensorflow as tf
import pickle

tf.logging.set_verbosity(tf.logging.INFO)


def train_model(dataset, model, name='', **kwargs):
    params = parse_kwargs(**kwargs)

    """
    parser = argparse.ArgumentParser(description='train process')
    parser.add_argument('--train_steps', default = 1600, type=int)
    args = parser.parse_args()
    """

    # Create the Estimator
    # Build the model function by extracting the parameters
    model_fn = getattr(model_fns, model)

    if name == '':
        model_dir = '/Users/remydubois/Documents/tensorflow/saved/' + str(model) + '-' + '-'.join(
            map(lambda k: str(k), kwargs.values()))
    else:
        model_dir = '/Users/remydubois/Documents/tensorflow/saved/' + name

    #########################
    #      Instantiate      #
    #########################
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(
            save_checkpoints_steps=40,
            save_summary_steps=40,
            keep_checkpoint_max=40),
        params=params
    )

    #########################
    #         Train         #
    #########################
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": dataset.train_data},
        y=dataset.train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=params['train_steps'],
        hooks=[])

    # Save the estimator object. This is not a model properly speaking.
    with open(model_dir + '/estimator.pkl', 'wb+') as f:
        pickle.dump(mnist_classifier, f)

    #########################
    #      Write info       #
    #########################
    with open(model_dir + '/info.txt', 'w+') as f:
        f.write("####################################################\n")
        f.write("Test CNN written for MNIST classification adaptation\n")
        f.write('params:\n')
        for k in kwargs.items():
            f.write(str(k) + '\n')
        f.close()
