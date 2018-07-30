from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from transfer_learning.tools import *
import tensorflow as tf
import pickle
import argparse


def eval_model(dataset, name=''):
    # Get the path from command-line, current directory if not specified
    parser = argparse.ArgumentParser(description='Evaluation process')
    parser.add_argument('--expe_name', default='/Users/remydubois/Documents/tensorflow/saved/' + name + '/')
    args = parser.parse_args()
    model_dir = args.expe_name

    print('Evaluating checkpoints from ' + str(model_dir))

    # Load the estimator object (Not clear how that works)
    with open(model_dir + 'estimator.pkl', 'rb') as f:
        model = pickle.load(f)

    #########################
    #       Evaluate        #
    #########################
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": dataset.eval_data},
        y=dataset.eval_labels,
        # num_epochs=1, #None
        batch_size=200,
        shuffle=False)
    checkpoints = re.findall(pattern='model\.ckpt-[0-9]*', string=''.join(os.listdir(model_dir)))
    # Filter doublons
    checkpoints = list(set(checkpoints))
    # Sort
    checkpoints.sort(key=lambda x: (len(x), x))
    # Loop to eval
    for ck in checkpoints:
        eval_results = model.evaluate(
            input_fn=eval_input_fn,
            checkpoint_path=model_dir + ck,
            steps=1  # Eval on a whole epoch (time consuming)
        )


if __name__ == '__main__':
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mixed_mnist = Multilabel_dataset(mnist)

    eval_model(mixed_mnist)
