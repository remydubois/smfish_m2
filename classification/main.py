import argparse
from models import *
from utils import *
from callbacks import *
from keras.optimizers import Adam, SGD
from mnist_poc import *
import shutil
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tqdm
from sklearn.decomposition import IncrementalPCA
import time
import numpy
import luigi
import math
from keras.models import load_model
from MulticoreTSNE import MulticoreTSNE
from sklearn.utils import class_weight
import pickle
import json
from Preprocessing import Merge, StackSimulations, inherits
import pandas
import keras
from parameters import input_shape
from callbacks import MyTensorBoard, PredictionHistory
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, model_from_json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import timeline
from keras.utils import multi_gpu_model
from sklearn.manifold import TSNE
from keras.backend.tensorflow_backend import set_session
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 50

if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    LOCAL = '/Users/remydubois/Dropbox/Remy/results/'
else:
    LOCAL = '/cbio/donnees/rdubois/results/'


# config = keras.backend.tf.ConfigProto(intra_op_parallelism_threads=4,
#                                       inter_op_parallelism_threads=2,
#                                       log_device_placement=True)
# keras.backend.set_session(keras.backend.tf.Session(config=config))

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Load dataset into batch generators:
@inherits(Merge)
class Split(luigi.Task):
    """
    Be careful as Luigi will not check whether this task was called with the same arguments when checking dependencies.

    This task is necessary to keep track of the dataset used and analyse results correctly. 
    """

    def requires(self):
        return self.clone(Merge)

    def output(self):
        return [
            luigi.LocalTarget('/mnt/data40T_v2/rdubois/data/tmp/train_%s.pkl' %
                              self.simulation_folders.replace('/', '_')),
            luigi.LocalTarget('/mnt/data40T_v2/rdubois/data/tmp/test_%s.pkl' %
                              self.simulation_folders.replace('/', '_')),
            luigi.LocalTarget('/mnt/data40T_v2/rdubois/data/tmp/eval_%s.pkl' %
                              self.simulation_folders.replace('/', '_'))
        ]

    def run(self):
        list_of_dfs = [pandas.read_pickle(i.path) for i in self.input()]
        dataset = pandas.concat(list_of_dfs, ignore_index=True)

        le = LabelEncoder()
        dataset['labels'] = le.fit_transform(dataset.pattern_name)

        train_dataset, test_dataset, validation_dataset = numpy.split(
            dataset.sample(frac=1), [int(.6 * len(dataset)), int(.8 * len(dataset))])

        print('Writing sets in "/mnt/data40T_v2/rdubois/data/tmp/".')
        for o, df in zip(self.output(), [train_dataset, test_dataset, validation_dataset]):
            df.to_pickle(o.path)


@inherits(Split)
class MixandSplit(luigi.Task):
    repeatsyn = luigi.FloatParameter(default=2.)
    repeatreal = luigi.FloatParameter(default=1.)

    def requires(self):
        return self.clone(Split)

    def output(self):
        return [
            luigi.LocalTarget('/mnt/data40T_v2/rdubois/data/tmp/train_mix_%s_%s_%s.pkl' % (str(
                self.repeatsyn), str(self.repeatreal), self.simulation_folders.replace('/', '_'))),
            luigi.LocalTarget('/mnt/data40T_v2/rdubois/data/tmp/test_mix_%s_%s_%s.pkl' % (str(
                self.repeatsyn), str(self.repeatreal), self.simulation_folders.replace('/', '_')))
        ]

    def run(self):

        with open('/mnt/data40T_v2/rdubois/data/Experimental/merged_file.pkl', 'rb') as f:
            exp_dataset = pickle.load(f)
        with open(self.input()[0].path, 'rb') as f:
            eval_dataset = pickle.load(f)

        exp_dataset['n_RNA'] = exp_dataset['pos'].apply(
            lambda m: numpy.bincount(m[-1])[-1])
        vc = pandas.cut(exp_dataset['n_RNA'], numpy.arange(
            0, exp_dataset['n_RNA'].max() + 50, 50))
        frequencies = vc.value_counts()

        eval_bins = pandas.cut(eval_dataset.n_RNA, numpy.arange(
            0, exp_dataset['n_RNA'].max() + 50, 50))
        idx = []

        for b, f in tqdm.tqdm(frequencies.iteritems(), total=frequencies.shape[0], disable=False):
            try:
                # print(f)
                omega = eval_bins[eval_bins == b].sample(
                    n=int(f * (self.repeatsyn + 1)), replace=True)
                idx.extend(omega.index)
            except:
                pass
        extract = eval_dataset.loc[idx]

        extract['domain_label'] = 0
        exp_dataset['domain_label'] = 1

        syn_train, syn_test = train_test_split(
            extract, train_size=self.repeatsyn / (1 + self.repeatsyn), test_size=1 / (1 + self.repeatsyn))

        exp_train = exp_dataset.sample(frac=self.repeatreal, replace=True)
        mix_train = pandas.concat(
            [exp_train, syn_train], ignore_index=True, sort=False)
        mix_train = mix_train.sample(frac=1)

        mix_train['pattern_label'] = -1
        le = LabelEncoder()
        mix_train.pattern_label[mix_train.domain_label == 0] = le.fit_transform(
            mix_train['pattern_name'][mix_train.domain_label == 0])
        mix_train.fillna('Nan')

        exp_test = exp_dataset.sample(frac=1, replace=True)
        mix_test = pandas.concat(
            [exp_test, syn_test], ignore_index=True, sort=False)
        mix_test = mix_test.sample(frac=1)

        mix_test['pattern_label'] = -1
        mix_test.pattern_label[mix_test.domain_label == 0] = le.transform(
            mix_test['pattern_name'][mix_test.domain_label == 0])
        mix_test.fillna('Nan')

        # train, test = train_test_split(mix, train_size=0.7, test_size=0.3)

        mix_train.to_pickle(self.output()[0].path)
        mix_test.to_pickle(self.output()[1].path)


@inherits(Split)
class Train(luigi.Task):
    logdir = luigi.Parameter(default='')
    gpu = luigi.Parameter(default='0')
    channels = luigi.Parameter(default='all')
    model = luigi.Parameter(default='squeezenet')
    batchsize = luigi.IntParameter(default=50)
    pretrain = luigi.IntParameter(default=1)

    def requires(self):
        return self.clone(Split)

    def output(self):
        return luigi.LocalTarget()

    def run(self):
        #######################################################################
        # Config
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = False
        # set_session(tf.Session(config=config))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        #######################################################################

        train_dataset = pandas.read_pickle(self.input()[0].path)[
            ['pos', 'labels']].values
        test_dataset = pandas.read_pickle(self.input()[1].path)[
            ['pos', 'labels']].values
        numpy.random.shuffle(train_dataset)
        numpy.random.shuffle(test_dataset)

        # Another warning: num classes can be miscalculated because it only
        # looks at train
        num_classes = numpy.unique(train_dataset[:, 1]).shape[0]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          numpy.unique(
                                                              train_dataset[:, 1]),
                                                          train_dataset[:, 1])

        train_generator = _batch_generator_binary_images(train_dataset,
                                                         self.batchsize,
                                                         num_classes,
                                                         None,
                                                         False
                                                         )
        test_generator = _batch_generator_binary_images(test_dataset,
                                                        self.batchsize,
                                                        num_classes,
                                                        None,
                                                        False
                                                        )

        input_ = Input(shape=next(train_generator)[0].shape[1:])
        if self.model.lower() == 'squeezenet':
            output_ = SqueezeNetOutput(input_, num_classes, bypass='simple')
        elif self.model.lower() == 'inception':
            output_ = InceptionOutput(
                input_, num_classes, pretrain=self.pretrain)
        elif self.model.lower() == 'ae':
            output_ = AE(input_)
        else:
            raise ValueError('Unknown model.')

        gpus = self.gpu.split(',')
        if len(gpus) > 1:
            # with tf.device('/cpu:0'):
            model = Model(input_, output_, name=self.model)
            # Keras recommendation for NV links - connected gpus
            model = multi_gpu_model(model, gpus=len(
                gpus), cpu_merge=False, cpu_relocation=False)
        else:
            model = Model(input_, output_, name=self.model)
            # embedder = Model(input_, embedding_layer)
        # print([l.name for l in model.layers])

        adam = Adam(lr=1e-4)

        logdir = LOCAL + self.logdir

        if not os.path.exists(logdir):
            os.makedirs(logdir)
        else:
            try:
                print('Picking up checkpoint')
                model.load_weights(logdir + '/model-ckpt')
            except OSError:
                pass

        #######################################################################
        # Misc
        # Tensorboard
        tb = MyTensorBoard(log_dir=logdir,
                           histogram_freq=0,
                           write_batch_performance=True,
                           write_grads=False
                           # embeddings_freq=20,
                           # embeddings_layer_names=['globalaveragepooling']
                           # write_images=True
                           )
        # Checkpoint
        checkpointer = ModelCheckpoint(
            filepath='%s/model-ckpt' % logdir, verbose=0, save_best_only=False)
        earl = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
        # reduceLROnplateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=0.001)
        # Timeline
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # Prediction History
        ph = PredictionHistory(logdir, test_generator,
                               test_dataset.shape[0] // self.batchsize)
        #######################################################################

        model.compile(loss='categorical_crossentropy' if self.model != 'ae' else 'binary_crossentropy',
                      optimizer=adam,
                      metrics=['acc'],
                      options=run_options,
                      run_metadata=run_metadata
                      )

        # Fit on generator
        # with K.tf.device('/gpu:0'):
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_dataset.shape[0] // BATCH_SIZE,
            callbacks=[tb, checkpointer, reduce_lr, earl],
            validation_data=test_generator,
            validation_steps=test_dataset.shape[0] // BATCH_SIZE,
            epochs=50,
            verbose=1,
            max_queue_size=5,
            workers=1,
            use_multiprocessing=False,
            class_weight=class_weights
        )

        # ########################################################################################################
        # # Save architecture
        # js = model.to_json()
        # with open(logdir + '/architecture.json', 'w') as fout:
        #     fout.write(js)

        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open('%s/timeline.ctf.json' % logdir, 'w') as f:
        #     f.write(trace.generate_chrome_trace_format())

    def complete(self):
        return False


@inherits(MixandSplit)
class TrainDA(luigi.Task):
    lam = luigi.FloatParameter(default=0.5)
    lambdaloc = luigi.FloatParameter(default=5.)
    # logdir = luigi.Parameter()
    gpu = luigi.Parameter(default='None')
    batchsize = luigi.IntParameter(default=60)
    domainweight = luigi.FloatParameter(default=0.5)
    fex = luigi.Parameter(default='simple')
    epochs = luigi.IntParameter(default=50)

    def requires(self):
        return self.clone(MixandSplit)

    def output(self):
        return []

    def run(self):
        if self.gpu != 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # read in
        train_dataset = pandas.read_pickle(self.input()[0].path)[
            ['pos', 'pattern_label', 'domain_label']].values
        test_dataset = pandas.read_pickle(self.input()[1].path)[
            ['pos', 'pattern_label', 'domain_label']].values

        num_classes = train_dataset[:, 1].max() + 1
        pattern_class_weights = class_weight.compute_class_weight('balanced',
                                                                  numpy.unique(
                                                                      train_dataset[:, 1]),
                                                                  train_dataset[:, 1])
        # domain_class_weights = class_weight.compute_class_weight('balanced',
        #                                          numpy.unique(train_dataset[:, 2]),
        #                                          train_dataset[:, 2])
        train_generator = datagenerator_DA(train_dataset,
                                           self.batchsize,
                                           num_classes,
                                           )
        test_generator = datagenerator_DA(test_dataset,
                                          min(2000, test_dataset.shape[0]),
                                          num_classes,
                                          )
        Xtest, ytest = next(test_generator)

        n_steps_train = train_dataset.shape[0] // self.batchsize + 1
        n_steps_test = test_dataset.shape[0] // self.batchsize + 1

        # Misc
        logdir = LOCAL + 'sqn_smfish_dw_' + str(self.domainweight) + '_lam_' + str(
            self.lam) + '_lamloc_' + str(self.lambdaloc) + '_fex_' + str(self.fex) + '/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        else:
            try:
                print('Picking up checkpoint')
                # sqn_da.load_weights(logdir+'/model-ckpt')
            except OSError:
                pass

        lam = K.variable(numpy.array(0., dtype='float64'), name='lambda_grl')

        tb = MyTB2(log_dir=logdir,
                   histogram_freq=1,
                   write_grads=True,
                   extra=lam,
                   record_images=True
                   )
        ck = ModelCheckpoint(filepath='%smodel-ckpt' % logdir,
                             verbose=0, save_best_only=True, save_weights_only=True)
        earl = EarlyStopping(
            monitor='val_pattern_classifier_loss', min_delta=0, patience=15)
        # lambdaloc sets in how many epochs Lambda will reach its maximum value
        # (scale parameter)
        lc = LambdaChanger(lam, loc=self.lambdaloc *
                           n_steps_train, scale=self.lam)

        # Model
        input_ = Input(shape=next(train_generator)[0].shape[1:])
        # output_ = SqueezeNetOutputDA(input_, 10, lam=lam, bypass=None)
        # sqn_da = Model(input_, output_)
        if self.fex == 'simple':
            fex = smf_feature_ex(input_)
        if self.fex == 'sqn':
            fex = sqn_fex(input_)

        output_1 = pattern_classifier(fex, l=1. - lam)
        output_2 = domain_classifier(fex, l=lam)

        sqn_da = Model(input_, [output_1, output_2])
        # # ad = Adam(lr=0.0001, clipnorm=1.)
        sgd = SGD(lr=1e-3, clipnorm=1., momentum=0.9, nesterov=True)
        sqn_da.compile(loss={'pattern_classifier': patternloss, 'domain_classifier': 'categorical_crossentropy'},
                       optimizer=sgd,
                       metrics={'pattern_classifier': myacc,
                                'domain_classifier': 'acc'},
                       loss_weights={
                           'pattern_classifier': 1 - self.domainweight, 'domain_classifier': self.domainweight}
                       )

        # Fit
        try:
            sqn_da.fit_generator(train_generator,
                                 steps_per_epoch=n_steps_train,
                                 epochs=self.epochs,
                                 callbacks=[lc, tb, ck],
                                 validation_data=(Xtest, ytest),
                                 max_queue_size=10,
                                 class_weight={
                                     'pattern_classifier': pattern_class_weights}
                                 )
        except (KeyboardInterrupt, SystemError):
            pass

        sqn_da.load_weights('%smodel-ckpt' % logdir)

        embedder = Model(input_, [sqn_da.layers[[l.name for l in sqn_da.layers].index(
            'embedding_layer')].output, sqn_da.layers[[l.name for l in sqn_da.layers].index('pattern_classifier')].output])

        # test_generator = datagenerator(numpy.kron(Xtest, numpy.ones((1,2,2,1))), *ytest, batch_size=self.batchsize)
        features, pred_patterns = embedder.predict(Xtest)

        pred_patterns = numpy.argmax(pred_patterns, axis=1)
        rightwrong = (pred_patterns == numpy.argmax(
            ytest[0][:Xtest.shape[0]], axis=1)).astype(int)
        tsne = MulticoreTSNE(n_components=2, n_jobs=24,
                             verbose=111, n_iter=500)
        plan = tsne.fit_transform(features)

        # saver = tf.train.Saver([plan])
        embedding_var = K.variable(plan, name='features')
        embeddings_metadata_path = os.path.join(logdir, 'metadata.tsv')

        with open(embeddings_metadata_path, 'w') as f:
            f.write("Index\tPattern\tDomain\tRightWrong\n")
            for index, label in enumerate(zip(test_dataset[:Xtest.shape[0], 1], test_dataset[:Xtest.shape[0], 2], rightwrong)):
                f.write("%d\t%d\t%d\t%d\n" %
                        (index, label[0], label[1], label[2]))

        # K.set_value(embedding_var, plan)
        saver = tf.train.Saver([embedding_var])
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = embeddings_metadata_path
        writer = tf.summary.FileWriter(logdir)
        projector.visualize_embeddings(writer, config)
        saver.save(K.get_session(), logdir + 'embeddings.ckpt', 1)


# @inherits(MixandSplit)
class TrainMNIST(luigi.Task):
    lam = luigi.FloatParameter(default=.5)
    lambdaloc = luigi.FloatParameter(default=5.)
    epochs = luigi.IntParameter(default=120)
    # logdir = luigi.Parameter()
    gpu = luigi.Parameter(default='None')
    batchsize = luigi.IntParameter(default=100)
    domainweight = luigi.FloatParameter(default=0.5)
    da = luigi.IntParameter(default=0)
    noise = luigi.FloatParameter(default=3)
    fex = luigi.Parameter(default='simple')

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        if self.gpu != 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        Xtrain, Xtest, ytrain, y_test = load_mnist()
        ytest = y_test.copy()
        noise = numpy.load(SOURCE + 'noise.npy') * self.noise
        noise = numpy.repeat(noise, 2, axis=0)[:Xtrain.shape[0]]
        noise = shuffle(noise, random_state=123)

        # 1 for real images
        synratio = 0.5
        realtrain = numpy.array([True, ] * math.ceil((1 - synratio) * Xtrain.shape[0]) + [
                                False, ] * math.ceil(synratio * Xtrain.shape[0]))[:Xtrain.shape[0]]
        realtrain = shuffle(realtrain, random_state=1234)
        # realtest = numpy.array([True, ] * math.floor((1 - synratio) * Xtest.shape[0]) + [False, ] * math.ceil(synratio * Xtest.shape[0]))
        # numpy.random.shuffle(realtest)
        # realtrain = numpy.random.randint(0, 2, size=Xtrain.shape[0]).astype(bool)
        numpy.random.seed(12345)
        realtest = numpy.random.randint(0, 2, size=Xtest.shape[0]).astype(bool)

        Xtest[realtest] += noise[:Xtest.shape[0]][realtest]
        # This ensures that loss and acc are computed on real examples during
        # testing
        ytest[numpy.logical_not(realtest)] *= 0
        Xtrain[realtrain] += noise[realtrain]
        # This ensures that loss & acc are computed only on synthetic examples
        # during training
        ytrain[realtrain] *= 0

        # ytrain = [numpy.hstack((ytrain, to_categorical(realtrain.astype(int), num_classes=2))), to_categorical(realtrain.astype(int), num_classes=2)]
        # ytest = [numpy.hstack((ytest, to_categorical(realtest.astype(int), num_classes=2))), to_categorical(realtest.astype(int), num_classes=2)]
        ytrain = [ytrain, to_categorical(realtrain.astype(int), num_classes=2)]
        ytest = [ytest, to_categorical(realtest.astype(int), num_classes=2)]

        train_generator = datagenerator(numpy.kron(Xtrain, numpy.ones(
            (1, 2, 2, 1))), *ytrain, batch_size=self.batchsize)
        test_generator = datagenerator(numpy.kron(Xtest, numpy.ones(
            (1, 2, 2, 1))), *ytest, batch_size=self.batchsize)

        n_steps_train = Xtrain.shape[0] // self.batchsize
        n_steps_test = Xtest.shape[0] // self.batchsize

        # Misc
        logdir = LOCAL + 'MNIST/sqn_mnist_dw_' + str(self.domainweight) + '_lam_' + str(
            self.lam) + '_noise_' + str(self.noise) + '_lamloc_' + str(self.lambdaloc) + '_fex_' + str(self.fex) + '/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        else:
            try:
                print('Picking up checkpoint')
                # sqn_da.load_weights(logdir+'/model-ckpt')
            except OSError:
                pass

        lam = K.variable(numpy.array(0., dtype='float64'), name='lambda_grl')

        tb = MyTB2(log_dir=logdir,
                   histogram_freq=1,
                   write_grads=True,
                   extra=lam,
                   record_images=True
                   )
        ck = ModelCheckpoint(filepath='%smodel-ckpt' % logdir,
                             verbose=0, save_best_only=True, save_weights_only=True)
        earl = EarlyStopping(
            monitor='val_pattern_classifier_loss', min_delta=0, patience=5)

        # Model
        input_ = Input(shape=next(train_generator)[0].shape[1:])
        # output_ = SqueezeNetOutputDA(input_, 10, lam=lam, bypass=None)
        # sqn_da = Model(input_, output_)
        if self.fex == 'simple':
            fex = feature_ex(input_)
        if self.fex == 'sqn':
            fex = sqn_fex(input_)

        output_1 = digit_classifier(fex, l=1. - lam)
        output_2 = domain_classifier(fex, l=lam)

        sqn_da = Model(input_, [output_1, output_2])
        # # ad = Adam(lr=0.0001, clipnorm=1.)
        sgd = SGD(lr=1e-3, clipnorm=1., momentum=0.9, nesterov=True)
        sqn_da.compile(loss={'pattern_classifier': patternloss, 'domain_classifier': 'categorical_crossentropy'},
                       optimizer=sgd,
                       metrics={'pattern_classifier': myacc,
                                'domain_classifier': 'acc'},
                       loss_weights={'pattern_classifier': 1,
                                     'domain_classifier': 1}
                       )

        # lambdaloc sets in how many epochs Lambda will reach its maximum value
        # (scale parameter)
        lc = LambdaChanger(lam, loc=self.lambdaloc *
                           n_steps_train, scale=self.lam)

        # Fit
        try:
            sqn_da.fit(numpy.kron(Xtrain, numpy.ones((1, 2, 2, 1))),
                       ytrain,
                       batch_size=100,
                       epochs=self.epochs,
                       callbacks=[lc, tb, ck],
                       validation_data=(numpy.kron(
                           Xtest, numpy.ones((1, 2, 2, 1))), ytest)
                       )
        except (KeyboardInterrupt, SystemError):
            pass

        sqn_da.load_weights('%smodel-ckpt' % logdir)

        embedder = Model(input_, [sqn_da.layers[[l.name for l in sqn_da.layers].index(
            'embedding_layer')].output, sqn_da.layers[[l.name for l in sqn_da.layers].index('pattern_classifier')].output])

        # test_generator = datagenerator(numpy.kron(Xtest, numpy.ones((1,2,2,1))), *ytest, batch_size=self.batchsize)
        features, pred_patterns = embedder.predict(
            numpy.kron(Xtest, numpy.ones((1, 2, 2, 1))))

        pred_digits = numpy.argmax(pred_patterns, axis=1)
        rightwrong = (pred_digits == numpy.argmax(y_test, axis=1)).astype(int)
        tsne = MulticoreTSNE(n_components=2, n_jobs=24,
                             verbose=111, n_iter=500)
        plan = tsne.fit_transform(features)

        # saver = tf.train.Saver([plan])
        embedding_var = K.variable(plan, name='features')
        embeddings_metadata_path = os.path.join(logdir, 'metadata.tsv')

        with open(embeddings_metadata_path, 'w') as f:
            f.write("Index\tPattern\tDomain\tRightWrong\n")
            for index, label in enumerate(zip(numpy.argmax(y_test, axis=1), numpy.argmax(ytest[1], axis=1), rightwrong)):
                f.write("%d\t%d\t%d\t%d\n" %
                        (index, label[0], label[1], label[2]))

        # K.set_value(embedding_var, plan)
        saver = tf.train.Saver([embedding_var])
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = embeddings_metadata_path
        writer = tf.summary.FileWriter(logdir)
        projector.visualize_embeddings(writer, config)
        saver.save(K.get_session(), logdir + 'embeddings.ckpt', 1)


class ReconstructorMNIST(luigi.Task):
    lam = luigi.FloatParameter(default=.5)
    lambdaloc = luigi.FloatParameter(default=5.)
    epochs = luigi.IntParameter(default=120)
    # logdir = luigi.Parameter()
    gpu = luigi.Parameter(default='None')
    batchsize = luigi.IntParameter(default=100)
    domainweight = luigi.FloatParameter(default=0.5)
    da = luigi.IntParameter(default=0)
    noise = luigi.FloatParameter(default=5)
    fex = luigi.Parameter(default='simple')

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        if self.gpu != 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        Xtrain, Xtest, ytrain, y_test = load_mnist()
        # Pad to make round numbers in downsampling
        Xtrain = numpy.pad(
            Xtrain, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
        Xtest = numpy.pad(
            Xtest, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
        ytest = y_test.copy()
        noise = numpy.load(SOURCE + 'noise.npy') * self.noise
        noise = numpy.pad(
            noise, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
        noise = numpy.repeat(noise, 2, axis=0)[:Xtrain.shape[0]]
        noise = shuffle(noise, random_state=123)

        # 1 for real images
        synratio = 0.5
        realtrain = numpy.array([True, ] * math.ceil((1 - synratio) * Xtrain.shape[0]) + [
                                False, ] * math.ceil(synratio * Xtrain.shape[0]))[:Xtrain.shape[0]]
        realtrain = shuffle(realtrain, random_state=1234)
        numpy.random.seed(12345)
        realtest = numpy.random.randint(0, 2, size=Xtest.shape[0]).astype(bool)

        Xtest[realtest] += noise[:Xtest.shape[0]][realtest]
        # This ensures that loss and acc are computed on real examples during
        # testing
        ytest[numpy.logical_not(realtest)] *= 0
        Xtrain[realtrain] += noise[realtrain]
        del noise
        # This ensures that loss & acc are computed only on synthetic examples
        # during training
        ytrain[realtrain] *= 0

        ytrain = [ytrain, Xtrain * realtrain.reshape(-1, 1, 1, 1)]
        ytest = [ytest, Xtest * realtest.reshape(-1, 1, 1, 1)]

        train_generator = datagenerator(
            Xtrain, *ytrain, batch_size=self.batchsize)
        test_generator = datagenerator(
            Xtest, *ytest, batch_size=self.batchsize)

        n_steps_train = Xtrain.shape[0] // self.batchsize
        n_steps_test = Xtest.shape[0] // self.batchsize

        # Misc
        logdir = LOCAL + 'MNIST/sqn_mnist_dw_' + str(self.domainweight) + '_lam_' + str(
            self.lam) + '_noise_' + str(self.noise) + '_lamloc_' + str(self.lambdaloc) + '_fex_' + str(self.fex) + '/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        else:
            shutil.rmtree(logdir)
            os.makedirs(logdir)

        # embeddings_metadata_path = os.path.join(logdir, 'metadata.tsv')

        # with open(embeddings_metadata_path,'w+') as f:
        #     f.write("Index\tPattern\tDomain\t")
        #     for index,label in enumerate(zip(numpy.argmax(y_test, axis=1), realtest)):
        #         f.write("%d\t%d\t%d\n" % (index, label[0], label[1]))

        lam = K.variable(numpy.array(0., dtype='float64'),
                         name='lambda_reconstruction')

        # def reconstructionloss(y_true, y_pred):
        #     return lam * K.mean(K.square(y_pred - y_true), axis=-1)

        tb = MyTB2(log_dir=logdir,
                   histogram_freq=1,
                   write_grads=True,
                   extra=lam,
                   record_images=True,
                   save_reconstructions=True,
                   project_features=False
                   )

        ck = ModelCheckpoint(filepath='%smodel-ckpt' % logdir,
                             verbose=0, save_best_only=True, save_weights_only=True)
        earl = EarlyStopping(
            monitor='val_pattern_classifier_loss', min_delta=0, patience=5)

        # Model
        input_ = Input(shape=next(train_generator)[0].shape[1:])
        # output_ = SqueezeNetOutputDA(input_, 10, lam=lam, bypass=None)
        # sqn_da = Model(input_, output_)
        # if self.fex=='simple':
        fex = feature_ex(input_)
        # if self.fex=='sqn':
        # fex = sqn_fex(input_)

        output_1 = digit_classifier(fex, l=1. - lam / 2)
        output_2 = reconstructor(fex, l=lam)

        sqn_da = Model(input_, [output_1, output_2])
        ad = Adam(lr=0.0001, clipnorm=1.)
        # sgd = SGD(lr=1e-3, clipnorm=1., momentum=0.9, nesterov=True)
        sqn_da.compile(loss={'pattern_classifier': patternloss, 'reconstruction_scaler': reconstructionloss},
                       optimizer=ad,
                       metrics={'pattern_classifier': myacc,
                                'reconstruction_scaler': 'acc'},
                       loss_weights={'pattern_classifier': 1,
                                     'reconstruction_scaler': 1}
                       )

        # lambdaloc sets in how many epochs Lambda will reach its maximum value
        # (scale parameter)
        lc = LambdaChanger(lam, loc=self.lambdaloc *
                           n_steps_train, scale=self.lam)

        # Fit
        try:
            sqn_da.fit(Xtrain,
                       ytrain,
                       batch_size=100,
                       epochs=self.epochs,
                       callbacks=[lc, tb, ck],
                       validation_data=(Xtest, ytest)
                       )
        except (KeyboardInterrupt, SystemError):
            pass

        # sqn_da.load_weights('%smodel-ckpt' % logdir)

        # embedder = Model(input_, [sqn_da.layers[[l.name for l in sqn_da.layers].index('embedding_layer')].output, sqn_da.layers[[l.name for l in sqn_da.layers].index('pattern_classifier')].output])

        # # test_generator = datagenerator(numpy.kron(Xtest, numpy.ones((1,2,2,1))), *ytest, batch_size=self.batchsize)
        # features, pred_patterns = embedder.predict(numpy.kron(Xtest, numpy.ones((1,2,2,1))))

        # pred_digits = numpy.argmax(pred_patterns, axis=1)
        # rightwrong = (pred_digits == numpy.argmax(y_test, axis=1)).astype(int)
        # tsne = MulticoreTSNE(n_components=2, n_jobs=24, verbose=111, n_iter=500)
        # plan = tsne.fit_transform(features)

        # # saver = tf.train.Saver([plan])
        # embedding_var = K.variable(plan, name='features')
        # embeddings_metadata_path = os.path.join(logdir, 'metadata.tsv')

        # with open(embeddings_metadata_path,'w') as f:
        #     f.write("Index\tPattern\tDomain\tRightWrong\n")
        #     for index,label in enumerate(zip(numpy.argmax(y_test, axis=1), numpy.argmax(ytest[1], axis=1), rightwrong)):
        #         f.write("%d\t%d\t%d\t%d\n" % (index, label[0], label[1], label[2]))

        # # K.set_value(embedding_var, plan)
        # saver = tf.train.Saver([embedding_var])
        # config = projector.ProjectorConfig()
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        # embedding.metadata_path = embeddings_metadata_path
        # writer = tf.summary.FileWriter(logdir)
        # projector.visualize_embeddings(writer, config)
        # saver.save(K.get_session(), logdir + 'embeddings.ckpt', 1)


@inherits(Merge)
class Score(luigi.Task):
    logdir = luigi.Parameter()
    gpu = luigi.Parameter(default='3')
    print_accuracy = luigi.IntParameter(default=1)

    def requires(self):
        t = self.clone(Merge)
        return t

    def output(self):
        return luigi.LocalTarget(LOCAL + self.logdir + self.name)

    def run(self):
        # with self.input() as fin:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        model = load_model(LOCAL + self.logdir + 'model-ckpt')

        dataset = pandas.read_pickle(self.input().path)[
            ['pos', 'labels']].values
        num_classes = numpy.unique(dataset[:, 1]).shape[0]
        eval_generator = _batch_generator_binary_images(dataset,
                                                        50,
                                                        num_classes,
                                                        None,
                                                        False
                                                        )

        evals = model.evaluate_generator(eval_generator,
                                         steps=dataset.shape[0] // 50,
                                         max_queue_size=5)

        print(evals)


@inherits(Train)
class Predict(luigi.Task):

    def requires(self):
        return self.clone(Train)

    def output(self):
        return None

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        model = load_model(LOCAL + self.logdir + 'model-ckpt')

        df = pandas.read_pickle(self.input()[-1].path)
        dataset = df[['pos', 'labels']].values
        num_classes = numpy.unique(dataset[:, 1]).shape[0]
        eval_generator = _batch_generator_binary_images(dataset,
                                                        50,
                                                        num_classes,
                                                        None,
                                                        False
                                                        )

        predictions = model.predict_generator(eval_generator,
                                              steps=dataset.shape[0] // 50,
                                              max_queue_size=5,
                                              verbose=1)

        preds = numpy.argmax(predictions, axis=1)

        df['pred'] = preds

        df.to_pickle(self.input().path)

if __name__ == '__main__':
    luigi.run()
