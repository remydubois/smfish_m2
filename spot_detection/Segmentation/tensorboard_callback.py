import tensorflow as tf
from keras.callbacks import Callback
from keras import backend as K
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy
import matplotlib
from keras.callbacks import TensorBoard


class MyTensorBoard(Callback):
    """Tensorboard basic visualizations.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    TensorBoard is a visualization tool provided with TensorFlow.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        write_batch_performance: whether to write training metrics on batch
            completion
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 write_batch_performance=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 save_predictions=0,
                 embeddings_metadata=None):
        super(MyTensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.write_batch_performance = write_batch_performance
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.seen = 0
        self.save_predictions = save_predictions

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        tf.summary.histogram('{}_grad'.format(weight.name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, self.seen)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        if self.validation_data and self.save_predictions:
            images = numpy.argmax(self.model.predict(self.validation_data[0]), axis=-1)
            images = K.variable(images, dtype='uint8')
            # print(images.dtype)
            imsum = tf.summary.image('segmented_images', K.variable(self.validation_data[0]), max_outputs=10)
            self.merged = tf.summary.merge([self.merged, imsum])

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()
        self.seen += self.batch_size

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance == True:
            for name, value in logs.items():
                if name in ['batch','size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size

class MyTB2(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](https://www.tensorflow.org/programmers_guide/embedding)
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 extra=None,
                 save_predictions=False,
                 save_reconstructions=False):

        super(MyTB2, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        if K.backend() != 'tensorflow':
            if histogram_freq != 0:
                warnings.warn('You are not using the TensorFlow backend. '
                              'histogram_freq was set to 0')
                histogram_freq = 0
            if write_graph:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_graph was set to False')
                write_graph = False
            if write_images:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_images was set to False')
                write_images = False
            if embeddings_freq != 0:
                warnings.warn('You are not using the TensorFlow backend. '
                              'embeddings_freq was set to 0')
                embeddings_freq = 0

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.embeddings_data = embeddings_data
        self.extra = extra
        self.save_predictions = save_predictions
        self.batches_seen = 0
        self.save_reconstructions = save_reconstructions

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            layers = [l for l in self.model.layers if l.name in ['conv24']]
            
            for layer in layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        # print('{}_grad'.format(mapped_weight_name))
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name),
                                             layer.output)

        if self.save_predictions:
            inputs = self.model.layers[0].input
            images = self.model.output
            truth = self.model.targets[0]

            labels = tf.argmax(images, axis=-1, output_type=tf.int32)
            predictions = tf.one_hot(labels, depth=images.shape[-1], dtype='float32')
            pred_eroded = tf.nn.erosion2d(
                predictions, 
                tf.ones((3, 3, images.shape[-1]), dtype='float32'), 
                [1, 1, 1, 1], 
                [1, 1, 1, 1], 
                "SAME")
            diff_pred = predictions - pred_eroded
            
            mask_pred = tf.where(tf.equal(diff_pred[:, :, :, 1], tf.ones_like(diff_pred[:, :, :, 1], dtype='float32')),
                inputs[:, :, :, 0],
                tf.ones_like(diff_pred[:, :, :, 1], dtype='float32') * tf.reduce_max(inputs))
            # antimask = tf.where(tf.equal(diff[:, :, :, 1], tf.ones_like(diff[:, :, :, 1], dtype='float32')),
            #     inputs[:, :, :, 0],
            #     tf.zeros_like(diff[:, :, :, 1], dtype='float32'))

            # truth = tf.one_hot(targets, depth=images.shape[-1], dtype='float32')
            truth_eroded = tf.nn.erosion2d(
                truth, 
                tf.ones((3, 3, images.shape[-1]), dtype='float32'), 
                [1, 1, 1, 1], 
                [1, 1, 1, 1], 
                "SAME")
            diff_truth = truth - truth_eroded
            
            mask_truth = tf.where(tf.equal(diff_truth[:, :, :, 1], tf.ones_like(diff_truth[:, :, :, 1], dtype='float32')),
                inputs[:, :, :, 0],
                tf.ones_like(diff_truth[:, :, :, 1], dtype='float32') * tf.reduce_max(inputs))
            antimask = tf.where(tf.logical_and(tf.equal(diff_pred[:, :, :, 1], tf.ones_like(diff_pred[:, :, :, 1], dtype='float32')), tf.equal(diff_truth[:, :, :, 1], tf.ones_like(diff_truth[:, :, :, 1], dtype='float32'))),
                inputs[:, :, :, 0],
                tf.zeros_like(diff_truth[:, :, :, 1], dtype='float32'))
            
            indices = tf.to_int32(tf.round(labels * 255 / (tf.shape(images)[-1] - 1)))
            cm = matplotlib.cm.viridis
            colors = tf.constant(cm.colors, dtype=tf.float32)
            colored = tf.gather(colors, indices)
            
            tf.summary.image('predictions', colored, max_outputs=6)
            tf.summary.image('boundaries', tf.stack([antimask, mask_pred, mask_truth], axis=-1), max_outputs=6)
            
        if self.save_reconstructions:
            tf.summary.image('reconstructed', tf.expand_dims(self.model.outputs[0] * 255, axis=-1), max_outputs=5)
            tf.summary.image('inputs', self.model.inputs[0], max_outputs=5)
            tf.summary.image('targets', tf.expand_dims(self.model.targets[0], axis=-1), max_outputs=5)

        self.merged = tf.summary.merge_all()


        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq and self.embeddings_data is not None:
            self.embeddings_data = standardize_input_data(self.embeddings_data, model.input_names)

            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = np.prod(embedding_input.shape[1:])
                    embedding_input = tf.reshape(embedding_input,
                                                 (step, int(embedding_size)))
                    shape = (self.embeddings_data[0].shape[0], int(embedding_size))
                    embedding = tf.Variable(tf.zeros(shape),
                                            name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(embedding[batch_id:batch_id + step],
                                      embedding_input)
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")

        # if self.save_predictions:
        #     images = K.variable(numpy.argmax(self.model.predict(self.validation_data[0]), axis=-1), dtype='uint8')
        #     imsum = tf.summary.image('predictions', images, max_outputs=5)
        #     print('saving...')
        #     self.merged = tf.summary.merge([self.merged, imsum])


        if self.validation_data and self.histogram_freq:
            # print(self.validation_data)
            if epoch % self.histogram_freq == 0:
                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))

                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.validation_data and self.save_predictions:
            # print(self.validation_data)
            val_data = self.validation_data
            tensors = (self.model.inputs +
                       self.model.targets +
                       self.model.sample_weights)

            if self.model.uses_learning_phase:
                tensors += [K.learning_phase()]

            assert len(val_data) == len(tensors)
            val_size = val_data[0].shape[0]
            i = 0
            while i < val_size:
                step = min(self.batch_size, val_size - i)
                if self.model.uses_learning_phase:
                    # do not slice the learning phase
                    batch_val = [x[i:i + step] for x in val_data[:-1]]
                    batch_val.append(val_data[-1])
                else:
                    batch_val = [x[i:i + step] for x in val_data]
                assert len(batch_val) == len(tensors)
                feed_dict = dict(zip(tensors, batch_val))

                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)
                i += self.batch_size

        if self.validation_data and self.save_reconstructions:
            # print(self.validation_data)
            val_data = self.validation_data
            tensors = (self.model.inputs +
                       self.model.targets +
                       self.model.sample_weights)

            if self.model.uses_learning_phase:
                tensors += [K.learning_phase()]

            assert len(val_data) == len(tensors)
            val_size = val_data[0].shape[0]
            i = 0
            while i < val_size:
                step = min(self.batch_size, val_size - i)
                if self.model.uses_learning_phase:
                    # do not slice the learning phase
                    batch_val = [x[i:i + step] for x in val_data[:-1]]
                    batch_val.append(val_data[-1])
                else:
                    batch_val = [x[i:i + step] for x in val_data]
                assert len(batch_val) == len(tensors)
                feed_dict = dict(zip(tensors, batch_val))

                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)
                i += self.batch_size
            # if self.save_predictions:
            #     print('saving....')
            #     val_data = self.validation_data
            #     tensors = [self.model.layers[-1].output]
                
            #     # if self.model.uses_learning_phase:
            #         # tensors += [K.learning_phase()]

            #     val_size = val_data[0].shape[0]
            #     i = 0
            #     while i < val_size:
            #         step = min(self.batch_size, val_size - i)
            #         if self.model.uses_learning_phase:
            #             # do not slice the learning phase
            #             batch_val = [x[i:i + step] for x in val_data[:-1]]
            #             batch_val.append(val_data[-1])
            #         else:
            #             batch_val = [x[i:i + step] for x in val_data]
            #         assert len(batch_val) == len(tensors)
            #         feed_dict = dict(zip(tensors, batch_val))

            #         result = self.sess.run([self.merged], feed_dict=feed_dict)
            #         summary_str = result[0]
            #         self.writer.add_summary(summary_str, epoch)
            #         i += self.batch_size



        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {model_input: embeddings_data[idx][batch]
                                     for idx, model_input in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir, 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()


    # def on_batch_begin(self, batch, logs=None):
    #     if self.batches_seen == 0 and self.save_predictions:
    #         images = K.variable(self.validation_data[0])
    #         imsum = tf.summary.image('recorded_images', images, max_outputs=5)
    #         self.merged = tf.summary.merge([self.merged, imsum])
            # self.merged = tf.summary.merge_all()
            

    def on_batch_end(self, batch, logs=None):
            
            logs = logs or {}

            if self.extra is not None:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = K.get_value(self.extra)
                summary_value.tag = 'extra'
                self.writer.add_summary(summary, self.batches_seen)

            self.writer.flush()
            self.batches_seen += 1


    def on_train_end(self, _):
        self.writer.close()


class VarChanger(Callback):

    def __init__(self, lam, scale, loc=5):
        self.lam = lam
        self.scale = scale
        self.loc = loc # Middle of the increase pent
        self.batch_seen = 0
        self.epoch_seen = 0

    def on_batch_end(self, batch, logs={}):
        # if self.epoch_seen < 1:
        self.batch_seen += 1
        # 1/(1+exp(-5(x-30*80)/(15*80)))
        val = (1 / (1 + numpy.exp(-5 * (self.batch_seen - self.loc) / (self.loc / 3)))) * (self.scale - 1) + numpy.ones_like(self.scale)
        K.set_value(self.lam, val)