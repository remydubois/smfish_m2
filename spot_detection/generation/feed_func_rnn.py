import numpy
import math

def batch_generator(seqs, batch_size):

    steps_per_epoch = math.ceil(len(seqs) / batch_size)
    """
    Yields padded sequences.
    """
    i = 0

    # Loops indefinitely, as precised in https://keras.io/models/sequential/
    while True:
        sl = slice(i * batch_size, (i + 1) * batch_size)

        lines = seqs[sl]

        X = [seq[::math.ceil(seq.shape[0]/200)] for seq in lines]
        # max_sent_length = numpy.max([m.shape[0] for m in mats])
        y = [seq[::math.ceil(seq.shape[0])] for seq in seqs[sl]]

        batch = numpy.stack([resize(x, (512, 512)) for x in X])[:, :, :, numpy.newaxis]
        y = numpy.stack([resize((m > 0).astype(int), (512, 512)) for m in y])[:, :, :, numpy.newaxis]

        # Avoid storing too large numbers by modulo.
        i = (i + 1) % steps_per_epoch

        yield (batch, y)