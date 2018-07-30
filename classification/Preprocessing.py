import luigi
import pandas
import json
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm
import warnings
import re
import numpy
import pickle
import sys
from luigi.util import inherits
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed

if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    DATA_PATH = '/Users/remydubois/Dropbox/Remy/Data/180601/'
else:
    DATA_PATH = '/mnt/data40T_v2/rdubois/data/'


class StackSimulations(luigi.ExternalTask):
    simulation_folders = luigi.Parameter(default='smFishSimulations/v2_130k/,smFishSimulations/v3_mRNA25/,smFishSimulations/v4/,smFishSimulations/v5_random/,smFishSimulations/v6_mRNA50/')

    def requires(self):
        return []

    def output(self):
        
        return [luigi.LocalTarget(DATA_PATH + sf + "stacked_file.json") for sf in self.simulation_folders.split(',')]

    def run(self):
        for o in self.output():
            # Retrieve the simulation batches
            simulation_batches_path = [DATA_PATH + self.simulation_folder + 'batches/' + f for f in os.listdir(DATA_PATH + self.simulation_folder + 'batches/') if
                                       f.lower().endswith('.json.gz')]
            
            # Combine all the simulations
            simulations = []
            for b in simulation_batches_path:
                df = pandas.read_json(b)
                simulations.append(df)
            dataset = pandas.concat(simulations)
            # dataset = dataset[['cell_ID', 'RNA_pos', 'pattern_name']]

            # Now shuffle
            dataset = dataset.sample(frac=1).reset_index(drop=True)

            dataset.to_json(o.path, orient='records', lines=True)


def _stack_indices(r):
    prune_3d = lambda m: m[[0, 1, 3]] if m.shape[0] > 3 else m
    try:
        l = [prune_3d(numpy.hstack([e, [[i], ] * len(e)]).T).astype(numpy.int16) for i, e in enumerate(r)]
        return numpy.hstack(l)
    except ValueError:
        warnings.warn('Unconsistent simulation found, ignoring.')
        return numpy.nan

    


@inherits(StackSimulations)
class Merge(luigi.Task):

    def requires(self):
        return self.clone(StackSimulations)

    def output(self):
        return [luigi.LocalTarget(i.path.replace("stacked_file.json", "merged_file.pkl")) for i in self.input()]

    def run(self):
        for i, o in zip(self.input(), self.output()):
            # Read stacked simulations
            # with open(self.output().path, 'w') as fout:
            simulations = pandas.read_json(i.path, orient='records', lines=True)
            # simulations = simulations[['cell_ID', 'RNA_pos', 'pattern_name']]

            # Read cell Library
            jsoncells = pandas.read_json(DATA_PATH + 'cellLibrary.json')

            # Merge (join) the two databases
            jsoncells['cell_ID'] = jsoncells.index + 1
            merged = simulations.merge(jsoncells, on='cell_ID')

            # This operation is most likely vectorizable, using some sort of index unraveling.
            l = Parallel(n_jobs=56, verbose=11)(delayed(_stack_indices)(r) for r in list(zip(*merged[['pos_nuc', 'pos_cell', 'RNA_pos']].iterrows()))[1])
            merged['pos'] = l
            merged.dropna(inplace=True)

            # Prepare labels. Huge mistake, labels are not attributed consistently between the different simulations
            # le = LabelEncoder()
            # merged['labels'] = le.transform(merged['pattern_name'])
            # merged = merged[['pos', 'labels']].values

            # Output
            merged.to_pickle(o.path) #, orient='records', lines=True)
