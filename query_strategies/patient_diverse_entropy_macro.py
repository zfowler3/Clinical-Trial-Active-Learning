import numpy as np
from .sampler import Sampler


class PatientDiverseEntropyMacroSampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        '''Constructor implemented in sampler'''
        super(PatientDiverseEntropyMacroSampler, self).__init__(n_pool, start_idxs)

    def query(self, n, probs):
        '''Returns samples with highest entropy in the output distribution.
        Parameters:=
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''

        # Get probabilities and their indices
        # indices = np.squeeze(probs['indices'])
        probabilities = probs['probs']
        patients = probs['IDs']

        # get max entropy
        logs = np.log2(probabilities)
        mult = logs*probabilities
        entropy = np.sum(mult, axis=1)
        prob_inds = np.argsort(entropy)
        # entropy_indices_sorted = indices[prob_inds]

        # chose n unique entropy indices corresponding do patients in descending entropy order
        id_idx = dict()
        for idx in prob_inds:
            if patients[idx] not in id_idx:
                id_idx[patients[idx]] = idx
            if len(id_idx.keys()) == n:
                break
        return list(id_idx.values())


