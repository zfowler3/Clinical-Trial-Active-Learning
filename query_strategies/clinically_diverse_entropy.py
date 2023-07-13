import numpy as np
import random
from .sampler import Sampler

class ClinicallyDiverseEntropySampler(Sampler):
    '''Class for randomly sampling the the most diverse set of patient in each batch. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        '''Constructor implemented in sampler'''
        super(ClinicallyDiverseEntropySampler, self).__init__(n_pool, start_idxs)

    def query(self, n, probs):

        # shuffled the list of bio indicators
        rand_idxs = random.sample(range(0, len(np.squeeze(probs['indices']))), len(np.squeeze(probs['indices'])))

        rand_bios = list(probs['IDs'][rand_idxs])

        # indices for patients with same attribute
        bio_idx = dict()
        seen = []  # prevents the same image being selected when considering a diff attribute
        for idx, bio in enumerate(rand_bios):
            for j, b in enumerate(list(bio)):
                if b != 0:
                    # if rand_idxs[idx] not in seen:
                    if idx not in seen:
                        if b not in bio_idx:
                            bio_idx[b] = [rand_idxs[idx]]
                            # bio_idx[b] = [idx]
                        else:
                            bio_idx[b].append(rand_idxs[idx])
                            # bio_idx[b].append(idx)
                        seen.append(rand_idxs[idx])
                        # seen.append(idx)
                # elif j == 0 and b == 0:
                #     if rand_idxs[idx] not in seen:
                #     # if idx not in seen:
                #         if b not in bio_idx:
                #             bio_idx[b] = [rand_idxs[idx]]
                #             # bio_idx[b] = [idx]
                #         else:
                #             bio_idx[b].append(rand_idxs[idx])
                #             # bio_idx[b].append(idx)
                #         seen.append(rand_idxs[idx])
                        # seen.append(idx)

        # Randomly select one attribute
        # unique_bio = random.choice(list(bio_idx.keys()))
        unique_bios = random.choices(list(bio_idx.keys()), k=n)
        # find n images with that attribute
        # unique_bios = random.choices(bio_idx[unique_bio], k=n)

        probabilities = probs['probs'][rand_idxs]
        # probabilities = probs['probs']
        inds = []

        for bio in unique_bios:
            if len(bio_idx[bio]) > 0:
                logs = np.log2(probabilities[bio_idx[bio]])
                mult = logs*probabilities[bio_idx[bio]]
                entropy = np.sum(mult, axis=1)
                prob_inds = np.argmax(entropy)
                # prob_inds = np.argsort(entropy)[:n]
                index = bio_idx[bio][prob_inds]
                inds.append(index)
                bio_idx[bio].remove(index)
                # inds = np.array(bio_idx[unique_bio])[prob_inds]
            else:
                # creates a new dict without keys having no values
                temp_dict = {k: v for k, v in bio_idx.items() if v}
                all_keys = list(temp_dict.keys())
                new_bio = random.choice(all_keys)
                logs = np.log2(probabilities[bio_idx[new_bio]])
                mult = logs*probabilities[bio_idx[new_bio]]
                entropy = np.sum(mult, axis=1)
                prob_inds = np.argmax(entropy)
                # prob_inds = np.argsort(entropy)[:n]
                index = bio_idx[new_bio][prob_inds]
                inds.append(index)
                bio_idx[new_bio].remove(index)
                # inds = np.array(bio_idx[new_bio])[prob_inds]
        return inds


