import random
from .sampler import Sampler

class ClinicallyDiverseSampler(Sampler):
    '''Class for randomly sampling the the most diverse set of patient in each batch. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        '''Constructor implemented in sampler'''
        super(ClinicallyDiverseSampler, self).__init__(n_pool, start_idxs)

    def query(self, n, clincal_dict):

        # shuffles the list of bio indicators to remove bias towards the 1st entries in list
        rand_idxs = random.sample(range(0, len(clincal_dict['bio'])), len(clincal_dict['bio']))

        rand_bios = clincal_dict['bio'][rand_idxs]

        # indices for patients with same attribute
        bio_idx = dict()
        seen = []  # prevents the same image being selected when considering a diff attribute
        for idx, bio in enumerate(rand_bios):
            for b in bio:
                if rand_idxs[idx] not in seen:
                    if b not in bio_idx:
                        bio_idx[b] = [rand_idxs[idx]]
                    else:
                        bio_idx[b].append(rand_idxs[idx])
                    seen.append(rand_idxs[idx])

        # Random list of not necessarily unique bios
        unique_bios = random.choices(list(bio_idx.keys()), k=n)

        inds = []

        for bio in unique_bios:
            if len(bio_idx[bio]) > 0:
                rand_idx = random.choice(bio_idx[bio])
                inds.append(rand_idx)
                bio_idx[bio].remove(rand_idx)
            else:
                # creates a new dict without keys having no values
                temp_dict = {k: v for k, v in bio_idx.items() if v}
                all_keys = list(temp_dict.keys())
                new_bio = random.choice(all_keys)
                rand_idx = random.choice(bio_idx[new_bio])
                inds.append(rand_idx)
                bio_idx[new_bio].remove(rand_idx)
        return inds
