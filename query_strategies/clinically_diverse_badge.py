import numpy as np
import random
import pdb
from scipy import stats
from sklearn.metrics import pairwise_distances
from .sampler import Sampler


# kmeans ++ initialization from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
def init_centers(X, K, debug=False):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('Starting K-Means++')
    if debug:
        print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if debug:
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


class ClinicallyDiverseBadgeSampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, ideal_path=None):
        '''Constructor implemented in sampler'''
        super(ClinicallyDiverseBadgeSampler, self).__init__(n_pool, start_idxs)

        # load embeddings if ideal sampling
        if ideal_path is not None:
            self.embeddings = np.load(ideal_path)

    def query(self, n, embeddings=None):
        '''Performs badge sampling with k-means++ for increased diversity.
        Parameters:
            :param embeddings: datastructure containing the gradient embeddings of the penultimate layer
            :type embeddings: dict
            :param n: number of samples to be queried
            :type n: int'''

        rand_idxs = random.sample(range(0, len(np.squeeze(embeddings['indices']))), len(np.squeeze(embeddings['indices'])))

        if embeddings is not None:
            # get probabilities and their indices
            indices = embeddings['indices']
            grad_embedding = embeddings['embeddings'][rand_idxs]
            rand_bios = embeddings['IDs'][rand_idxs]
        else:
            indices = np.arange(self.embeddings.shape[0])
            indices = np.delete(indices, self.idx_current)
            grad_embedding = np.delete(self.embeddings, self.idx_current, axis=0)

        bio_idx = dict()
        seen = []  # prevents the same image being selected when considering a diff attribute
        for idx, bio in enumerate(rand_bios):
            for j, b in enumerate(list(bio)):
                if b != 0:
                    if rand_idxs[idx] not in seen:
                        if b not in bio_idx:
                            bio_idx[b] = [rand_idxs[idx]]
                        else:
                            bio_idx[b].append(rand_idxs[idx])
                        seen.append(rand_idxs[idx])
                # elif j == 0 and b == 0:
                #     if rand_idxs[idx] not in seen:
                #         if b not in bio_idx:
                #             bio_idx[b] = [rand_idxs[idx]]
                #         else:
                #             bio_idx[b].append(rand_idxs[idx])
                #         seen.append(rand_idxs[idx])

        # Random list of not necessarily unique bios
        unique_bios = random.choices(list(bio_idx.keys()), k=n)

        inds = []

        for bio in unique_bios:
            if len(bio_idx[bio]) > 0:
                bio_embed_ind = init_centers(grad_embedding[bio_idx[bio]], 1)
                idx = bio_idx[bio][bio_embed_ind[0]]
                inds.append(idx)
                bio_idx[bio].remove(idx)
            else:
                # creates a new dict with only keys from bio_idx that have values
                temp_dict = {k: v for k, v in bio_idx.items() if v}
                all_keys = list(temp_dict.keys())
                new_bio = random.choice(all_keys)
                bio_embed_ind = init_centers(grad_embedding[bio_idx[new_bio]], 1)
                idx = bio_idx[new_bio][bio_embed_ind[0]]
                inds.append(idx)
                bio_idx[new_bio].remove(idx)
        return inds