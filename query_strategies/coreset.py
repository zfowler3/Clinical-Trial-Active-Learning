import numpy as np
from sklearn.metrics import pairwise_distances
from .sampler import Sampler
from .config import BaseConfig


# implementation inspired from https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
# as well as https://github.com/svdesai/coreset-al/blob/master/coreset.py
def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! help')
    else:
        dist_ctr = pairwise_distances(X, X_set)
        print('len of dist ctr: ', dist_ctr.shape)
        print(len(np.unique(dist_ctr)))
        min_dist = np.amin(dist_ctr, axis=1)
        print('min dist shape: ', min_dist.shape)
    print('coreset idxssss unique: ', len(np.unique(min_dist)))
    idxs = []

    for i in range(n):
        idx = min_dist.argmax()
        print('idx: ', idx)
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
    ii = np.array(idxs)
    print('coreset idxs unique: ', len(np.unique(ii)))
    return idxs


class CoresetSampler(Sampler):
    def __init__(self, n_pool, start_idxs, total):
        super(CoresetSampler, self).__init__(n_pool, start_idxs, total)
        self.min_distances = None
        self.total_features = None
        self.already_selected = []

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]
            #print('centers: ', centers)

        if centers is not None:
            x = self.total_features[centers]  # pick only centers
            dist = pairwise_distances(self.total_features, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
        return

    def query(self, n: int, trainer):
        # initially updating the distances
        # idx_current is an array of all INDICES of currently selected images
        embeddings_un, unlabeled_indices = trainer.embeddings(mode='tr', loader_type='unlabeled')
        embeddings_lab, labeled_indices = trainer.embeddings(mode='tr', loader_type='labeled')
        #self.total_features = embeddings_lab + embeddings_un
        self.total_features = np.concatenate((embeddings_lab, embeddings_un))
        total_indices = np.concatenate((labeled_indices, unlabeled_indices))

        labeled_inds_reset = np.arange(0, len(embeddings_lab[0]))
        self.already_selected = labeled_inds_reset
        print('wtf?: ', labeled_inds_reset.shape)
        print('tttttttttttotal features shape: ', self.total_features.shape)
        # reshape
        # feature_len = self.total_features[0].shape[1]
        # self.total_features = self.total_features.reshape(-1,feature_len)
        #print('total features shape vol 2: ', self.total_features.shape)

        self.update_dist(labeled_inds_reset, only_new=False, reset_dist=True)

        #print('SOMEBODY HELP ME: ', len(np.unique(unlabeled_indices)))
        print('already selected shape: ', self.already_selected.shape)
        print('x: ', self.already_selected)
        print('min dists: ', self.min_distances)
        print('hh: ', self.min_distances.shape)
        new_batch = []
        for _ in range(n):
            ind = np.argmax(self.min_distances)
            # if not self.already_selected:
            #     ind = np.random.choice(np.arange(self.dset_size))
            # else:
            #     ind = np.argmax(self.min_distances)
            print('IND: ', ind)
            assert ind not in self.already_selected
            self.update_dist([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('max of inds: ', max(new_batch))
        new_batch = np.array(new_batch)
        actual_idxs = total_indices[new_batch]
        print('actual idxs shape for querying: ', np.array(actual_idxs).astype(int))
        return np.array(actual_idxs).astype(int)

    # def query(self, n: int, trainer):
    #
    #     embeddings_un = trainer.get_embeddings(mode='tr', loader_type='unlabeled')
    #     embeddings_lab = trainer.get_embeddings(mode='tr', loader_type='labeled')
    #
    #     unlabeled_indices = embeddings_un['indices']
    #     print('SOMEBODY HELP ME: ', len(np.unique(unlabeled_indices)))
    #
    #     # do coreset algorithm
    #     chosen = furthest_first(embeddings_un['nongrad_embeddings'], embeddings_lab['nongrad_embeddings'], n)
    #
    #     # derive final indices
    #     inds = unlabeled_indices[chosen]
    #     print('INDS SHAPE: ', inds.shape)
    #
    #     return inds.squeeze()

    def query_te(self, n, trainer):
        embeddings_un = trainer.get_embeddings(mode='te', loader_type='unlabeled')
        embeddings_lab = trainer.get_embeddings(mode='te', loader_type='labeled')

        unlabeled_indices = embeddings_un['indices']

        # do coreset algorithm
        chosen = furthest_first(embeddings_un['nongrad_embeddings'], embeddings_lab['nongrad_embeddings'], n)

        # derive final indices
        inds = unlabeled_indices[chosen]
        print('INDS SHAPE: ', inds.shape)

        return inds