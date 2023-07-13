import numpy as np
from .sampler import Sampler


class RandomSampling(Sampler):
    '''Class for random sampling algorithm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, total):
        super(RandomSampling, self).__init__(n_pool, start_idxs, total)

    def query(self, n, visit, opt=False):
        '''Performs random query of points'''
        previous_idxs = self.idx_current
        print('length of idx current in rand: ', len(previous_idxs))
        if visit.all() != None:
            # previous weeks samples (unused)
            if not opt:
                xx = np.isin(self.total_given, previous_idxs, invert=True)
                print('size of total given: ', self.total_given.shape)
                print('shape prev idxs: ', previous_idxs.shape)
                prev_unused_idxs = np.where(xx)[0]
                #print('xyxyxyxyx')
                # For dynamic test set size where full visit is added at each round
                if len(prev_unused_idxs) == 0 or n == 0:
                    print('random sampling dynam test size! ')
                    inds = visit
                    return inds[np.random.permutation(len(inds))]

                prev_unused = self.total_given[prev_unused_idxs]
                print('prev unused shape: ', prev_unused.shape) #3596
                # new allowed samples
                inds = visit # new visit's indexes
                print('new visit idxs: ', inds.shape)
                inds = np.concatenate((prev_unused, inds))
                self.total_given = np.concatenate((self.total_given, visit))
                print('UNUSED IDXS UP FOR SELECTING (past + new): ', inds.shape)
            else:
                # only allow current visit
                print('random is working correctly')
                inds = visit
        else:
            print('Retrospective random sampling')
            inds = np.where(self.total_pool == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]