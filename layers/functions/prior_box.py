from __future__ import division             #python未来支持的语言特征division(精确除法)，当我们没有在程序中导入该特征时，"/"操作符执行的是截断除法(Truncating Division),当我们导入精确除法之后，"/"执行的是精确除法
from math import sqrt as sqrt
from itertools import product as product       #
import torch

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        #number of priors for feature map location(either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratio = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):            #(i,j)由于repeat=2则都是一样的数
                f_k = self.image_size / self.steps[k]
                #unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #aspect_ratio: 1
                #rel size:min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                #aspect_ratio: 1
                #rel_size:sqrt(s_k*s_(k+1))
                s_k_prime = sqrt(s_k*(self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #rest of aspect ratios
                for ar in self.aspect_ratio[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        #back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output