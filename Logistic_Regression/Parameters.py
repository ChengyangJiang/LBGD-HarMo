class Parameters:
    """
    Parameters class used for all the experiments, redefine a string representation to summarize the experiment
    """

    def __init__(self,
                 num_epoch,
                 lr_type,
                 initial_lr=None,
                 regularizer=None,
                 epoch_decay_lr=None,
                 consensus_lr=None,
                 quantization="full",
                 # proportion α in Top-α quantization
                 coordinates_to_keep=None,
                 estimate='final',
                 name=None,
                 # number of machines
                 n_cores=1,
                 Topology='fully',
                 method='DSGD',
                 distribute_data=False,
                 split_data_strategy=None,
                 tau=None,
                 real_update_every=1,
                 random_seed=None,
                 split_data_random_seed=None,
                 # LBGD parameters
                 m1=None, m2=None,
                 ):
        # === sanity checks ===
        assert num_epoch >= 0
        assert lr_type in ['constant', 'epoch-decay', 'decay', 'bottou']

        if lr_type in ['constant', 'decay']:
            assert initial_lr > 0
        if lr_type == 'decay':
            assert initial_lr and tau
            assert regularizer > 0
        if lr_type == 'epoch-decay':
            assert epoch_decay_lr is not None
        if method in ['CHOCO']:
            assert consensus_lr > 0
        else:
            assert consensus_lr is None

        assert quantization in ['full', 'Top', 'LBGD', 'Sign']
        if quantization == 'full':
            assert not coordinates_to_keep
        elif quantization == 'Top':
            assert coordinates_to_keep > 0
        elif quantization == 'LBGD':
            assert (m1 is not None) and (m2 is not None) and (m1 > 0) and (m2 >= 0)
        elif quantization == 'Sign':
            assert not coordinates_to_keep
            assert m1 is None and m2 is None

        assert estimate in ['final', 'mean', 't+tau', '(t+tau)^2']
        assert n_cores > 0
        assert Topology in ['fully', 'ring', 'torus', 'er']
        assert method in ['CHOCO', 'DSGD', 'LBGD', 'MoTEF']

        if not distribute_data:
            assert not split_data_strategy
        else:
            assert split_data_strategy in ['naive', 'random', 'label-sorted']

        self.num_epoch = num_epoch
        self.lr_type = lr_type
        self.initial_lr = initial_lr
        self.regularizer = regularizer
        self.epoch_decay_lr = epoch_decay_lr
        self.consensus_lr = consensus_lr
        self.quantization = quantization
        self.coordinates_to_keep = coordinates_to_keep
        self.estimate = estimate
        self.name = name
        self.n_cores = n_cores
        self.Topology = Topology
        self.tau = tau
        self.real_update_every = real_update_every
        self.random_seed = random_seed
        self.method = method
        self.distribute_data = distribute_data
        self.split_data_strategy = split_data_strategy
        self.split_data_random_seed = split_data_random_seed
        self.m1 = m1
        self.m2 = m2

    def __str__(self):
        if self.name:
            return self.name

        lr_str = self.lr_str()
        sparse_str = self.sparse_str()
        reg_str = ""
        if self.regularizer:
            reg_str = f"-reg{self.regularizer}"

        return f"epoch{self.num_epoch}-{lr_str}{reg_str}-{sparse_str}-{self.estimate}"

    def lr_str(self):
        if self.lr_type == 'constant':
            return f"lr{self.initial_lr}"
        elif self.lr_type == 'decay':
            return f"lr{self.initial_lr}decay{self.epoch_decay_lr}"
        else:
            return f"lr-{self.lr_type}"
    
    def sparse_str(self):
        sparse_str = self.quantization
        if self.quantization == 'Top':
            sparse_str += f"{self.coordinates_to_keep}"
        elif self.quantization == 'LBGD':
            sparse_str += f"(m1={self.m1},m2={self.m2})"
        elif self.quantization == 'Sign':
            sparse_str += ""
        return sparse_str

    def __repr__(self):
        return f"Parameter('{str(self)}')"
