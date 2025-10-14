def train_distributed(self, A, y_init):
    """
    Distributed training loop supporting DSGD, CHOCO, LBGD_HarMo, LBGD_Sign, and MoTEF.
    """
    y = np.copy(y_init)
    num_samples, num_features = A.shape
    p = self.params

    losses = np.zeros(p.num_epoch + 1)

    # Initialize model parameters if not already set
    if self.x is None:
        self.x = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))
        self.x = np.tile(self.x, (p.n_cores, 1)).T
        self.x_estimate = np.copy(self.x)
        self.x_hat = np.copy(self.x)
        self.sigma = np.copy(self.x)
        self.q = np.copy(self.x)
        self.sigma0 = np.copy(self.x)
        self.H = np.zeros_like(self.x)
        self.M = np.zeros_like(self.x)
        self.V = np.zeros_like(self.x)
        self.G = np.zeros_like(self.x)

    # Partition data across machines
    if p.distribute_data:
        np.random.seed(p.split_data_random_seed)
        num_samples_per_machine = num_samples // p.n_cores
        if p.split_data_strategy == 'random':
            all_indexes = np.arange(num_samples)
            np.random.shuffle(all_indexes)
        elif p.split_data_strategy == 'naive':
            all_indexes = np.arange(num_samples)
        elif p.split_data_strategy == 'label-sorted':
            all_indexes = np.argsort(y)

        indices = []
        for machine in range(0, p.n_cores - 1):
            indices.append(all_indexes[num_samples_per_machine * machine:
                                       num_samples_per_machine * (machine + 1)])
        indices.append(all_indexes[num_samples_per_machine * (p.n_cores - 1):])
    else:
        num_samples_per_machine = num_samples
        indices = np.tile(np.arange(num_samples), (p.n_cores, 1))

    # Binary label conversion if needed
    if len(np.unique(y)) > 2:
        y[y < 5] = -1
        y[y >= 5] = 1

    losses[0] = self.loss(A, y)

    compute_loss_every = int(num_samples_per_machine / LOSS_PER_EPOCH)
    all_losses = np.zeros(int(num_samples_per_machine * p.num_epoch / compute_loss_every) + 1)

    train_start = time.time()
    np.random.seed(p.random_seed)

    for epoch in np.arange(p.num_epoch):
        for iteration in range(num_samples_per_machine):
            t = epoch * num_samples_per_machine + iteration

            # Monitor training loss
            if t % 1 == 0:
                loss = self.loss(A, y)
                print('{} t {} epoch {} iter {} loss {} elapsed {}s'.format(
                    p, t, epoch, iteration, loss, time.time() - train_start))
                all_losses[t // compute_loss_every] = loss
                if np.isinf(loss) or np.isnan(loss):
                    print("Training stopped due to NaN/Inf loss")
                    break

            lr = self.lr(epoch, iteration, num_samples_per_machine, num_features)

            # Local gradient computation
            x_plus = np.zeros_like(self.x)
            for machine in range(p.n_cores):
                batch_idx = np.random.choice(indices[machine], size=30000, replace=False)
                a_batch = A[batch_idx]
                y_batch = y[batch_idx]
                x = self.x[:, machine]

                pred = a_batch @ x
                minus_grad = (y_batch[:, None] * a_batch) * sigmoid(-y_batch * pred)[:, None]
                minus_grad = minus_grad.mean(axis=0)
                if isspmatrix(a_batch):
                    minus_grad = minus_grad.toarray().squeeze(0)
                if p.regularizer:
                    minus_grad -= p.regularizer * x
                x_plus[:, machine] = lr * minus_grad

            # Communication step
            if p.method == "DSGD":
                # Decentralized SGD
                self.x = (self.x + x_plus).dot(self.W)

            elif p.method == "CHOCO":
                # CHOCO-SGD with compression
                x_plus += self.x
                self.x = x_plus + p.consensus_lr * self.x_hat.dot(self.W - np.eye(p.n_cores))
                quantized = self.__quantize(self.x - self.x_hat)
                self.x_hat += quantized

            elif p.method == "LBGD_HarMo":
                # LBGD with Harmonic quantization
                g_t = 10 * (0.999999 ** (t + 1))
                self.sigma0 = self.sigma
                quantized = psi(t + 1, num_features)[..., None] @ (
                    self.__quantize((psi(t + 1, num_features) @ (self.x - self.sigma0) / g_t)[None, ...])
                )
                self.sigma = self.sigma0 + 0.005 * g_t * quantized
                self.x = self.x + 0.015 * x_plus + 0.001 * self.sigma0 @ (self.W - np.eye(p.n_cores))

            elif p.method == "LBGD_Sign":
                # LBGD with Sign quantization
                g_t = 5 * (0.999 ** (t + 1))
                self.sigma0 = self.sigma
                quantized = self.__quantize((self.x - self.sigma0) / g_t)
                self.sigma = self.sigma0 + 0.005 * g_t * quantized
                self.x = self.x + 0.015 * x_plus + 0.001 * self.sigma0 @ (self.W - np.eye(p.n_cores))

            elif p.method == "MoTEF":
                # Momentum-based tracking with error feedback
                x_new = self.x + p.gamma * self.H.dot(self.W - np.eye(p.n_cores)) - p.eta * self.V
                Qh = self.__quantize(x_new - self.H)
                self.H += Qh
                M_new = (1 - p.lam) * self.M
                lr_eff = lr if lr != 0 else 0.1
                M_new += (-lr_eff) * x_plus
                V_new = self.V + p.gamma * self.G.dot(self.W - np.eye(p.n_cores)) + (M_new - self.M)
                Qg = self.__quantize(V_new - self.G)
                self.G += Qg
                self.x = x_new
                self.M = M_new
                self.V = V_new

            self.update_estimate(t)

        losses[epoch + 1] = self.loss(A, y)
        print("epoch {}: loss {} score {}".format(epoch, losses[epoch + 1], self.score(A, y)))
        if np.isinf(losses[epoch + 1]) or np.isnan(losses[epoch + 1]):
            print("Training stopped due to NaN/Inf loss")
            break

    print("Training took: {}s".format(time.time() - train_start))
    return losses, all_losses
