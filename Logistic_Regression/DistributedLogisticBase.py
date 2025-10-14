import numpy as np
import time
from scipy.special import expit as sigmoid
from scipy.sparse import isspmatrix
from parameters import Parameters


class DistributedLogisticBase:
    """
    Base class for distributed logistic regression.
    Provides loss, prediction, accuracy evaluation, and parameter update rules.
    """

    def __init__(self, params: Parameters):
        self.params = params
        self.x_estimate = None
        self.x = None
        # For CHOCO and MoTEF
        self.x_hat = None
        self.H = None
        self.M = None
        self.V = None
        self.G = None

    def lr(self, epoch, iteration, num_samples, d):
        """Learning rate schedule"""
        p = self.params
        t = epoch * num_samples + iteration
        if p.lr_type == 'constant':
            return p.initial_lr
        if p.lr_type == 'decay':
            return p.initial_lr / (p.regularizer * (t + p.tau))

    def loss(self, A, y):
        """Logistic regression loss with optional regularization"""
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = x.copy().mean(axis=1)
        p = self.params
        loss = np.sum(np.log(1 + np.exp(-y * (A @ x)))) / A.shape[0]
        if p.regularizer:
            loss += p.regularizer * np.square(x).sum() / 2
        return loss

    def predict(self, A):
        """Binary prediction"""
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, A):
        """Return probability estimates"""
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = x.mean(axis=1)
        logits = A @ x
        return sigmoid(logits)

    def score(self, A, y):
        """Accuracy score"""
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 2 * (logits >= 0.) - 1
        acc = np.mean(pred == y)
        return acc

    def update_estimate(self, t):
        """Update parameter estimate based on averaging strategy"""
        t = int(t)
        p = self.params
        if p.estimate == 'final':
            self.x_estimate = self.x
        elif p.estimate == 'mean':
            rho = 1 / (t + 1)
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
        elif p.estimate == 't+tau':
            rho = 2 * (t + p.tau) / ((1 + t) * (t + 2 * p.tau))
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
        elif p.estimate == '(t+tau)^2':
            rho = 6 * ((t + p.tau) ** 2) / ((1 + t) * (6 * p.tau ** 2 + t + 6 * p.tau * t + 2 * (t ** 2)))
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho

    # === Placeholder for quantization (to be customized in child classes) ===
    def quantize(self, diff):
        return diff


# === Algorithms ===

class DSGD(DistributedLogisticBase):
    """Decentralized SGD with full communication"""

    def fit(self, A, y):
        p = self.params
        num_samples, num_features = A.shape
        losses = np.zeros(p.num_epoch + 1)

        if self.x is None:
            self.x = np.random.normal(0, 0.01, size=(num_features,))
            self.x = np.tile(self.x, (p.n_cores, 1)).T

        losses[0] = self.loss(A, y)
        start_time = time.time()

        for epoch in range(p.num_epoch):
            for it in range(num_samples // p.n_cores):
                lr = self.lr(epoch, it, num_samples, num_features)
                x_plus = np.zeros_like(self.x)

                for machine in range(p.n_cores):
                    idx = np.random.choice(num_samples, size=100, replace=False)
                    a_batch, y_batch = A[idx], y[idx]
                    x_m = self.x[:, machine]
                    pred = a_batch @ x_m
                    grad = (y_batch[:, None] * a_batch) * sigmoid(-y_batch * pred)[:, None]
                    grad = grad.mean(axis=0)
                    if isspmatrix(a_batch):
                        grad = grad.toarray().squeeze(0)
                    x_plus[:, machine] = lr * grad

                # DSGD update
                self.x = (self.x + x_plus).dot(self.W)

            losses[epoch + 1] = self.loss(A, y)
            print(f"[DSGD] Epoch {epoch}: loss {losses[epoch+1]} acc {self.score(A,y)} elapsed {time.time()-start_time:.2f}s")

        return losses


class CHOCO(DistributedLogisticBase):
    """CHOCO-SGD with compressed communication"""

    def fit(self, A, y):
        p = self.params
        num_samples, num_features = A.shape
        losses = np.zeros(p.num_epoch + 1)

        if self.x is None:
            self.x = np.random.normal(0, 0.01, size=(num_features,))
            self.x = np.tile(self.x, (p.n_cores, 1)).T
            self.x_hat = np.copy(self.x)

        losses[0] = self.loss(A, y)
        start_time = time.time()

        for epoch in range(p.num_epoch):
            for it in range(num_samples // p.n_cores):
                lr = self.lr(epoch, it, num_samples, num_features)
                x_plus = np.zeros_like(self.x)

                for machine in range(p.n_cores):
                    idx = np.random.choice(num_samples, size=100, replace=False)
                    a_batch, y_batch = A[idx], y[idx]
                    x_m = self.x[:, machine]
                    pred = a_batch @ x_m
                    grad = (y_batch[:, None] * a_batch) * sigmoid(-y_batch * pred)[:, None]
                    grad = grad.mean(axis=0)
                    x_plus[:, machine] = lr * grad

                # CHOCO update
                x_plus += self.x
                self.x = x_plus + p.consensus_lr * self.x_hat.dot(self.W - np.eye(p.n_cores))
                quantized = self.quantize(self.x - self.x_hat)
                self.x_hat += quantized

            losses[epoch + 1] = self.loss(A, y)
            print(f"[CHOCO] Epoch {epoch}: loss {losses[epoch+1]} acc {self.score(A,y)} elapsed {time.time()-start_time:.2f}s")

        return losses


class LBGD_Sign(DistributedLogisticBase):
    """Log-Bit Gradient Descent with Sign quantization"""

    def quantize(self, diff):
        return np.sign(diff)

    def fit(self, A, y):
        p = self.params
        num_samples, num_features = A.shape
        losses = np.zeros(p.num_epoch + 1)

        if self.x is None:
            self.x = np.random.normal(0, 0.01, size=(num_features,))
            self.x = np.tile(self.x, (p.n_cores, 1)).T
            self.sigma = np.copy(self.x)
            self.sigma0 = np.copy(self.x)

        losses[0] = self.loss(A, y)
        start_time = time.time()

        for epoch in range(p.num_epoch):
            for it in range(num_samples // p.n_cores):
                t = epoch * (num_samples // p.n_cores) + it
                lr = self.lr(epoch, it, num_samples, num_features)
                x_plus = np.zeros_like(self.x)

                for machine in range(p.n_cores):
                    idx = np.random.choice(num_samples, size=100, replace=False)
                    a_batch, y_batch = A[idx], y[idx]
                    x_m = self.x[:, machine]
                    pred = a_batch @ x_m
                    grad = (y_batch[:, None] * a_batch) * sigmoid(-y_batch * pred)[:, None]
                    grad = grad.mean(axis=0)
                    x_plus[:, machine] = lr * grad

                # LBGD-Sign update
                g_t = 5 * (0.999 ** (t+1))
                self.sigma0 = self.sigma
                quantized = self.quantize((self.x - self.sigma0) / g_t)
                self.sigma = self.sigma0 + 0.01 * g_t * quantized
                self.x = self.x + 0.015 * x_plus + 0.001 * self.sigma0 @ (self.W - np.eye(p.n_cores))
              
            losses[epoch + 1] = self.loss(A, y)
            print(f"[LBGD-Sign] Epoch {epoch}: loss {losses[epoch+1]} acc {self.score(A,y)} elapsed {time.time()-start_time:.2f}s")

        return losses


class LBGD_HarMo(DistributedLogisticBase):
    """Log-Bit Gradient Descent with Harmonic quantization"""

    def quantize(self, diff):
        m1, m2 = self.params.m1, self.params.m2
        K = 2 ** (m1 - 1)
        Delta = 2 ** (-m2)
        half = 0.5 * Delta
        y = np.clip(diff, -K + half, K - half)
        q = np.sign(y) * Delta * (np.floor(np.abs(y) / Delta) + 0.5)
        q = np.clip(q, -K + half, K - half)
        return q

    def fit(self, A, y):
        p = self.params
        num_samples, num_features = A.shape
        losses = np.zeros(p.num_epoch + 1)

        if self.x is None:
            self.x = np.random.normal(0, 0.01, size=(num_features,))
            self.x = np.tile(self.x, (p.n_cores, 1)).T
            self.sigma = np.copy(self.x)
            self.sigma0 = np.copy(self.x)

        losses[0] = self.loss(A, y)
        start_time = time.time()

        for epoch in range(p.num_epoch):
            for it in range(num_samples // p.n_cores):
                t = epoch * (num_samples // p.n_cores) + it
                lr = self.lr(epoch, it, num_samples, num_features)
                x_plus = np.zeros_like(self.x)

                for machine in range(p.n_cores):
                    idx = np.random.choice(num_samples, size=100, replace=False)
                    a_batch, y_batch = A[idx], y[idx]
                    x_m = self.x[:, machine]
                    pred = a_batch @ x_m
                    grad = (y_batch[:, None] * a_batch) * sigmoid(-y_batch * pred)[:, None]
                    grad = grad.mean(axis=0)
                    x_plus[:, machine] = lr * grad

                # LBGD-HarMo update
                g_t = 10 * (0.999999 ** (t+1))
                self.sigma0 = self.sigma
                quantized = self.quantize((self.x - self.sigma0) / g_t)
                self.sigma = self.sigma0 + 0.005 * g_t * quantized
                self.x = self.x + 0.015 * x_plus + 0.001 * self.sigma0 @ (self.W - np.eye(p.n_cores))

            losses[epoch + 1] = self.loss(A, y)
            print(f"[LBGD-HarMo] Epoch {epoch}: loss {losses[epoch+1]} acc {self.score(A,y)} elapsed {time.time()-start_time:.2f}s")

        return losses


class MoTEF(DistributedLogisticBase):
    """MoTEF algorithm with memory and error-feedback"""

    def fit(self, A, y):
        p = self.params
        num_samples, num_features = A.shape
        losses = np.zeros(p.num_epoch + 1)

        if self.x is None:
            self.x = np.random.normal(0, 0.01, size=(num_features,))
            self.x = np.tile(self.x, (p.n_cores, 1)).T
            self.H = np.zeros_like(self.x)
            self.M = np.zeros_like(self.x)
            self.V = np.zeros_like(self.x)
            self.G = np.zeros_like(self.x)

        losses[0] = self.loss(A, y)
        start_time = time.time()

        for epoch in range(p.num_epoch):
            for it in range(num_samples // p.n_cores):
                lr = self.lr(epoch, it, num_samples, num_features)
                x_plus = np.zeros_like(self.x)

                for machine in range(p.n_cores):
                    idx = np.random.choice(num_samples, size=100, replace=False)
                    a_batch, y_batch = A[idx], y[idx]
                    x_m = self.x[:, machine]
                    pred = a_batch @ x_m
                    grad = (y_batch[:, None] * a_batch) * sigmoid(-y_batch * pred)[:, None]
                    grad = grad.mean(axis=0)
                    x_plus[:, machine] = lr * grad

                # MoTEF update
                x_new = self.x + p.gamma * self.H.dot(self.W - np.eye(p.n_cores)) - p.eta * self.V
                Qh = self.quantize(x_new - self.H)
                self.H += Qh
                M_new = (1 - p.lam) * self.M - lr * x_plus
                V_new = self.V + p.gamma * self.G.dot(self.W - np.eye(p.n_cores)) + (M_new - self.M)
                Qg = self.quantize(V_new - self.G)
                self.G += Qg
                self.x = x_new
                self.M = M_new
                self.V = V_new

            losses[epoch + 1] = self.loss(A, y)
            print(f"[MoTEF] Epoch {epoch}: loss {losses[epoch+1]} acc {self.score(A,y)} elapsed {time.time()-start_time:.2f}s")

        return losses
