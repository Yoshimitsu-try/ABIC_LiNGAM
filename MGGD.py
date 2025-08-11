# SPDX-License-Identifier: MIT

import numpy as np
from scipy.stats import gamma
from scipy.special import digamma
from scipy.optimize import root_scalar

class MGGD:
    def __init__(self, mu, Sigma, beta=1, tol=1e-6):
        """
        MGGD (Multivariate Generalized Gaussian Distribution) クラス

        Parameters:
        mu : array-like
            平均ベクトル
        Sigma : array-like
            対称正定値行列 (分散共分散行列)
        beta : float
            分布の形状パラメータ
        tol : float
            分散共分散行列の正定値性の数値的許容誤差
        """
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        self.beta = beta
        self.tol = tol
        self._validate_parameters()

    def _validate_parameters(self):
        p = len(self.mu)
        if self.Sigma.shape[0] != p or self.Sigma.shape[1] != p:
            raise ValueError("Sigma は mu の長さに一致する正方行列でなければなりません。")
        if not np.allclose(self.Sigma, self.Sigma.T):
            raise ValueError("Sigma は対称行列でなければなりません。")
        if np.any(np.linalg.eigvals(self.Sigma) < self.tol * np.max(np.abs(np.linalg.eigvals(self.Sigma)))):
            raise ValueError("Sigma は対称正定値行列でなければなりません。")
        if self.beta <= 0:
            raise ValueError("beta は正の値でなければなりません。")

    def sample(self, n=1):
        """
        サンプルを生成する

        Parameters:
        n : int
            観測数

        Returns:
        np.ndarray
            p 列、n 行の行列
        """
        p = len(self.mu)
        eigvals, eigvecs = np.linalg.eigh(self.Sigma)
        A = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        r = gamma.rvs(a=p/(2*self.beta), scale=2, size=n)**(1/(2*self.beta))
        U = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)
        normU = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / normU

        result = (r[:, np.newaxis] * (U @ A.T)) + self.mu
        return result

class EstMGGD:
    def __init__(self, data, eps=1e-6, display=False, plot=False):
        self.data = data
        self.eps = eps
        self.display = display
        self.plot = plot
        self.mu = None
        self.Sigma = None
        self.beta = None

    def estimate(self):
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        p = self.data.shape[1]
        n = self.data.shape[0]
        self.mu = np.mean(self.data, axis=0)
        self.data = self.data - self.mu

        self.Sigma = np.eye(p)
        self.beta = 0.1
        beta_1 = np.inf
        betaseq = [] if self.plot else None

        k = 0
        while np.abs(self.beta - beta_1) > self.eps and k < 1000:
            k += 1
            sigmak = np.zeros((p, p))
            invSigma = np.dot(self.data, np.linalg.inv(self.Sigma))
            u = np.sum(invSigma * self.data, axis=1)

            for i in range(n):
                sigmak += u[i]**(self.beta-1) * np.outer(self.data[i], self.data[i])

            self.Sigma = sigmak / n
            self.Sigma = (p / np.sum(np.diag(self.Sigma))) * self.Sigma

            beta_1 = self.beta
            self.beta = self._estimate_beta(u, self.beta, p, self.eps)

            if self.plot:
                betaseq.append(self.beta)

            if self.display:
                print(self.beta, end="  ")

        if self.plot:
            import matplotlib.pyplot as plt
            plt.plot(betaseq)
            plt.show()

        if self.display:
            print()

        m = (self.beta / (p * n) * np.sum(u**self.beta))**(1/self.beta)
        self.Sigma *= m

        result = {'mu': self.mu, 'Sigma': self.Sigma, 'beta': self.beta, 'epsilon': self.eps, 'k': k}
        return result

    def _estimate_beta(self, u, beta0, p, eps=np.finfo(float).eps):
        N = len(u)

        def equation(z):
            term1 = p * N / (2 * np.sum(u**z)) * np.sum(np.log(u + eps) * u**z)
            term2 = p * N / (2 * z) * (np.log(2) + digamma(p / (2 * z)))
            term3 = N
            term4 = p * N / (2 * z) * np.log(z / (p * N) * np.sum(u**z + eps))
            return term1 - term2 - term3 - term4

        bracket = [eps, 2 * np.ceil(beta0)]
        f_a, f_b = equation(bracket[0]), equation(bracket[1])

        while f_a * f_b > 0:
            if f_a > 0:
                bracket[0] /= 2
            else:
                bracket[1] *= 2
            f_a, f_b = equation(bracket[0]), equation(bracket[1])

        result = root_scalar(equation, bracket=bracket, method='brentq')
        return result.root

# 使用例
# 平均ベクトルと分散共分散行列の設定
# mu = [0, 0, 0]
# Sigma = np.eye(3)
# beta = 1

# # サンプル生成
# mggd = MGGD(mu, Sigma, beta)
# data = mggd.sample(n=100)

# # パラメータ推定
# estimator = EstMGGD(data)
# result = estimator.estimate()
# print(result)
