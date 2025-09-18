# rl_portfolio_cvxpy.py
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

np.random.seed(42)

def cvxpy_projection(v, u, z=1.0):
    """
    Project candidate vector v onto:
        { w : sum(w)=z, 0 <= w_i <= u_i }
    by solving: minimize ||w - v||^2.
    """
    n = len(v)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(w - v))
    constraints = [cp.sum(w) == z, w >= 0, w <= u]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)  # or OSQP/CLARABEL if installed
    return np.array(w.value).flatten()

def simulate_returns(T, n, mu=0.0005, sigma=0.01):
    """
    Simulate daily returns for n assets.
    Replace with real historical returns for production.
    """
    return np.random.normal(loc=mu, scale=sigma, size=(T, n))

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

class LinearPolicy:
    def __init__(self, n_assets, feat_dim):
        self.W = 0.01 * np.random.randn(n_assets, feat_dim)
        self.b = np.zeros(n_assets)

    def logits(self, features):
        return self.W.dot(features) + self.b

    def candidate_weights(self, features):
        return softmax(self.logits(features))

    def get_params_vector(self):
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_params_vector(self, vec):
        W_size = self.W.size
        self.W = vec[:W_size].reshape(self.W.shape)
        self.b = vec[W_size:]

def evaluate_policy(policy, returns, cap_per_asset=0.5, transaction_cost=0.000):
    T, n = returns.shape
    wealth = 1.0
    wealth_hist = []
    prev_w = np.zeros(n)
    cap = np.ones(n) * cap_per_asset

    for t in range(T):
        feat = returns[t-1] if t > 0 else np.zeros(n)
        features = feat[:policy.W.shape[1]]  # trim/fit to policy input
        cand = policy.candidate_weights(features)
        w = cvxpy_projection(cand, cap, z=1.0)  # convex step here

        r = w.dot(returns[t])
        tc = transaction_cost * np.sum(np.abs(w - prev_w))
        net_r = r - tc
        wealth *= (1.0 + net_r)
        wealth_hist.append(wealth)
        prev_w = w.copy()

    cum_return = wealth - 1.0
    return cum_return, np.array(wealth_hist)

def train_es(policy, n_iter=100, pop_size=40, sigma_noise=0.05, lr=0.1,
             episode_len=50, n_assets=5, mu=0.0005, sigma=0.01):
    history = []
    param_vec = policy.get_params_vector().copy()
    dim = len(param_vec)

    for it in range(n_iter):
        noises = np.random.randn(pop_size, dim)
        rewards = np.zeros(pop_size)
        for i in range(pop_size):
            vec_try = param_vec + sigma_noise * noises[i]
            policy_try = LinearPolicy(n_assets, policy.W.shape[1])
            policy_try.set_params_vector(vec_try)
            rets = simulate_returns(episode_len, n_assets, mu=mu, sigma=sigma)
            rewards[i], _ = evaluate_policy(policy_try, rets)

        A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        grad_est = np.dot(A, noises) / (pop_size * sigma_noise)
        param_vec = param_vec + lr * grad_est
        policy.set_params_vector(param_vec)

        rets = simulate_returns(episode_len, n_assets, mu=mu, sigma=sigma)
        eval_score, _ = evaluate_policy(policy, rets)
        history.append(eval_score)

        if (it + 1) % max(1, n_iter // 10) == 0:
            print(f"Iter {it+1}/{n_iter}  eval_score={eval_score:.6f}")

    return history, policy

def one_step_risk_qp(returns_window, cap_per_asset=0.5, lam=1.0):
    """
    Solve: max_w mu^T w - lam * w^T Σ w
           s.t. sum(w)=1, 0 <= w <= cap_per_asset
    """
    mu_hat = returns_window.mean(axis=0)
    Sigma_hat = np.cov(returns_window, rowvar=False) + 1e-6*np.eye(returns_window.shape[1])
    n = len(mu_hat)
    w = cp.Variable(n)
    objective = cp.Maximize(mu_hat @ w - lam * cp.quad_form(w, Sigma_hat))
    constraints = [cp.sum(w) == 1, w >= 0, w <= cap_per_asset]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    return np.array(w.value).flatten()

if __name__ == "__main__":
    n_assets = 5
    feat_dim = n_assets
    policy = LinearPolicy(n_assets=n_assets, feat_dim=feat_dim)

    history, trained_policy = train_es(
        policy,
        n_iter=50,
        pop_size=30,
        sigma_noise=0.08,
        lr=0.12,
        episode_len=80,
        n_assets=n_assets,
        mu=0.0007,
        sigma=0.02
    )

    # Plot training progress
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Eval cumulative return")
    plt.title("ES training with convex projection (cvxpy)")
    plt.grid(True)
    plt.show()

    # Final evaluation
    rets = simulate_returns(80, n_assets, mu=0.0007, sigma=0.02)
    cum, wealth_hist = evaluate_policy(trained_policy, rets)
    print("Final wealth:", wealth_hist[-1])
