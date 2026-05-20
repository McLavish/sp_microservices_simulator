import argparse
from dataclasses import dataclass

import numpy as np

from rl.models.base import ModelReport
from rl.models.evt_tail import EVTTailModel
from rl.models.lognormal import LogNormalModel
from rl.models.lognormal_mixture import LogNormalMixtureModel


@dataclass
class DatasetSpec:
    name: str
    params: dict[str, float]


def make_lognormal(rng: np.random.Generator, n: int, mu: float, sigma: float) -> np.ndarray:
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def make_mixture(
    rng: np.random.Generator,
    n: int,
    w: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> np.ndarray:
    comp = rng.random(n) < w
    x = np.empty(n, dtype=np.float64)
    x[comp] = rng.lognormal(mean=mu1, sigma=sigma1, size=int(np.sum(comp)))
    x[~comp] = rng.lognormal(mean=mu2, sigma=sigma2, size=int(np.sum(~comp)))
    return x


def make_pareto(rng: np.random.Generator, n: int, alpha: float, x_min: float) -> np.ndarray:
    u = rng.random(n)
    return x_min / (1.0 - u) ** (1.0 / alpha)


def train_test_split(
    data: np.ndarray, seed: int, frac: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    split = int(len(data) * frac)
    train = data[idx[:split]]
    test = data[idx[split:]]
    return train, test


def evaluate_model(name: str, model, train: np.ndarray, test: np.ndarray) -> ModelReport:
    model.fit(train)
    nll = -model.log_likelihood(test) / len(test)
    p95 = np.quantile(test, 0.95)
    p99 = np.quantile(test, 0.99)
    p95_pred = model.quantile(0.95)
    p99_pred = model.quantile(0.99)
    return ModelReport(
        name=name,
        params=model.params(),
        nll=float(nll),
        p95_error=float(abs(p95_pred - p95)),
        p99_error=float(abs(p99_pred - p99)),
    )


def run_experiment(dataset: np.ndarray, seed: int) -> list[ModelReport]:
    train, test = train_test_split(dataset, seed=seed)
    models = [
        ("lognormal", LogNormalModel()),
        ("lognormal_mixture", LogNormalMixtureModel()),
        ("evt_tail", EVTTailModel()),
    ]
    return [evaluate_model(name, model, train, test) for name, model in models]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare latency models on synthetic data.")
    parser.add_argument("--dist", choices=["lognormal", "mixture", "pareto"], default="lognormal")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--w", type=float, default=0.7)
    parser.add_argument("--mu2", type=float, default=4.0)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--x-min", type=float, default=10.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.dist == "lognormal":
        dataset = make_lognormal(rng, args.n, args.mu, args.sigma)
    elif args.dist == "mixture":
        dataset = make_mixture(rng, args.n, args.w, args.mu, args.sigma, args.mu2, args.sigma2)
    else:
        dataset = make_pareto(rng, args.n, args.alpha, args.x_min)

    reports = run_experiment(dataset, seed=args.seed)
    print(f"dataset={args.dist} n={args.n}")
    for report in reports:
        print(
            f"{report.name:18s} nll={report.nll:.4f} "
            f"p95_err={report.p95_error:.3f} p99_err={report.p99_error:.3f}"
        )


if __name__ == "__main__":
    main()
