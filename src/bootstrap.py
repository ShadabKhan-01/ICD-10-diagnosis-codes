"""Bootstrap confidence interval utilities."""

from typing import Callable, Any, Tuple
import numpy as np


def bootstrap_ci(
    base_preds: Any,
    new_preds: Any,
    gold: Any,
    metric_func: Callable,
    seed: int = 42,
    n_iterations: int = 1000,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap confidence interval for (new - base).

    Returns:
        (ci_lower, ci_upper, mean_delta)
    """
    rng = np.random.RandomState(seed)
    n = len(base_preds)
    deltas = []

    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        b_p = [base_preds[i] for i in idx]
        n_p = [new_preds[i] for i in idx]
        g = [gold[i] for i in idx]

        m_base = metric_func(b_p, g)
        m_new = metric_func(n_p, g)
        deltas.append(m_new - m_base)

    deltas_arr = np.array(deltas)
    mean_delta = float(np.mean(deltas_arr))
    ci_lower = float(np.percentile(deltas_arr, 2.5))
    ci_upper = float(np.percentile(deltas_arr, 97.5))
    return ci_lower, ci_upper, mean_delta
