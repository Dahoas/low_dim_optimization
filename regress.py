import numpy as np
import matplotlib.pyplot as plt

from utils import get_linear_fit, linear_func

"""
xs intrinsic dimensions
ys are linear fit of the log-log test error during training

1 / ys appears to be roughly linear but with two separate slopes: the slope magnitude increases as d increases
- might only make sense to study for d << D

Looks like a ~ -2/d
"""
data = [(2, -0.41), (3, -0.18), (4, -0.14), (6, -0.1), (8, -0.07), (12, -0.04), (16, -0.03), (24, -0.02), (32, -0.01)]
xs = np.array([s[0] for s in data])
ys = np.array([s[1] for s in data])

drop_terms = 1
xs = xs[:-drop_terms]
ys = ys[:-drop_terms]

ys = 1 / ys

a, b = get_linear_fit(xs, ys)
y_preds = linear_func(xs, a, b)

plt.clf()
plt.plot(xs, ys, "-o")
plt.plot(xs, y_preds, "--", label=f"Linear fit: a={a:.4f}")

plt.title("d (intrinsic dim) vs. log_log validation decay rate during training")
plt.legend()
plt.savefig("figs/test.png")