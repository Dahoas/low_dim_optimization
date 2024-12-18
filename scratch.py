import numpy as np
import matplotlib.pyplot as plt


def pow(n, alpha):
    return 1/(n**(alpha))

def inv(d, a, y):
    return -1 / (a*d+2) + y

def exp(d, a, y):
    return -np.exp(-a*d) + y

def log(d, a, y):
    return np.log10(a*d) + y

ds_log = np.log10([2, 3, 4, 6, 8, 12, 16, 24, 32])
y_gt_log = [-0.40737714525030283, -0.18297822581775378, -0.1420332129347276, -0.09820395210028736, -0.06627103928107664, -0.03556623497731707, -0.026262270044368, -0.015535631606591343, -0.008163427240156727]


ds_log = ds_log[1:]
y_gt_log = y_gt_log[1:]
y_pred_log = inv(ds_log, 2, .2)
exp_pred_log = exp(ds_log, .25, .75)
log_pred_log = log(ds_log, .05, 1.5)


plt.plot(ds_log, y_gt_log, "-o", label="gt")
plt.plot(ds_log, y_pred_log, "-o", label="pred")
plt.plot(ds_log, exp_pred_log, "-o", label="exp_pred")
plt.plot(ds_log, log_pred_log, "-o", label="log_pred")
plt.legend()
plt.savefig("figs/test.png")