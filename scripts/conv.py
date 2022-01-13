import numpy as np
import matplotlib.pyplot as plt

def detach(x):
    return x

# ----------------------------------------------------
# version 1, non-differentiable

def f1(xi, p):
    if (xi < p):
        return 1
    else:
        return 3

def f1_grad(xi, p):
    if (xi < p):
        return 0 # partial derivate wrt p
    else:
        return 0 # partial derivate wrt p

# ----------------------------------------------------
# version 2, differentiable

def f2(xi, p):
    if (xi < p):
        return 1 * p / detach(p)
    else:
        return 3 * (1 - p) / detach(1 - p)

def f2_grad(xi, p):
    if (xi < p):
        return 1 / p       # partial derivate wrt p
    else:
        return 3 / (p - 1) # partial derivate wrt p

# ----------------------------------------------------
# integrator

def monte_carlo(f, xi, p):
    F = 0
    for i in range(len(xi)):
        F += f(xi[i], p)
    return F / len(xi)

def l2(x, y):
    return (x - y) ** 2

def l2_grad(x, y):
    return 2 * (x - y)

def optimize(p, TARGET, EPOCHS, SAMPLES, LR, replay=False, EPS=1e-6):
    for i in range(EPOCHS):
        xi = np.random.random((int(SAMPLES)))    # draw samples
        x = monte_carlo(f2, xi, p)          # forward simulation
        if not replay:
            xi = np.random.random((int(SAMPLES)))    # re-draw samples?
        dx = monte_carlo(f2_grad, xi, p)    # backward simulation (replay)
        dloss = l2_grad(x, TARGET)          # compute loss gradient
        p -= LR * dx * dloss                # gradient update step
        # print(f'epoch: {i+1:04}, p: {p:.2f}, x: {x:.4f}/{TARGET:.4f}, grad_x: {dx:.2f}, loss: {l2(x, TARGET):.2f}')
    return p

# optimize(0.1, 2.0, 10000, 1, 0.0001, False)

plt.style.use('_mpl-gallery')

# make data
x = np.linspace(1, 128, 100)
y_true = [optimize(0.1, 2.0, 10000, s, 0.0001, True) for s in x]
y_false = [optimize(0.1, 2.0, 10000, s, 0.0001, False) for s in x]

# plot
fig = plt.figure(figsize=(16/2, 9/2))
ax = fig.add_subplot()

ax.plot(x, y_true, linewidth=2.0, label='with replay')
ax.plot(x, y_false, linewidth=2.0, label='no replay')
ax.axhline(0.5, linestyle='--', label='target', color='0.5')

ax.set(xlim=(0, x[-1]), ylim=(0, 1))

# ax.set_title("Monte-Carlo backprop convergence")
ax.set_xlabel("N samples")
ax.set_ylabel("optimization result")
ax.legend()
# plt.tight_layout()


plt.savefig("convergence.pdf", format="pdf", bbox_inches="tight")
# plt.show()
