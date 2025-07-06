import numpy as np
import matplotlib.pyplot as plt

# %%ANN architecture

f = np.tanh
g = lambda x: x

neurons = 15
layers = 1
inputs = 1
outputs = 1

#data
x = np.arange(0, 10.25, 0.25)
y = np.sin(x)
y_train = (y - np.min(y)) / np.max(y - np.min(y))
x_train = x.reshape(1, -1)

#weights and biases
w_shapes = [(neurons, inputs)] + [(neurons, neurons)] * (layers - 1) + [(outputs, neurons)]
b_shapes = [(neurons, 1)] * layers + [(outputs, 1)]

n_weights = sum(np.prod(shape) for shape in w_shapes)
n_biases = sum(np.prod(shape) for shape in b_shapes)
n_params = n_weights + n_biases

# %%indexing the  ann parameter vector
def unpack_params(s):
    weights, biases = [], []
    idx = 0
    for shape in w_shapes:
        size = np.prod(shape)
        weights.append(s[idx:idx+size].reshape(shape))
        idx += size
    for shape in b_shapes:
        size = np.prod(shape)
        biases.append(s[idx:idx+size].reshape(shape))
        idx += size
    return weights, biases
# %% net output and error func

def forward_pass(s):
    weights, biases = unpack_params(s)
    a = x_train
    for i in range(layers):
        a = f(weights[i] @ a + biases[i])
    out = g(weights[-1] @ a + biases[-1])
    return out.flatten()

def cost_fun(s):
    return np.mean((y_train - forward_pass(s))**2)

# %%jacobian matrix

def jacobi(f, s, h):
    s = np.asarray(s)
    J = np.zeros_like(s)
    for i in range(len(s)):
        s1 = s.copy(); s1[i] += h
        s2 = s.copy(); s2[i] -= h
        J[i] = (f(s1) - f(s2)) / (2 * h)
    return J

# %%hessian matrix

def hesse(f, v, h):
    v = np.asarray(v)
    n = v.size
    H = np.full((n, n), np.nan)
    S = [np.eye(1, n, i).flatten() * h for i in range(n)]
    for i in range(n):
        for j in range(n):
            term1 = f(v + S[i] + S[j])
            term2 = f(v - S[i] + S[j])
            term3 = f(v + S[i] - S[j])
            term4 = f(v - S[i] - S[j])
            H[i, j] = (term1 - term2 - term3 + term4) / (4 * h**2)
    return H

# %% levenberg–marquardt
def solvelm(f, s, n_max, e_max, h):
    r = 1e-2
    d = 2
    e = [f(s)]
    for i in range(1, int(n_max) + 1):
        J = jacobi(f, s, h)
        H = hesse(f, s, h)
        try:
            ds = -np.linalg.solve(H + r * np.eye(len(s)), J.T)
        except np.linalg.LinAlgError:
            break
        n = 0
        while n < n_max / 10 and f(s + ds) >= e[-1]:
            r *= d
            try:
                ds = -np.linalg.solve(H + r * np.eye(len(s)), J.T)
            except np.linalg.LinAlgError:
                break
            n += 1
        e_new = f(s + ds)
        if e_new <= e_max or e_new > e[-1] or np.isnan(ds).any():
            break
        else:
            s = s + ds
            r /= d
            e.append(e_new)
    return s, np.array(e)

# %% training
h = 1e-6
epochs = 300
goal = 1e-4

s, e = solvelm(cost_fun, np.ones(n_params), epochs, goal, h)

weights, biases = unpack_params(s)

def net(x_input):
    a = x_input.reshape(1, -1)
    for i in range(layers):
        a = f(weights[i] @ a + biases[i])
    out = g(weights[-1] @ a + biases[-1])
    return np.max(y - np.min(y)) * out.flatten() + np.min(y)

plt.figure()
plt.semilogy(e, 'b-')
plt.xlabel('Epochs')
plt.ylabel('log(e)')
plt.grid(True)
plt.title('Error')
plt.tight_layout()

plt.figure()
plt.plot(x, net(x), '--r', label='Net Out')
plt.plot(x, y, 'b.', label='Real Out')
plt.legend()
plt.title(f'MSE = {e[-1]:.6f}')
plt.grid(True)
plt.tight_layout()
plt.show()