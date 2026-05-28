"""
mlp.py
------
Simple 2-layer MLP used as the local FL model.

Architecture mirrors the paper's MNIST experiments:
    Input  →  128 (ReLU)  →  n_classes (Softmax / logits)

All methods return plain NumPy arrays so the code stays
framework-agnostic and easy to inspect.
"""

import numpy as np
from typing import Dict, Tuple, List


# ---------------------------------------------------------------------------
# Activations & losses
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    return -np.log(probs[np.arange(n), y] + 1e-9).mean()

def cross_entropy_grad(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of softmax cross-entropy w.r.t. pre-softmax logits."""
    n = len(y)
    g = probs.copy()
    g[np.arange(n), y] -= 1
    return g / n


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(
    n_features: int,
    hidden: int = 128,
    n_classes: int = 2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Xavier / Glorot initialisation.
    Returns an OrderedDict-like dict with keys:
        W1, b1  – hidden layer
        W2, b2  – output layer
    """
    rng = np.random.default_rng(seed)
    def xavier(fan_in, fan_out):
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)

    return {
        "W1": xavier(n_features, hidden),
        "b1": np.zeros((1, hidden),    dtype=np.float32),
        "W2": xavier(hidden, n_classes),
        "b2": np.zeros((1, n_classes), dtype=np.float32),
    }


def copy_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in weights.items()}


def flatten_weights(weights: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all parameters into a 1-D vector."""
    return np.concatenate([v.ravel() for v in weights.values()])


def unflatten_weights(
    flat: np.ndarray, template: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Restore a flat vector back to the layer-dict structure."""
    out, idx = {}, 0
    for k, v in template.items():
        size = v.size
        out[k] = flat[idx: idx + size].reshape(v.shape)
        idx += size
    return out


def weight_dimension(weights: Dict[str, np.ndarray]) -> int:
    return sum(v.size for v in weights.values())


# ---------------------------------------------------------------------------
# Forward / backward pass
# ---------------------------------------------------------------------------

def forward(
    weights: Dict[str, np.ndarray], X: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Returns (probs, cache) where cache holds intermediates for backprop.
    """
    z1 = X @ weights["W1"] + weights["b1"]   # (N, 128)
    a1 = relu(z1)                             # (N, 128)
    z2 = a1 @ weights["W2"] + weights["b2"]  # (N, C)
    probs = softmax(z2)                       # (N, C)
    cache = dict(X=X, z1=z1, a1=a1, z2=z2)
    return probs, cache


def backward(
    weights: Dict[str, np.ndarray],
    cache: dict,
    y: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute gradients via backprop. Returns grad dict matching weights."""
    dz2 = cross_entropy_grad(probs, y)            # (N, C)
    dW2 = cache["a1"].T @ dz2                     # (128, C)
    db2 = dz2.sum(axis=0, keepdims=True)          # (1, C)

    da1 = dz2 @ weights["W2"].T                   # (N, 128)
    dz1 = da1 * relu_grad(cache["z1"])            # (N, 128)
    dW1 = cache["X"].T @ dz1                      # (F, 128)
    db1 = dz1.sum(axis=0, keepdims=True)          # (1, 128)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def predict(weights: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    probs, _ = forward(weights, X)
    return probs.argmax(axis=1)


def accuracy(weights: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    return (predict(weights, X) == y).mean()


def evaluate_loss(
    weights: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray
) -> float:
    probs, _ = forward(weights, X)
    return cross_entropy_loss(probs, y)


def local_train(
    weights: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 16,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    SGD training for `epochs` passes over the local dataset.
    Returns updated weights (original dict is NOT modified).
    """
    w = copy_weights(weights)
    rng = np.random.default_rng(seed)
    n = len(X)

    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start: start + batch_size]
            Xb, yb = X[batch], y[batch]
            probs, cache = forward(w, Xb)
            grads = backward(w, cache, yb, probs)
            for key in w:
                w[key] -= lr * grads[key]
    return w


# ---------------------------------------------------------------------------
# Layer-wise statistics  (used by adaptive clipping in ACS-FL)
# ---------------------------------------------------------------------------

def layer_stats(
    weights: Dict[str, np.ndarray]
) -> Dict[str, Tuple[float, float]]:
    """
    Returns {layer_name: (center c_l, radius r_l)} for each parameter tensor.

    c_l = mean of all weights in layer l
    r_l = max |w - c_l|  for w in layer l
    """
    stats = {}
    for name, W in weights.items():
        c = float(W.mean())
        r = float(np.abs(W - c).max())
        r = max(r, 1e-8)       # avoid zero radius
        stats[name] = (c, r)
    return stats


def average_weights(
    weight_list: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """FedAvg aggregation: element-wise mean across a list of weight dicts."""
    avg = copy_weights(weight_list[0])
    for key in avg:
        avg[key] = np.mean([w[key] for w in weight_list], axis=0)
    return avg


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 7)).astype(np.float32)
    y = rng.integers(0, 2, 50)

    w = init_weights(n_features=7, hidden=128, n_classes=2)
    loss_before = evaluate_loss(w, X, y)
    w_trained   = local_train(w, X, y, lr=0.01, epochs=10)
    loss_after  = evaluate_loss(w_trained, X, y)
    print(f"Loss before: {loss_before:.4f}  →  after: {loss_after:.4f}")
    print(f"Accuracy: {accuracy(w_trained, X, y):.3f}")
    print(f"Layer stats: {layer_stats(w_trained)}")
