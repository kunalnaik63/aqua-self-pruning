import unittest
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy
from nn.layers import Linear
from nn.optim import Adam
from prune.pruning import set_mask


def numerical_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps; xm[idx] -= eps
        grad[idx] = (fn(xp) - fn(xm)) / (2 * eps)
    return grad


class GradientChecks(unittest.TestCase):
    def check_unary(self, op, fn, x):
        t = Tensor(x); out = op(t).sum(); out.backward()
        np.testing.assert_allclose(t.grad, numerical_grad(lambda z: np.asarray(fn(z)).sum(), x), rtol=2e-4, atol=2e-5)

    def check_binary(self, op, fn, x, y):
        a, b = Tensor(x), Tensor(y)
        out = op(a, b).sum(); out.backward()
        gx = numerical_grad(lambda z: np.asarray(fn(z, y)).sum(), x)
        gy = numerical_grad(lambda z: np.asarray(fn(x, z)).sum(), y)
        np.testing.assert_allclose(a.grad, gx, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(b.grad, gy, rtol=2e-4, atol=2e-5)

    def test_add_sub_mul_div_broadcast(self):
        x = np.array([[1.2, 2.1, 3.2], [0.7, 1.4, 2.2]])
        y = np.array([2.0, 3.0, 4.0])
        self.check_binary(lambda a,b: a+b, lambda a,b:a+b, x, y)
        self.check_binary(lambda a,b: a-b, lambda a,b:a-b, x, y)
        self.check_binary(lambda a,b: a*b, lambda a,b:a*b, x, y)
        self.check_binary(lambda a,b: a/b, lambda a,b:a/b, x, y)

    def test_matmul(self):
        x = np.array([[0.2, -0.4, 0.8], [1.0, 0.5, -0.3]])
        w = np.array([[0.4, 0.1], [-0.2, 0.6], [0.3, -0.7]])
        self.check_binary(lambda a,b: a@b, lambda a,b:a@b, x, w)

    def test_sum_and_mean(self):
        x = np.array([[1., 2., 3.], [4., 5., 6.]])
        self.check_unary(lambda a:a.sum(axis=0), lambda z:z.sum(axis=0), x)
        self.check_unary(lambda a:a.mean(axis=1), lambda z:z.mean(axis=1), x)

    def test_activations(self):
        x = np.array([-1.3, -0.2, 0.4, 1.1])
        self.check_unary(lambda a:a.relu(), lambda z:np.maximum(z,0), x)
        self.check_unary(lambda a:a.tanh(), np.tanh, x)
        self.check_unary(lambda a:a.sigmoid(), lambda z:1/(1+np.exp(-z)), x)
        self.check_unary(lambda a:a.gelu(), lambda z:0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3))), x)

    def test_softmax(self):
        x = np.array([[0.2, -0.1, 0.7], [1.0, 0.5, -0.2]])
        weights = np.array([[0.3, -0.7, 1.2], [0.2, 0.9, -0.4]])
        def fn(z):
            e = np.exp(z-z.max(axis=1, keepdims=True)); return ((e/e.sum(axis=1, keepdims=True))*weights).sum()
        t = Tensor(x); out = (t.softmax(axis=1) * Tensor(weights)).sum(); out.backward()
        np.testing.assert_allclose(t.grad, numerical_grad(fn, x), rtol=2e-4, atol=2e-5)

    def test_shared_graph_gradient_accumulation(self):
        a = Tensor(3.0)
        b = a * a
        c = b + a
        c.backward()
        # d(a*a + a)/da = 2*a + 1 = 7; both graph paths must accumulate.
        self.assertAlmostEqual(float(a.grad), 7.0, places=8)

    def test_stable_cross_entropy_large_logits(self):
        logits = Tensor(np.array([[1000.0, 1001.0, 999.0]]))
        loss = softmax_cross_entropy(logits, np.array([1]))
        loss.backward()
        self.assertTrue(np.isfinite(loss.data).all())
        self.assertTrue(np.isfinite(logits.grad).all())

    def test_scalar_broadcast_gradient(self):
        x = Tensor(np.array([1., 2., 3.]))
        b = Tensor(2.0)
        loss = (x + b).sum(); loss.backward()
        self.assertEqual(float(b.grad), 3.0)

    def test_cross_entropy(self):
        x = np.array([[1.2, -0.4, 0.1], [0.2, 0.7, -0.1]])
        labels = np.array([0, 1])
        t = Tensor(x); loss = softmax_cross_entropy(t, labels); loss.backward()
        def fn(z):
            z=z-z.max(axis=1,keepdims=True); e=np.exp(z); p=e/e.sum(axis=1,keepdims=True)
            return -np.log(p[np.arange(len(labels)), labels]).mean()
        np.testing.assert_allclose(t.grad, numerical_grad(fn, x), rtol=2e-4, atol=2e-5)

    def test_masked_gradient_and_adam_revival_state(self):
        rng = np.random.default_rng(1)
        layer = Linear(2, 2, rng)
        opt = Adam(layer.parameters(), lr=0.01)
        opt.set_mask(layer.weight, layer.mask)
        x = Tensor(np.array([[1., 2.], [3., 4.]]))
        out = layer(x).sum(); out.backward()
        self.assertTrue(np.all(layer.weight.grad != 0))
        # Seed nonzero optimizer state, prune one connection, and verify zero grad/state.
        opt.m[id(layer.weight)][0, 0] = 7.0
        opt.v[id(layer.weight)][0, 0] = 9.0
        mask = layer.mask.copy(); mask[0, 0] = False
        set_mask(layer, opt, mask)
        x2 = Tensor(np.array([[1., 2.], [3., 4.]]))
        layer(x2).sum().backward()
        self.assertEqual(layer.weight.grad[0, 0], 0.0)
        self.assertEqual(layer.weight.data[0, 0], 0.0)
        self.assertEqual(opt.m[id(layer.weight)][0, 0], 0.0)
        self.assertEqual(opt.v[id(layer.weight)][0, 0], 0.0)
        # Revival is explicit and starts with fresh optimizer history.
        mask[0, 0] = True
        layer.weight.data[0, 0] = 0.05
        set_mask(layer, opt, mask)
        self.assertEqual(opt.m[id(layer.weight)][0, 0], 0.0)
        self.assertEqual(opt.v[id(layer.weight)][0, 0], 0.0)
        layer.weight.grad[0, 0] = 1.0
        opt.step()
        self.assertNotEqual(layer.weight.data[0, 0], 0.05)


if __name__ == "__main__":
    unittest.main()
