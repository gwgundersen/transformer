from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Model(nnx.Module):

    def __init__(self, d_in, d_mid, d_out, rngs: nnx.Rngs):
        self.linear = nnx.Linear(d_in, d_mid, rngs=rngs)
        self.bn = nnx.BatchNorm(d_mid, rngs=rngs)
        self.dropout = nnx.Dropout(0.2)
        self.linear_out = nnx.Linear(d_mid, d_out, rngs=rngs)
        self.rngs = rngs
    
    def __call__(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x, rngs=self.rngs)
        x = nnx.relu(x)
        return self.linear_out(x)


@nnx.jit
def train_step(model, optimizer, x, y):
    loss_fn = lambda model: ((model(x) - y) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# Data generation
NUM_SAMPLES = 100
X_DIM = 10
Y_DIM = 5
rng = jax.random.PRNGKey(0)
x_samples = jax.random.normal(rng, (NUM_SAMPLES, X_DIM))
w = jax.random.normal(rng, (X_DIM, Y_DIM))
b = jax.random.normal(rng, (Y_DIM,))
# y = W*x + b
y_samples = jnp.dot(x_samples, w) + b
print(x_samples.shape, y_samples.shape)

# Model and optimizer
model = Model(X_DIM, 64, Y_DIM, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# Training
N_STEPS = 100
loss_history = []
for _ in range(N_STEPS):
    loss = train_step(model, optimizer, x_samples, y_samples)
    loss_history.append(float(loss))
loss_history = np.array(loss_history)

print(loss_history)
