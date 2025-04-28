"""

Challenging nonlinear regression with neural networks using JAX and Optax.

"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from typing import List, Tuple, NamedTuple
from functools import partial
from time import time


# The default configuration will have around 21,000 parameters

# Configuration
class Config:
    # Data parameters
    data_size = 4_000
    train_ratio = 0.8
    noise_scale = 0.2
    # Model parameters
    hidden_layers = [128, 128, 32]
    activation = "selu"  # Options: "relu", "selu", "tanh", "sigmoid"
    # Training parameters
    batch_size = 128
    epochs = 20_000
    init_lr = 0.001
    min_lr = 0.0001
    warmup_steps = 100
    decay_steps = 300
    regularization_term = 1e-5
    # Evaluation
    eval_every = 100


@jax.jit
def f(x):
    """
    Function to be estimated.
    """
    term1 = 2 * jnp.sin(3 * x) * jnp.cos(x/2)
    term2 = 0.5 * x**2 * jnp.cos(5*x) / (1 + 0.1 * x**2)
    term3 = 3 * jnp.exp(-0.2 * (x - 4)**2) * jnp.sin(10*x)
    term4 = 1.5 * jnp.tanh(x/3) * jnp.sin(7*x)
    term5 = 0.8 * jnp.log(jnp.abs(x) + 1) * jnp.cos(x**2 / 8)
    term6 = jnp.where(x > 0, 2 * jnp.sin(3*x), -2 * jnp.sin(3*x))  # Discontinuity
    return term1 + term2 + term3 + term4 + term5 + term6


def generate_data(
        key: jax.Array,
        data_size: int = Config.data_size
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic nonlinear regression data.

    """
    x_key, y_key = jax.random.split(key)
    # x is generated uniformly
    x = jax.random.uniform(
            x_key, (data_size, 1), minval=-10.0, maxval=10.0
        )
    # y = f(x) + noise
    σ =  Config.noise_scale
    w = σ * jax.random.normal(y_key, shape=x.shape)
    y = f(x) + w
    return x, y


class LayerParams(NamedTuple):
    """
    Stores parameters for one layer of the neural network.

    """
    W: jnp.ndarray     # weights
    b: jnp.ndarray     # biases


def init_layer_params(
        key: jax.Array, 
        in_dim: int, 
        out_dim: int,
        activation_name: str = Config.activation
    ) -> Tuple[LayerParams, jax.Array]:
    """
    Initialize parameters for a single layer using appropriate initialization
    based on the activation function.
    
    - He initialization for ReLU and its variants
    - LeCun initialization for SELU
    - Glorot/Xavier initialization for tanh and sigmoid

    """
    key, w_key, b_key = jax.random.split(key, 3)
    
    # Choose initialization strategy based on activation function
    if activation_name == "selu":
        # LeCun initialization 
        s = jnp.sqrt(1.0 / in_dim)
        W = jax.random.normal(w_key, (in_dim, out_dim)) * s
        b = jnp.zeros((out_dim,))
    elif activation_name in ["tanh", "sigmoid"]:
        # Glorot/Xavier initialization
        s = jnp.sqrt(6.0 / (in_dim + out_dim))
        W = jax.random.uniform(w_key, (in_dim, out_dim), minval=-s, maxval=s)
        b = jnp.zeros((out_dim,))
    else:
        # He initialization (default for ReLU and variants)
        s = jnp.sqrt(2.0 / in_dim)
        W = jax.random.normal(w_key, (in_dim, out_dim)) * s
        b = jnp.zeros((out_dim,))
    
    return LayerParams(W=W, b=b), key


def initialize_network_params(
        key: jax.Array, 
        layer_sizes: List[int],
        activation_name: str = Config.activation
    ) -> List[LayerParams]:
    """
    Initialize all parameters for the network.

    """
    θ = []
    # For all layers but the last one
    for i in range(len(layer_sizes) - 1):
        # Generate an instance of LayerParams corresponding to layer i
        layer, key = init_layer_params(
            key, 
            layer_sizes[i],      # in dimension for layer
            layer_sizes[i + 1],  # out dimension for layer
            activation_name
        )
        # And append it to the list the contains all network parameters.
        θ.append(layer)
        
    return θ


@partial(jax.jit, static_argnames=['activation'])
def forward(
        θ: List[LayerParams], 
        x: jnp.ndarray, 
        activation: str = Config.activation
    ) -> jnp.ndarray:

    """
    Forward pass through the neural network.
    
    Args:
        θ: network parameters
        x: input data
        activation: activation function name (static argument)
    """
    
    # Select the activation function based on name
    if activation == "relu":
        σ = jax.nn.relu
    elif activation == "selu":
        σ = jax.nn.selu
    elif activation == "tanh":
        σ = jnp.tanh
    elif activation == "gelu":
        σ = jax.nn.gelu
    elif activation == "sigmoid":
        σ = jax.nn.sigmoid
    elif activation == "elu":
        σ = jax.nn.elu
    else:
        # Default to selu
        σ = jax.nn.selu
    
    # Apply all layers except the last, with activation
    for W, b in θ[:-1]:
        x = σ(x @ W + b)
    # Apply last layer without activation (for regression)
    W, b = θ[-1]
    output = x @ W + b
    
    return output


@partial(jax.jit, static_argnames=['activation'])
def mse_loss(
        params: List[LayerParams], 
        x: jnp.ndarray, 
        y: jnp.ndarray,
        activation: str = "relu"
    ) -> jnp.ndarray:

    """
    Mean squared error loss function.

    """
    y_pred = forward(params, x, activation=activation)
    return jnp.mean((y_pred - y) ** 2)


@partial(jax.jit, static_argnames=['activation'])
def regularized_loss(
        params: List[LayerParams], 
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        activation: str = "selu",
        λ: float = Config.regularization_term
    ) -> jnp.ndarray:
    """
    Loss function with L2 regularization.

    """
    mse = mse_loss(params, x, y, activation=activation)
    
    # L2 regularization
    l2_penalty = 0.0
    for layer in params:
        l2_penalty += jnp.sum(layer.W ** 2)
    
    return mse + λ * l2_penalty


def create_lr_schedule():
    """
    Create an Optax learning rate schedule with warmup and decay.

    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=Config.init_lr,
        transition_steps=Config.warmup_steps
    )
    
    decay_fn = optax.exponential_decay(
        init_value=Config.init_lr,
        transition_steps=Config.decay_steps,
        decay_rate=0.5,
        end_value=Config.min_lr
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[Config.warmup_steps]
    )


def training_step_factory(optimizer, activation: str = Config.activation):
    """
    Create a JIT-compiled training step function.

    """
    
    # Create a specialized loss gradient function for this activation
    loss_grad = jax.grad(lambda p, x, y: regularized_loss(p, x, y, activation=activation))
    
    @jax.jit
    def train_step(θ, opt_state, x_batch, y_batch):
        """Single training step."""
        grads = loss_grad(θ, x_batch, y_batch)
        loss_val = regularized_loss(θ, x_batch, y_batch, activation=activation)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, θ)
        θ = optax.apply_updates(θ, updates)
        
        return θ, new_opt_state, loss_val
    
    return train_step


def create_data_batch_iterator(
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        key: jax.Array,
        batch_size: int,
    ) -> List[Tuple[jax.Array]]:
    """
    Create a list of batched data.  Each element of the list is a tuple
    (x_batch, y_batch), containing a batch of data for training.

    """
    num_samples = x.shape[0]
    
    # Shuffle the data
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    
    # Create batches
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batches = []
    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        batches.append((x_batch, y_batch))
    
    return batches


def evaluate_mse(
        θ: List[LayerParams], 
        x: jnp.ndarray, 
        y: jnp.ndarray,
        activation: str = Config.activation
    ) -> float:
    """
    Compute the loss on data (x, y) without regularization.

    """
    return float(mse_loss(θ, x, y, activation=activation))



def train(θ, activation, x_train, y_train, x_val, y_val, key):
    """
    Train the neural network.

    """
    
    # Create optimizer with learning rate schedule
    lr_schedule = create_lr_schedule()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(θ)
    
    # Create training step function
    train_step_fn = training_step_factory(optimizer, activation)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_params = θ
    patience_counter = 0
    patience = 50   # Early stopping patience (in terms of evaluation intervals)
    
    print(f"Starting training for {Config.epochs} epochs...")
    start = time()

    # One epoch is a complete pass through the data set
    for epoch in range(Config.epochs):

        # Create shuffled batches for this epoch
        key, subkey = jax.random.split(key)
        batches = create_data_batch_iterator(x_train, y_train, subkey, Config.batch_size)
        
        # Process each batch, updating parameters 
        epoch_losses = []
        for x_batch, y_batch in batches:
            θ, opt_state, loss = train_step_fn(θ, opt_state, x_batch,
                                                    y_batch)
            epoch_losses.append(loss)
            
        # Calculate average loss for this epoch
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set periodically
        if epoch % Config.eval_every == 0 or epoch == Config.epochs - 1:

            val_loss = evaluate_mse(θ, x_val, y_val, activation)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_theta = jax.tree.map(lambda p: p, θ)  # Copy the params
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    elapsed = time() - start

    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return θ, (train_losses, val_losses)


def run_activation_comparison():
    """Compare different activation functions."""
    activations_to_compare = ["relu", "selu", "tanh", "gelu"]
    results = {}
    
    for activation in activations_to_compare:
        print(f"\n{'='*50}")
        print(f"Training with {activation.upper()} activation")
        print(f"{'='*50}\n")
        
        # Train with current activation function
        θ, (train_losses, val_losses) = train(SEED, activation)
        
        # Store results
        results[activation] = {
            "θ": θ,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_val_loss": val_losses[-1]
        }
    
    # Compare final validation losses
    print("\nComparison of final validation losses:")
    print("-" * 40)
    for name, res in sorted(results.items(), key=lambda x: x[1]["final_val_loss"]):
        print(f"{name.upper():10s}: {res['final_val_loss']:.6f}")
    
    # Plot comparative learning curves
    fig, ax = plt.subplots()
    for name, res in results.items():
        ax.plot(
            np.arange(0, len(res["val_losses"]) * Config.eval_every, Config.eval_every),
            res["val_losses"],
            label=f"{name.upper()} (final: {res['final_val_loss']:.6f})"
        )
    
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation MSE Loss')
    ax.set_title('Comparison of activation functions')
    ax.legend()
    ax.set_yscale('log')  # Log scale often better shows differences
    plt.show()
    
    return results


print(f"Using JAX version: {jax.__version__}")
print(f"Device: {jax.devices()[0]}")

# == Train == #

SEED = 42 # Set random seed for reproducibility
key = jax.random.PRNGKey(SEED)

# Produce separate keys for training and validation data
key, train_data_key, val_data_key = jax.random.split(key, 3)

# Generate training and validation data
print("Generating data...")
train_data_size = Config.data_size
x_train, y_train = generate_data(train_data_key, train_data_size)
val_data_size = int(Config.data_size * 0.5)  # half of training data size
x_val, y_val = generate_data(val_data_key, val_data_size)

# Define model architecture
input_dim = 1  # scalar input
output_dim = 1 # scalar output
layer_sizes = [input_dim] + Config.hidden_layers + [output_dim]

# Choose activation function
activation = Config.activation
print(f"Using activation function: {activation}")

# Initialize all the parameters in the network
print(f"Initializing model with layer sizes: {layer_sizes}")
key, subkey = jax.random.split(key)
θ_init = initialize_network_params(subkey, layer_sizes, activation)
    
# Train the model using training data, compute losses using validation data
θ, (train_losses, val_losses) = train(
        θ_init, activation, x_train, y_train, x_val, y_val, key
)


def plot_regression():
    """
    Plot original and fitted functions.

    """
    x_grid = jnp.linspace(-10.0, 10.0, 200)
    y_pred = forward(θ, x_grid.reshape(-1, 1), activation=activation)
    
    fig, ax = plt.subplots()
    # Plot training data
    ax.scatter(x_train.flatten(), y_train.flatten(), 
                alpha=0.2, color='blue', label='training data')
    
    # Plot the predicted curve
    ax.plot(x_grid, y_pred.flatten(), 
             color='red', 
             linewidth=2, 
             linestyle='--',
             label='model prediction')
    
    # Plot the true function (without noise)
    y_true = f(x_grid)
    ax.plot(x_grid, y_true, 
             color='black', ls='--',
             linewidth=2, label='true function')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend()
    
    plt.show()


def plot_learning_curves():
    """
    Plot the MSE curves on training and validation data over epochs.

    """
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='training Loss')
    ax.plot(np.arange(0, len(val_losses) * Config.eval_every, Config.eval_every), 
             val_losses, label='validation Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Learning curves with {Config.activation.upper()}')
    ax.legend()
    plt.show()

# Option 2: Uncomment to run comparative analysis
# results = run_activation_comparison()
