"""
Nonlinear Regression with Neural Networks using JAX and Optax

This code demonstrates using neural networks to solve a nonlinear regression problem
with JAX and Optax. It includes:
- Data generation with complex nonlinear patterns
- Model definition using a functional approach
- Parameter initialization and management
- Training with mini-batches and learning rate scheduling
- Model evaluation and visualization
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable, NamedTuple
from functools import partial

# Set random seed for reproducibility
SEED = 42
key = jax.random.PRNGKey(SEED)

# Configuration
class Config:
    # Data parameters
    data_size = 1000
    train_ratio = 0.8
    noise_scale = 0.2
    
    # Model parameters
    hidden_layers = [64, 32, 16]
    activation = jax.nn.relu
    
    # Training parameters
    batch_size = 32
    epochs = 500
    init_lr = 0.001
    min_lr = 0.0001
    warmup_steps = 100
    decay_steps = 300
    weight_decay = 1e-5
    
    # Evaluation
    eval_every = 50


# Data Generation
def generate_data(key: jax.Array, 
                  size: int = Config.data_size, 
                  noise_scale: float = Config.noise_scale) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic nonlinear regression data.
    
    The target function is a complex nonlinear function:
    y = sin(x) + 0.1 * x^2 + 0.5 * cos(3*x) + noise
    """
    key, subkey = jax.random.split(key)
    
    # Generate x values between -3 and 3
    x = jax.random.uniform(key, (size, 1), minval=-3.0, maxval=3.0)
    
    # Generate nonlinear target
    y_clean = jnp.sin(x) + 0.1 * x**2 + 0.5 * jnp.cos(3*x)
    
    # Add noise
    noise = jax.random.normal(subkey, shape=y_clean.shape) * noise_scale
    y = y_clean + noise
    
    return x, y


# Split data into training and validation sets
def train_val_split(x: jnp.ndarray, 
                    y: jnp.ndarray, 
                    train_ratio: float = Config.train_ratio) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split data into training and validation sets."""
    n = x.shape[0]
    indices = jnp.arange(n)
    train_size = int(n * train_ratio)
    
    # Shuffle indices
    shuffled_indices = jax.random.permutation(key, indices)
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]
    
    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]
    
    return x_train, y_train, x_val, y_val


# Model Definition
class LayerParams(NamedTuple):
    """Parameters for a single neural network layer."""
    weights: jnp.ndarray
    bias: jnp.ndarray


def init_layer_params(key: jax.Array, 
                      in_dim: int, 
                      out_dim: int) -> LayerParams:
    """Initialize parameters for a single layer using He initialization."""
    # He initialization
    key, w_key, b_key = jax.random.split(key, 3)
    scale = jnp.sqrt(2.0 / in_dim)
    weights = jax.random.normal(w_key, (in_dim, out_dim)) * scale
    bias = jnp.zeros((out_dim,))
    
    return LayerParams(weights=weights, bias=bias), key


def init_network_params(key: jax.Array, 
                        layer_sizes: List[int]) -> List[LayerParams]:
    """Initialize all parameters for the network."""
    params = []
    
    for i in range(len(layer_sizes) - 1):
        layer_params, key = init_layer_params(key, layer_sizes[i], layer_sizes[i + 1])
        params.append(layer_params)
        
    return params


@jax.jit
def forward(params: List[LayerParams], 
            x: jnp.ndarray, 
            activation: Callable = Config.activation) -> jnp.ndarray:
    """Forward pass through the neural network."""
    # Implementation using a functional approach with JAX
    activations = x
    
    # Apply all layers except the last with activation
    for layer_params in params[:-1]:
        activations = activation(activations @ layer_params.weights + layer_params.bias)
    
    # Apply last layer without activation (linear output)
    final_params = params[-1]
    output = activations @ final_params.weights + final_params.bias
    
    return output


@jax.jit
def mse_loss(params: List[LayerParams], 
             x: jnp.ndarray, 
             y: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss function."""
    y_pred = forward(params, x)
    return jnp.mean((y_pred - y) ** 2)


@jax.jit
def regularized_loss(params: List[LayerParams], 
                     x: jnp.ndarray, 
                     y: jnp.ndarray, 
                     weight_decay: float = Config.weight_decay) -> jnp.ndarray:
    """Loss function with L2 regularization."""
    mse = mse_loss(params, x, y)
    
    # L2 regularization
    l2_penalty = 0.0
    for param in params:
        l2_penalty += jnp.sum(param.weights ** 2)
    
    return mse + weight_decay * l2_penalty


# Create learning rate schedule
def create_lr_schedule():
    """Create learning rate schedule with warmup and decay."""
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


def create_train_step(optimizer):
    """Create a JIT-compiled training step function."""
    
    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        """Single training step."""
        loss_fn = lambda p: regularized_loss(p, batch_x, batch_y)
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss_val
    
    return train_step


def create_data_batch_iterator(x: jnp.ndarray, 
                              y: jnp.ndarray, 
                              batch_size: int = Config.batch_size,
                              key: jax.Array = None):
    """Create a batched data iterator."""
    num_samples = x.shape[0]
    
    # Create a new key if none provided
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Shuffle the data
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    
    # Create batches
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batches = []
    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_x, batch_y))
    
    return batches


def evaluate(params: List[LayerParams], 
             x: jnp.ndarray, 
             y: jnp.ndarray) -> float:
    """Evaluate the model on validation data."""
    return float(mse_loss(params, x, y))


def plot_predictions(params: List[LayerParams],
                    x_train: jnp.ndarray,
                    y_train: jnp.ndarray,
                    x_val: jnp.ndarray,
                    y_val: jnp.ndarray):
    """Plot the data and model predictions."""
    # Create a grid of x values for the curve
    x_grid = jnp.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    
    # Get predictions
    y_pred = forward(params, x_grid)
    
    # Convert to numpy for matplotlib
    x_grid_np = np.array(x_grid).flatten()
    y_pred_np = np.array(y_pred).flatten()
    x_train_np = np.array(x_train).flatten()
    y_train_np = np.array(y_train).flatten()
    x_val_np = np.array(x_val).flatten()
    y_val_np = np.array(y_val).flatten()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(x_train_np, y_train_np, alpha=0.5, color='blue', label='Training data')
    
    # Plot validation data
    plt.scatter(x_val_np, y_val_np, alpha=0.5, color='green', label='Validation data')
    
    # Plot the predicted curve
    plt.plot(x_grid_np, y_pred_np, color='red', linewidth=2, label='Model prediction')
    
    # Plot the true function (without noise)
    y_true = np.sin(x_grid_np) + 0.1 * x_grid_np**2 + 0.5 * np.cos(3*x_grid_np)
    plt.plot(x_grid_np, y_true, color='black', linestyle='--', linewidth=2, label='True function')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Neural Network Nonlinear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt


def train(key: jax.Array = SEED):
    """Train the neural network model."""
    key = jax.random.PRNGKey(key)
    key, subkey = jax.random.split(key)
    
    print("Generating data...")
    x, y = generate_data(subkey)
    x_train, y_train, x_val, y_val = train_val_split(x, y)
    
    print(f"Train data shape: {x_train.shape}, Validation data shape: {x_val.shape}")
    
    # Define model architecture
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
    layer_sizes = [input_dim] + Config.hidden_layers + [output_dim]
    
    print(f"Initializing model with layer sizes: {layer_sizes}")
    key, subkey = jax.random.split(key)
    params = init_network_params(subkey, layer_sizes)
    
    # Create optimizer with learning rate schedule
    lr_schedule = create_lr_schedule()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(params)
    
    # Create training step function
    train_step_fn = create_train_step(optimizer)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_params = params
    patience_counter = 0
    patience = 20  # Early stopping patience (in terms of evaluation intervals)
    
    print(f"Starting training for {Config.epochs} epochs...")
    for epoch in range(Config.epochs):
        # Create shuffled batches for this epoch
        key, subkey = jax.random.split(key)
        batches = create_data_batch_iterator(x_train, y_train, Config.batch_size, subkey)
        
        # Process each batch
        epoch_losses = []
        for batch_x, batch_y in batches:
            params, opt_state, loss = train_step_fn(params, opt_state, batch_x, batch_y)
            epoch_losses.append(loss)
            
        # Calculate average loss for this epoch
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set periodically
        if epoch % Config.eval_every == 0 or epoch == Config.epochs - 1:
            val_loss = evaluate(params, x_val, y_val)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = jax.tree_map(lambda p: p, params)  # Copy the params
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Generate the final plot
    plot = plot_predictions(best_params, x_train, y_train, x_val, y_val)
    plot.show()
    
    return best_params, (train_losses, val_losses)


if __name__ == "__main__":
    print(f"Using JAX version: {jax.__version__}")
    print(f"Device: {jax.devices()[0]}")
    
    # Train the model
    params, (train_losses, val_losses) = train()
    
    # Plot learning curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(np.arange(0, len(val_losses) * Config.eval_every, Config.eval_every), 
             val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
