from micrograd.nn import MLP
from micrograd.engine import Value

# Create model: 2 → [4] → 1
model = MLP(2, [4, 1])

# XOR dataset
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [0.0, 1.0, 1.0, 0.0]

# Convert ys to Value objects
ys = [Value(y) for y in ys]

# Training loop
for epoch in range(50):

    # Forward pass: model output for each input
    y_pred = [model(x) for x in xs]   # each result is a Value

    # Mean squared error loss
    loss = sum((yp - yt) * (yp - yt) for yp, yt in zip(y_pred, ys))

    # Zero gradients
    for p in model.parameters():
        p.grad = 0.0

    # Backpropagation
    loss.backward()

    # SGD update
    lr = 0.1
    for p in model.parameters():
        p.data -= lr * p.grad

    print(f"Epoch {epoch}: Loss = {loss.data}")
