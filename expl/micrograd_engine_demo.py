from micrograd.engine import Value

# Create input values
x = Value(2.0)
y = Value(3.0)

# Forward pass: simple function
# f = x * y + x
f = x * y + x

print("Forward output:", f.data)

# Backward pass
f.backward()

print("df/dx =", x.grad)
print("df/dy =", y.grad)
