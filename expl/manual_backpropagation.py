# Simple manual backprop example
# We will compute:
# a = x * y
# b = a + z
# L = b

# Step 1: Forward pass
x = 2.0
y = -3.0
z = 10.0

a = x * y       # a = -6
b = a + z       # b = 4
L = b           # L = 4

print("Forward Pass:")
print("a =", a)
print("b =", b)
print("L =", L)

# Step 2: Backward pass (manual gradients)
dL_dL = 1.0     # final output gradient

# For b = a + z → da = 1 * dL, dz = 1 * dL
dL_db = dL_dL
dL_da = 1.0 * dL_db
dL_dz = 1.0 * dL_db

# For a = x * y → dx = y * dL, dy = x * dL
dL_dx = y * dL_da
dL_dy = x * dL_da

print("\nBackward Pass (Gradients):")
print("dL/dx =", dL_dx)
print("dL/dy =", dL_dy)
print("dL/dz =", dL_dz)
