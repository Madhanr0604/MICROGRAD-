def f(x, y):
    return x * y       # f(x, y) = x*y

def dfdx(x, y):
    return y           # ∂f/∂x = y

def dfdy(x, y):
    return x           # ∂f/∂y = x

x = 3.0
y = 4.0

print("f(x, y) =", f(x, y))
print("df/dx =", dfdx(x, y))
print("df/dy =", dfdy(x, y))

# Numerical check
epsilon = 1e-5

num_dfdx = (f(x + epsilon, y) - f(x, y)) / epsilon
num_dfdy = (f(x, y + epsilon) - f(x, y)) / epsilon

print("numerical df/dx =", num_dfdx)
print("numerical df/dy =", num_dfdy)
