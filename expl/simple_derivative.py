import math

def f(x):
    return x * x        # f(x) = x^2

def df(x):              # derivative = 2x
    return 2 * x

x = 3.0
print("f(x) =", f(x))
print("f'(x) =", df(x))

# Let's test how f changes with a small step
epsilon = 1e-5
numerical_derivative = (f(x + epsilon) - f(x)) / epsilon

print("numerical derivative approx =", numerical_derivative)
