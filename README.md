# 🌟 MICROGRAD  


---

# 🚀 What is Micrograd? 

Micrograd is a **tiny automatic differentiation engine** created by Andrej Karpathy.  
It helps you compute gradients for any mathematical expression by:

1. **Building a computation graph during the forward pass**
2. **Flowing gradients backward through the graph using chain rule**
3. **Accumulating gradients on every node**
4. **Allowing you to optimize neural network parameters**

Think of Micrograd as:

🧮 A calculator that not only computes your answer…  
…but also tells you **how the answer changes if every input is nudged slightly** — automatically.

Or in simpler terms:

> “Micrograd lets you write a math expression normally…  
> and magically gives you all the derivatives needed for training neural networks.”

---

# 🌱Why Micrograd Exists

Modern deep learning frameworks like **PyTorch** and **TensorFlow** have powerful autograd engines.  
Micrograd is a **minimal** version of that engine.

- No GPU  
- No tensors  
- No layers  
- No optimizers  
- Just **values** and **gradients**.

It teaches you:

✔ how computation graphs are built  
✔ how gradients flow backward  
✔ how chain rule works in real code  
✔ how neural networks learn under the hood  

---

# 🧠 How Micrograd Works (In One Sentence)
> Micrograd stores every operation between numbers as a node in a graph,  
> then applies the **chain rule** from the output backward to compute gradients.

---

# 📘 This Covers

1. Simple derivative (f(x)=x²)  
2. Numerical derivative check  
3. Multiple derivatives (f(x,y)=x·y)  
4. Full manual backpropagation example (your entire step-by-step story)  
5. Micrograd’s backward() system  
6. Gradient accumulation  
7. Difference between PyTorch & Micrograd  
8. Full annotated Micrograd code  

---

This  is built for **students**, **beginners**, **developers**, and **anyone trying to understand autograd**.

# 📌 1. Single Derivative — f(x) = x²

Let’s start with the simplest function:

\[
f(x) = x^2
\]

The derivative is:

\[
f'(x) = 2x
\]

Now let's evaluate both at \( x = 3 \):

```
f(3)  = 3² = 9  
df(3) = 2×3 = 6
```

So:

- The function value is **9**
- The slope at that point is **6**

This means:

> “If you nudge x a tiny bit, the output changes 6× that tiny amount.”

That's what gradient means.

---

# 📌 2. Numerical Derivative (Finite Difference Method)

Now we verify the derivative numerically.

We use a very tiny number **ε (epsilon)** and compute:

\[
\frac{f(x+\varepsilon) - f(x)}{\varepsilon}
\]

This should be close to the real derivative.

---

### ✔ Python-style demonstration

```python
def f(x):
    return x*x

x = 3.0
eps = 1e-6

# analytical derivative
df_analytic = 2*x

# numerical derivative
df_numerical = (f(x+eps) - f(x)) / eps

print("f(3) =", f(x))
print("Analytical df =", df_analytic)
print("Numerical df =", df_numerical)
```

### ✔ Output (Example)

```
f(3) = 9
Analytical df = 6
Numerical df ≈ 5.99999999976
```

---

# 📌 3. Interpretation

Point to the output and say:

> “See this? The **analytical derivative (6)** and the **numerical derivative (≈6)** match.  
> Micrograd automates this for *every node* in a computation graph.”

This is the core idea behind **automatic differentiation**.

---

# 📌 4. Why Numerical Derivative First?

Before understanding backpropagation, we must understand:

- A function
- Its slope
- How to approximate slope numerically
- How analytic slope and numerical slope match

Micrograd does NOT use numerical derivatives — that would be extremely slow.

It uses **symbolic chain rule** across a graph.  
But the *idea* is exactly the same as what we just computed.

---

# 📌 5. Summary

| Concept | Meaning |
|--------|---------|
| f(x) | Function value |
| f′(x) | Slope at x |
| Numerical derivative | Verify the slope with tiny ε |
| Matching values | Shows correctness |
| Micrograd | Automates all of this for giant networks |


---

# 🧠 6. How Micrograd Works (One Sentence)

**Micrograd builds a graph of `Value` nodes during the forward pass
and computes gradients by walking backward through that graph.**

Simple. Transparent. Beautiful.

---

# 🧩 7. Value Class — The Brain of Micrograd

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

### 🔍 What it does:

* Stores a **number**
* Tracks its **gradient**
* Remembers **which nodes created it**
* Stores **the operation** (+, *, tanh…)
* Holds a custom **backward function**

This is exactly how PyTorch tensors work — but simplified.

---

# 🔙 8. Backpropagation — Simple Explanation

Backprop = “How does changing this input change the final output?”

### ✔ Step 1: Forward Pass

Builds the graph by performing operations:

```
a → b → c → ... → loss
```

### ✔ Step 2: Set Final Gradient

```
loss.grad = 1
```

### ✔ Step 3: Walk Backward

Use chain rule:

```
parent.grad += child.grad * derivative
```

### ✔ Step 4: Repeat Until All Nodes Updated

This is the heart of deep learning.

---

# 🏗️ 9. Building & Training an MLP in Micrograd

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh()
```

Stack neurons → layer
Stack layers → MLP
Forward pass → output
Backward pass → gradients
Update weights → learning

This is literally how PyTorch works internally.

---

# 🔄 9. Advanced Concepts (Made Easy)

### 🔹 Fan-Out

When a value is used multiple times, its gradient appears multiple times.

### 🔹 **Gradient Accumulation**

```
v.grad += incoming_grad
```

NOT replace — **add**.

### 🔹 **New Operations**

Micrograd easily extends to:

* tanh
* exp
* power
* relu
* sigmoid

Just define the forward + backward rule.

---

# 🆚 **10. Micrograd vs PyTorch**

| Feature                     | Micrograd           | PyTorch                  |
| --------------------------- | ------------------- | ------------------------ |
| Purpose                     | Teaching            | Production Deep Learning |
| Speed                       | Slow                | Extremely Fast (GPU/TPU) |
| Supports Tensors?           | ❌ No, only scalars  | ✔ Yes                    |
| Builds Graph Automatically? | ✔ Yes               | ✔ Yes                    |
| Backprop?                   | ✔ Manual chain rule | ✔ Highly optimized       |
| Best Use                    | Learning internals  | Real-world models        |

---

# 📊 **11. Full Working Demo Code**

```python
from micrograd.engine import Value

# tiny dataset
xs = [
    [Value(2.0), Value(3.0)],
    [Value(1.0), Value(-1.0)],
]

ys = [Value(1.0), Value(-1.0)]

# simple neuron
n = Neuron(2)

for epoch in range(20):
    ypred = [n(x) for x in xs]
    loss = sum((yout - yt)**2 for yout, yt in zip(ypred, ys))

    # backward
    for p in n.parameters(): p.grad = 0
    loss.backward()

    # update
    for p in n.parameters():
        p.data -= 0.1 * p.grad

    print(epoch, loss.data)
```

---

# 🏁 12. Final Summary

Micrograd teaches you:

✔ how neural nets work
✔ how gradients flow
✔ how autograd engines function
✔ how forward & backward graph traversal works
✔ how to build models **from scratch**

It is the **cleanest**,
**purest**,
**most elegant**
deep learning educational tool ever created.

---



