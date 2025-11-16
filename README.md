

# 🌟 **Micrograd**

### *A Tiny Autograd Engine That Teaches You How Deep Learning REALLY Works*

✨ *Clean • Elegant • Educational • Minimal • Powerful*

---

<div align="center">



**A heartfelt tribute to Karpathy’s tiny autograd engine.
Rebuilt. Explained. Beautified.**

</div>

---

# 📘 **Table of Contents**

1. 🌱 Introduction
2. ⚡ Why Micrograd Exists
3. 🧠 How Micrograd Works
4. 🧩 Building Blocks (Value Class Explained)
5. 🔙 Backpropagation – Simple Explanation
6. 🏗️ Building & Training a Small MLP
7. 🔄 Fan-Out, Accumulation & Advanced Backprop Concepts
8. 🆚 Micrograd vs PyTorch
9. 📊 Demo Code
10. 🏁 Final Summary

---

# 🌱 **1. Introduction**

**Micrograd** is a *tiny automatic differentiation engine* built by **Andrej Karpathy**.
It is only **~100 lines of code**, yet it teaches:

* ✔ what is a computation graph
* ✔ how forward pass builds the graph
* ✔ how backward pass walks through it
* ✔ how gradients flow
* ✔ how neural nets learn—*from scratch*

This repo gives:

📌 **Ultra clean implementation**
📌 **Beginner-friendly commentary**
📌 **MLP training using Micrograd**
📌 **Educational visuals + explanations**

---

# ⚡ **2. Why Micrograd Exists?**

Deep learning libraries like **PyTorch** do:

```python
loss.backward()
```

Magically gradients appear.

But how?

Micrograd shows:

* No magic
* No complex tensors
* No abstractions

Just:

* **a graph**
* **nodes**
* **chain rule**
* **reverse traversal**

This is the *absolute core* of deep learning.

---

# 🧠 **3. How Micrograd Works (One Sentence)**

**Micrograd builds a graph of `Value` nodes during the forward pass
and computes gradients by walking backward through that graph.**

Simple. Transparent. Beautiful.

---

# 🧩 **4. Value Class — The Brain of Micrograd**

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

# 🔙 **5. Backpropagation — Simple Explanation**

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

# 🏗️ **6. Building & Training an MLP in Micrograd**

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

# 🔄 **7. Advanced Concepts (Made Easy)**

### 🔹 **Fan-Out**

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

# 🆚 **8. Micrograd vs PyTorch**

| Feature                     | Micrograd           | PyTorch                  |
| --------------------------- | ------------------- | ------------------------ |
| Purpose                     | Teaching            | Production Deep Learning |
| Speed                       | Slow                | Extremely Fast (GPU/TPU) |
| Supports Tensors?           | ❌ No, only scalars  | ✔ Yes                    |
| Builds Graph Automatically? | ✔ Yes               | ✔ Yes                    |
| Backprop?                   | ✔ Manual chain rule | ✔ Highly optimized       |
| Best Use                    | Learning internals  | Real-world models        |

---

# 📊 **9. Full Working Demo Code**

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

# 🏁 **10. Final Summary**

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

### ❤️ If this helped, ⭐ the repo!

