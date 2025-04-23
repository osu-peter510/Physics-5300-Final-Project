# Welcome to Physics 5300 Final Project for Peter!

# Introduction

## 1. Physics Model Setup

We consider heat conduction in a two-dimensional isotropic solid occupying a square domain  
\(\Omega = [0,L]\times[0,L]\).  The temperature field \(u(x,y,t)\) satisfies the heat equation

\[
\frac{\partial u}{\partial t}
\;=\;
\alpha\!\bigl(u_{xx} + u_{yy}\bigr),
\quad
(x,y)\in\Omega,\;t\in[0,T],
\]

where  
- \(\alpha\) is the (constant) thermal diffusivity,  
- subscripts \(x\), \(y\) denote second partial derivatives in space.

**Dirichlet Boundary Conditions**  
\[
u(0,y,t) = 0,\quad
u(L,y,t) = 0,\quad
u(x,0,t) = 0,\quad
u(x,L,t) = 100,
\]
for all \(0\le x,y\le L\) and \(0\le t\le T\).

**Initial Condition**  
At \(t=0\), the temperature is zero everywhere except in a central square “hot spot” of value 100:
```python
def u_center(u):
    mid = u.shape[0] // 2
    span = round(u.shape[0] * 0.2)
    u[mid-span:mid+span, mid-span:mid+span] = 100

u0 = np.zeros((N_x, N_x), dtype=np.float32)
u_center(u0)
solver.set_initial(u0)
