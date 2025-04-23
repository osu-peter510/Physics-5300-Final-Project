# Welcome to Physics 5300 Final Project for Peter!

## Introduction

### 1. Physics Model Setup

We study heat conduction on a one-dimensional rod of length \(L\), governed by the diffusion equation
\[
\frac{\partial u}{\partial t}
=\alpha\,\frac{\partial^2 u}{\partial x^2},
\quad
x\in[0,L],\;t\in[0,T],
\]
where \(u(x,t)\) is the temperature and \(\alpha\) is the thermal diffusivity.

**Dirichlet Boundary Conditions**  
The temperature is held fixed at both ends of the rod for all time:
\[
u(0,t)=0,
\quad
u(L,t)=0,
\quad
0\le t\le T.
\]

**Initial Condition**  
At \(t=0\), we impose a nontrivial temperature profile that combines a narrow Gaussian pulse centered at the midpoint \(x=L/2\) with a low-frequency sinusoid:
\[
u(x,0)
= \exp\bigl(-100\,(x - \tfrac{L}{2})^2\bigr)
\;+\;\sin\!\bigl(\tfrac{x}{5}\bigr),
\quad
0\le x\le L.
\]

---

### 2. Data from Finite-Difference Solver

We discretize space into \(N_x\) points \(x_i=i\,\Delta x\) and time into \(N_t\) steps \(t_k=k\,\Delta t\).  The explicit finite-difference update reads
\[
u_i^{\,k+1}
= u_i^{\,k}
\;+\;
\frac{\alpha\,\Delta t}{\Delta x^2}
\Bigl(
u_{i+1}^{\,k}
-2\,u_i^{\,k}
+u_{i-1}^{\,k}
\Bigr),
\]
where each new temperature \(u_i^{k+1}\) is computed from its two neighbors at the same time \(k\).  Stability of this scheme requires the CFL condition
\[
\Delta t \;\le\;\frac{\Delta x^2}{2\,\alpha}.
\]
Running this solver over all \(i,k\) produces the full reference solution array
\(\{U_{\rm FD}[i,k]\}\approx u(x_i,t_k)\), which we use both for visualization and as “ground truth” in training.

---

### 3. PINN Model Setup

We approximate the map \((x,t)\mapsto u(x,t)\) with a fully-connected neural network  
\(\widehat u(x,t;\theta)\colon\mathbb R^2\to\mathbb R\).  The network takes \([x,t]\) as input and outputs a temperature prediction.

The training loss combines two terms:
1. **Data loss** on boundary/initial points,
2. **PDE residual loss** enforcing the heat equation via automatic differentiation:

\[
\mathcal{L}(\theta)
=
\frac{1}{N_{\rm data}}
\sum_{(x_i,t_i)\in\rm BC/IC}
\bigl|\widehat u(x_i,t_i)-U_{\rm FD}(x_i,t_i)\bigr|^2
\;+\;
\frac{1}{N_f}
\sum_{j=1}^{N_f}
\bigl|\widehat u_t(x_f^j,t_f^j)
-\alpha\,\widehat u_{xx}(x_f^j,t_f^j)\bigr|^2.
\]

---

### 4. Training & Testing Sets

1. **Boundary/Initial Conditions**  
   - Initial: \(\{(x_i,0,U_{\rm FD}[i,0])\}\) for all spatial nodes \(i\).  
   - Boundaries: \(\{(0,t_k,0)\}\) and \(\{(L,t_k,0)\}\) for all time steps \(k\).  
   These are assembled into
   \[
     X_{\rm nu}\in\mathbb R^{N_{\rm nu}\times2},
     \quad
     Y_{\rm nu}\in\mathbb R^{N_{\rm nu}\times1}.
   \]
2. **Collocation Points**  
   Sample \(N_f\) interior \((x_f^j,t_f^j)\) (e.g.\ Latin-Hypercube) to enforce the PDE residual.  
   Form \(X_f\in\mathbb R^{N_f\times2}\).
3. **Train/Test Split**  
   - **Training set**: all BC/IC points and all collocation points.  
   - **Testing set**: a held-out subset of FD solution points to compute test MSE:
     \(\displaystyle
       \frac1{N_{\rm test}}\sum\bigl|\widehat u - U_{\rm FD}\bigr|^2.
     \)

This framework ensures the network learns both the prescribed data and the underlying physics of the heat equation.  
