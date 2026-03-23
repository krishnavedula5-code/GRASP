# GRASP — Global Reliability Analysis & Solver Profiling

**GRASP (Global Reliability Analysis & Solver Profiling)** is a research-grade framework for analyzing the **global behavior of numerical root-finding algorithms**.

> Basin-of-attraction analysis | Solver benchmarking | Convergence diagnostics | Validation

---

## 🧠 What is GRASP?

GRASP is a **numerical experimentation and validation framework** designed to study solver behavior across large sets of initial conditions.

Unlike traditional tools that evaluate solvers from a single starting point, GRASP analyzes:

- Global convergence behavior  
- Basin-of-attraction structure  
- Statistical reliability (Monte Carlo)  
- Failure regions and instability patterns  
- Expected vs observed solver behavior  
- Automated validation of results  

GRASP is not just a solver.

It is a system for understanding:

> **Which solver to trust — and why**

---

## 🚀 Key Capabilities

- 🔍 **Global Analysis** across thousands of initial conditions  
- 📊 **Monte Carlo reliability estimation**  
- ⚙️ **Multiple solver support** (Newton, Secant, Bisection, Brent, Hybrid, Safeguarded Newton)  
- 📈 **Statistical metrics**
  - Success / failure probability  
  - Confidence intervals  
  - Mean & median iterations  
  - Root coverage  
- 🧠 **Automated interpretation**
  - Key observations  
  - Solver recommendations  
- ✅ **Validation layer (core contribution)**
  - Checks consistency between expected and observed behavior  

---

## 🌐 Live Demo

Try GRASP interactively:

👉 https://root-finding-reliability-framework.vercel.app

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/krishnavedula5-code/GRASP.git
cd GRASP
