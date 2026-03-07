from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.methods.safeguarded_newton import SafeguardedNewtonSolver


def linspace(a,b,n):
    step=(b-a)/(n-1)
    return [a+i*step for i in range(n)]


def empirical_ccdf(values):
    values=sorted(values)
    n=len(values)
    ks=np.arange(0,max(values)+1)

    ys=[]
    for k in ks:
        ys.append(sum(v>=k for v in values)/n)

    return ks,np.array(ys)


def collect_newton(f,df,xs,tol,max_iter):
    vals=[]

    for x0 in xs:

        solver=NewtonSolver(
            f=f,
            df=df,
            x0=x0,
            tol=tol,
            max_iter=max_iter
        )

        r=solver.solve()

        if r.status=="converged":
            vals.append(r.iterations)
        else:
            vals.append(max_iter)

    return vals


def collect_safe(f,df,a,b,xs,tol,max_iter):

    vals=[]

    for x0 in xs:

        solver=SafeguardedNewtonSolver(
            f=f,
            df=df,
            a=a,
            b=b,
            x0=x0,
            tol=tol,
            max_iter=max_iter
        )

        r=solver.solve()

        vals.append(r.iterations)

    return vals


def estimate_tail_slope(ks,ys):

    mask=(ys>0)

    ks=ks[mask]
    ys=ys[mask]

    logx=np.log(ks+1)
    logy=np.log(ys)

    coeff=np.polyfit(logx,logy,1)

    return coeff[0]


def plot_loglog(path,ks,ys,title):

    plt.figure(figsize=(7,5))

    plt.loglog(ks,ys)

    plt.xlabel("k")

    plt.ylabel("P(K ≥ k)")

    plt.title(title)

    plt.tight_layout()

    path.parent.mkdir(parents=True,exist_ok=True)

    plt.savefig(path,dpi=180)

    plt.close()


def main():

    f=lambda x:(x-1)**2*(x+2)
    df=lambda x:2*(x-1)*(x+2)+(x-1)**2

    a=-3
    b=-1

    xs=linspace(-4,4,1000)

    newton=collect_newton(f,df,xs,1e-10,100)
    safe=collect_safe(f,df,a,b,xs,1e-10,100)

    ks_n,ys_n=empirical_ccdf(newton)
    ks_s,ys_s=empirical_ccdf(safe)

    slope_newton=estimate_tail_slope(ks_n,ys_n)
    slope_safe=estimate_tail_slope(ks_s,ys_s)

    print("\nTail slope estimates")
    print("Newton:",slope_newton)
    print("Safeguarded Newton:",slope_safe)

    plot_loglog(
        Path("outputs/tail/newton_tail.png"),
        ks_n,
        ys_n,
        "Newton tail (log-log)"
    )

    plot_loglog(
        Path("outputs/tail/safe_tail.png"),
        ks_s,
        ys_s,
        "Safeguarded Newton tail (log-log)"
    )


if __name__=="__main__":
    main()