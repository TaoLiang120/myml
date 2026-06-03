import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

float_format = '%.8f'


def f1(x, a0, a1, a2):
    y = a0 + a1 * (1 - np.power(a2 * x, 2.0 / 3.0))
    return y


def f2(x, a2, a1, a0):
    y = a2 * np.exp(-a1 * x) + a0
    return y


def plot_shap(x, y, xseps=[0.25, 0.4]):
    fontsize = 14
    p0 = np.array([2096.49779804, 9.67202147, 100.0])
    colors = ["b", "r"]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x, y, s=100, marker="o", c="navy")
    ymin = np.min(y)
    ybase = -100
    for i in range(len(xseps)):
        xsep = xseps[i]
        inds = np.arange(len(x), dtype=int)
        if i == 0:
            inds = np.compress(x < xsep, inds)
        else:
            inds = np.compress(x > xsep, inds)
        thisx = x[inds]
        thisy = y[inds] - ybase
        if i == 1:
            thisx -= 0.25
        if i == 0:
            popt, pcov = curve_fit(eval('f2'), thisx, thisy, p0=p0)
            xnew = np.linspace(np.min(thisx), np.max(thisx) + 0.3, 7)
            ynew = eval("f2(xnew,*popt)")
            ax.plot(xnew, ynew + ybase, color=colors[i], linestyle='dashed', linewidth=1.0)
        else:
            popt, pcov = curve_fit(eval('f2'), thisx, thisy, p0=p0)
            xnew = np.linspace(np.min(thisx) - 0.06, np.max(thisx), 7)
            ynew = eval("f2(xnew,*popt)")
            ax.plot(xnew + 0.25, ynew + ybase, color=colors[i], linestyle='dashed', linewidth=1.0)
        print(popt)
        print("---")

    ax.set_ylim([np.min(y) - 40, np.max(y) + 40])
    ax.set_yticks([-750, -500, -250, 0, 250, 500])
    ax.set_xlim([np.min(x) - 0.1, np.max(x) + 0.2])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    plt.setp(ax.get_xticklabels(), fontsize=fontsize + 2)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize + 2)

    return fig


def get_xy_PDP(df, xkey):
    ykey = "DY_D" + xkey
    x = df[xkey].to_numpy()
    y = df[ykey].to_numpy()

    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    return x, y


def get_xy_SHAP(df, xkey, df_ref):
    ykey = "DY_D" + xkey
    xorg = df[xkey].to_numpy()
    yorg = df[ykey].to_numpy()

    inds = np.argsort(xorg)
    xorg = xorg[inds]
    yorg = yorg[inds]

    xkeyref = "T_ratio"
    xref = df_ref[xkeyref].to_numpy()
    xref = np.sort(xref)
    xref = np.append([0.0], xref)
    yref = np.zeros(len(xref), dtype=float)
    for i in range(1, len(xref)):
        thisxmax = xref[i]
        thisxmin = xref[i - 1]
        thiscount = 0
        for j in range(len(xorg)):
            if xorg[j] < thisxmax and xorg[i] >= thisxmin:
                yref[i] += yorg[j]
                thiscount += 1
        if thiscount < 2:
            x = np.compress(xorg < thisxmax, xorg)
            ind = len(x)
            istart = ind - 1
            iend = ind + 1
            if istart < 0: istart = 0
            if iend >= len(xorg): iend = len(xorg)
            thiscount = 0
            for j in range(istart, iend):
                yref[i] += yorg[j]
                thiscount += 1
        yref[i] /= thiscount
    x = xref[1: len(xref)]
    y = yref[1: len(yref)]
    return x, y


def plot_Yield_PDP(outfile="PDP_temp.png", savefig=False):
    fname = "Yield_EXP_reg_PDP.csv"
    df = pd.read_csv(fname)
    cols = df.columns.to_numpy()
    xkey = "T_ratio"
    x, y = get_xy_PDP(df, xkey)
    fig = plot_shap(x, y)
    if savefig:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
    else:
        plt.close()


def plot_Yield_SHAP(outfile="SHAP_temp.png", savefig=False):
    fname = "Yield_EXP_reg_SHAP.csv"
    df = pd.read_csv(fname)
    fname = "Yield_EXP_reg_PDP.csv"
    df_ref = pd.read_csv(fname)
    xkey = "T_ratio"
    x, y = get_xy_SHAP(df, xkey, df_ref)
    plot_shap(x, y)
    if savefig:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
    else:
        plt.close()
