import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from myml.data.data import myData, DATA_PATH
from myml.plot.plot import plot_linear_xys, get_3d_bar

Element_list = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]

XKEYs = ["Hmix_TC", "Gmix_TC", "E_negativity", "Exp_Bulk_ROM", "Exp_Youngs_ROM", "Exp_Shear_ROM",
         "volume_DELTA", "volume_DISTORT", "bcc2hcp", "ElasticEnergy", "bcc2hcp_Gmix_TC2CPA", "bcc2hcp_Gmix_TC",
         "Exp_Shear_DISTORT",  "Exp_Surf_ROM",
         "E_GBR_ROM", "DOS_EF_ROM", "E_USF_ROM",
         "ShearBurger", "ShearBurger_Surf", "USF_Surf",
         "volume_DISTORT_Max", "Exp_Poisson_Comp",
         "Exp_Poisson_ROM", "SH_ratio", "sigma_y0", "delta_Eb0", "density", "VEC"]

fname_db = "bccCall_HF_GBR.csv"
thisdb = myData(os.path.join(DATA_PATH, fname_db))

rename_dict = {"Hmix_TC": "Hmix", "Gmix_TC": "Gmix", "Exp_Bulk_ROM": "Bulk_ROM", "Exp_Youngs_ROM": "Youngs_ROM",
               "Exp_Shear_ROM": "Shear_ROM",
               "volume_DELTA": "volume_DELTA", "Exp_Shear_ROM_DELTA": "Shear_DELTA", "Exp_Bulk_ROM_DELTA": "Bulk_DELTA",
               "Exp_Youngs_ROM_DELTA": "Youngs_DELTA",
               "radius_DISTORT": "radius_DISTORT", "Exp_Shear_DISTORT": "Shear_DISTORT", "ElasticEnergy": "H_elastic",
               "Exp_Youngs_DISTORT": "Youngs_DISTORT", "Exp_Poisson_ROM": "Poisson_ROM", "Yield_MC": "YS_MC",
               "Gmix_SCALED": "G_mix", "Sconf": "S_mix",
               "bcc2hcp": "b2h_CPA", "bcc2hcp_Gmix_TC": "b2h_TC", "bcc2hcp_Gmix_TC2CPA": "b2h_TC2CPA"}


def get_this_plot(thispf, xs, ys, xkeys, ykeys, style, colormaps, xaxis2count, isROM,
                  label_abnormal, label_axis, savefig, outfile, **plot_setts):
    fig, = thispf.plot2d(xs, ys, xkeys, ykeys, style=style, colormaps=colormaps,
                         xaxis2count=xaxis2count, isROM=isROM,
                         label_abnormal=label_abnormal, label_axis=label_axis, savefig=savefig, outfile=outfile,
                         **plot_setts)

    return fig


def get_linear_plot(xs, ys, plot_xx=False, compute_R2=True, Label_R2=False, curvefit=True, Comp_Labels=None,
                    Label_Comp=False):
    fig, thisdf = plot_linear_xys(xs, ys, plot_xx=plot_xx, compute_R2=compute_R2,
                                  Label_R2=Label_R2, curvefit=curvefit, Comp_Labels=Comp_Labels, Label_Comp=Label_Comp)
    return fig, thisdf


def get_3d_bar(xs, ys, zs, yticks=["AMO", "FCC", "MIX", "BCC"], isstr=False):
    yticks = list(yticks)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = ['r', 'g', 'b', 'y']
    markers = ['o', '^', '*', 'd']

    xss = []
    zss = []
    xlabels = []
    for i in range(len(yticks)):
        ykey = yticks[i]
        inds = np.arange(len(xs), dtype=int)
        inds = np.compress(ys == ykey, inds)
        thisx = xs[inds]
        thisz = zs[inds]
        xlabels.append(thisx)
        if isstr:
            thisx = np.arange(len(xlabels), dtype=int)
        print(f"i:{i} ykey:{ykey} length:{len(thisz)}")
        xss.append(thisx)
        zss.append(thisz)

    for c, k in zip(colors, yticks):
        ik = yticks.index(k)
        thisx = xss[ik]
        thisz = zss[ik]
        thisy = [ik] * len(thisx)
        thisxlabel = xlabels[ik]
        cs = [c] * len(thisx)
        m = markers[ik]

        ax.bar(thisxlabel, thisz, zs=ik, zdir='y', color=cs, alpha=0.8)

    if not isstr:
        xmax = np.max(xs)
        xmin = np.min(xs)
        if xmax - xmin > 10:
            ndec = 0
        elif xmax - xmin > 1:
            ndec = 1
        elif xmax - xmin > 0.1:
            ndec = 2
        else:
            ndec = 3
        xticks = np.linspace(xmin, xmax, 5)
        xticks = np.around(xticks, decimals=ndec)
        ax.set_xticks(xticks)
        plt.setp(ax.get_xticklabels(), fontsize=14)

    xmax = np.max(zs)
    xmin = np.min(zs)
    if xmax - xmin > 10:
        ndec = 0
    elif xmax - xmin > 1:
        ndec = 1
    elif xmax - xmin > 0.1:
        ndec = 2
    else:
        ndec = 3
    zticks = np.linspace(xmin, xmax, 5)
    zticks = np.around(zticks, decimals=ndec)
    ax.set_zticks(zticks)
    plt.setp(ax.get_zticklabels(), fontsize=14)
    #ax.set_yticklabels(yticks)
    plt.setp(ax.get_yticklabels(), visible=False)
    return fig


def Investig_thisDB(thisdata, ykey, style, Plot_style=None, thresc=0.6):
    xkeys = XKEYs[0:len(XKEYs)]
    valid_keys = xkeys[0:len(xkeys)]
    if style == 1:
        ykeys = ["Crystal"] * len(xkeys)
        zkeys = [ykey] * len(xkeys)
    else:
        ykeys = [ykey] * len(xkeys)
        zkeys = [ykey] * len(xkeys)

    inds = []
    istart = 0
    iend = len(xkeys)
    for ikey in range(istart, iend):
        xkey = xkeys[ikey]
        ykey = ykeys[ikey]
        zkey = zkeys[ikey]
        print(f"xkey:{xkey} ykey:{ykey} zkey:{zkey}")

        xs = thisdata.gooddf[xkey].to_numpy()
        ys = thisdata.gooddf[ykey].to_numpy()
        zs = thisdata.gooddf[zkey].to_numpy()
        thismean = np.mean(xs)
        thisstd = np.std(xs)
        thismin = thismean - thisstd
        thismax = thismean + thisstd

        xdb = thisdb.gooddf[xkey].to_numpy()
        dbmean = np.mean(xdb)
        dbstd = np.std(xdb)
        dbmin = dbmean - dbstd
        dbmax = dbmean + dbstd

        totmin = min(thismin, dbmin)
        totmax = max(thismax, dbmax)

        totr = totmax - totmin
        thisr = thismax - thismin
        dbr = dbmax - dbmin
        thisc = thisr / totr
        dbc = dbr / totr
        thisvar = thisstd / thismean
        dbvar = dbstd / dbmean
        ##############################
        thismin3 = np.min(xs)
        thismax3 = np.max(xs)

        dbmin3 = np.min(xdb)
        dbmax3 = np.max(xdb)

        totmin3 = min(thismin3, dbmin3)
        totmax3 = max(thismax3, dbmax3)

        totr3 = totmax3 - totmin3
        thisr3 = thismax3 - thismin3
        dbr3 = dbmax3 - dbmin3
        thisc3 = thisr3 / totr3
        dbc3 = dbr3 / totr3
        ##################################3
        thismin2 = thismean - 2 * thisstd
        thismax2 = thismean + 2 * thisstd

        dbmin2 = dbmean - 2 * dbstd
        dbmax2 = dbmean + 2 * dbstd

        totmin2 = min(thismin2, dbmin2)
        totmax2 = max(thismax2, dbmax2)

        totr2 = totmax2 - totmin2
        thisr2 = thismax2 - thismin2
        dbr2 = dbmax2 - dbmin2
        thisc2 = thisr2 / totr2
        dbc2 = dbr2 / totr2
        ############################
        if thisc2 >= thresc and dbc2 >= thresc:
            isValid = True
            inds.append(ikey)
        else:
            isValid = False

        print(f"min:{thismin} max:{thismax} thisr:{thisr}")
        print(f"dbmin:{dbmin} dbmax:{dbmax} dbr:{dbr}")
        print(f"totmin:{totmin} totmax:{totmax} totr:{totr}")
        print(f"min2:{thismin2} max2:{thismax2} thisr2:{thisr2}")
        print(f"dbmin2:{dbmin2} dbmax2:{dbmax2} dbr2:{dbr2}")
        print(f"totmin2:{totmin2} totmax2:{totmax2} totr2:{totr2}")
        print(f"min3:{thismin3} max3:{thismax3} thisr3:{thisr3}")
        print(f"dbmin3:{dbmin3} dbmax3:{dbmax3} dbr3:{dbr3}")
        print(f"totmin3:{totmin3} totmax3:{totmax3} totr3:{totr3}")
        print(f"thiscov1:{thisc} dbcover1:{dbc}")
        print(f"thiscov2:{thisc2} dbcover2:{dbc2}")
        print(f"thiscov3:{thisc3} dbcover3:{dbc3}")
        print(f"thisvariation:{thisvar} dbvariation:{dbvar}")
        print(f"xkey:{xkey} isValid:{isValid}")
        print("=====")

        Plot_linear = False
        Plot_3D = False
        if Plot_style is None:
            pass
        elif Plot_style == 0:
            Plot_linear = True
        else:
            if style == 0:
                Plot_linear = True
            else:
                Plot_3D = True

        if Plot_linear:
            fig, thisdf = get_linear_plot(zs, xs, plot_xx=False, compute_R2=False, Label_R2=False, curvefit=True)
            thisoutfig = "All_PNGs/" + xkey + "_" + "elongation.png"
            plt.savefig(thisoutfig, bbox_inches="tight")
            plt.close()
        elif Plot_3D:
            fig = get_3d_bar(xs, ys, zs)
            thisoutfig = "All_PNGs/" + xkey + "_" + "elongation_bar3D.png"
            plt.savefig(thisoutfig, bbox_inches="tight")
            plt.close()

    valid_keys = np.array(valid_keys)
    inds = np.array(inds)
    valid_keys = valid_keys[inds]
    return valid_keys


def bar_xy(xkeys, y, norm=1):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.barh(xkeys, y)
    if norm == 1:
        ax.set_xticks(np.linspace(np.min(y), np.max(y), 5))
    elif norm == 2:
        ax.set_xticks(np.linspace(0, 1.0, 5))

    labels = ax.get_xticklabels()
    #plt.setp(labels, rotation=45, horizontalalignment='right', fontsize = 16)
    plt.setp(labels, fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)

    #ax.set_xlabel("Absolute R2", fontsize=18)
    #ax.set_ylabel("Features", fontsize=18)
    return fig


def Linear_xkeys(thisdata, xkeys, ykey):
    ykeys = [ykey] * len(xkeys)
    outfig = ykey + ".png"

    cols = ["xkey", "p0_fit", "p1_fit", "r2_fit"]
    a = np.zeros([len(xkeys), len(cols)])
    mydf = pd.DataFrame(a, columns=cols)
    mydf["xkey"] = xkeys
    mydf = mydf.set_index("xkey")

    istart = 0
    iend = len(xkeys)
    for ikey in range(istart, iend):
        xkey = xkeys[ikey]
        ykey = ykeys[ikey]
        print(f"xkey:{xkey} ykey:{ykey}")

        xs = thisdata.gooddf[xkey].to_numpy()
        ys = thisdata.gooddf[ykey].to_numpy()
        compstrs = thisdata.gooddf["Composition"].to_numpy()

        fig, thisdf = get_linear_plot(xs, ys, plot_xx=False, compute_R2=True,
                                      Label_R2=False, curvefit=True, Comp_Labels=compstrs, Label_Comp=False)
        for key in cols[1:len(cols)]:
            mydf.loc[xkey, key] = thisdf.loc[0, key]
        plt.close()

    mydf.to_csv("features.csv", index=True)

    mydf = pd.read_csv("features.csv")

    ys = mydf["r2_fit"].to_numpy()
    mydf["abs_r2"] = np.absolute(ys)
    mydf = mydf.sort_values("abs_r2")

    xkeys = mydf["xkey"].to_numpy()
    for ikey in range(len(xkeys)):
        xkey = xkeys[ikey]
        try:
            xkeys[ikey] = rename_dict[xkey]
        except:
            pass

    y = mydf["abs_r2"].to_numpy()
    fig = bar_xy(xkeys, y)

    plt.savefig(outfig, bbox_inches='tight')
    plt.show()
