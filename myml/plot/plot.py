import copy
import itertools

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pylab import *
from matplotlib.font_manager import FontProperties
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.util.string import latexify
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from myml.myglobal import Element_negativity


def f1(x, a0, a1):
    y = a0 + a1 * x
    return y


def f2(x, a0, a1, a2):
    y = a0 + a1 * np.exp(-a2 * x)
    return y

def get_ticks_labels(ys, nyticks):
    vmax = np.max(ys)
    vmin = np.min(ys)
    vticks = np.linspace(vmin, vmax, nyticks)
    if vmax - vmin >= 10:
        ticklabels = [f'{int(round(v))}' for v in vticks]
    elif vmax - vmin >= 2:
        ticklabels = [f'{v:1.1f}' for v in vticks]
    else:
        ticklabels = [f'{v:1.2f}' for v in vticks]
    return vticks, ticklabels


def plot_linear_xys(x, ys, style="scatter", plot_xx=False,
                    Xlabel=None, Ylabel=None, Title=None, compute_R2=True, Label_R2=False,
                    curvefit=False, cmap="jet", colors=None, ShowColorbar=False, Comp_Labels=None, Label_Comp=False,
                    fontsize=16, nxticks=None, nyticks=None):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    def init_performance_df(n):
        keys = ["iplot", "mse", "rmse", "mae", "rmae", "r2", "p0_fit", "p1_fit", "r2_fit"]
        a = np.zeros([n, len(keys)])
        performance_df = pd.DataFrame(a, columns=keys)
        performance_df["iplot"] = np.arange(n, dtype=int)
        performance_df = performance_df.set_index("iplot")
        return performance_df

    def init_performance_dict():
        keys = ["mse", "rmse", "mae", "rmae", "r2", "p0_fit", "p1_fit", "r2_fit"]
        performance_dict = {}
        for key in keys:
            performance_dict[key] = 0.0
        return performance_dict

    def assign_dict2df(iplot, thisdict, performance_df):
        performance_df.loc[iplot] = thisdict
        return performance_df

    ShowColor = True
    if colors is None:
        colors = x[0:len(x)]
        ShowColor = False
    vmin = np.min(colors)
    vmax = np.max(colors)

    if len(ys.shape) == 1: ys = np.array([ys])
    nsubplot = len(ys)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})

    fig = plt.figure()
    axs = [None] * nsubplot
    font = FontProperties(family='times new roman', weight="bold", size=fontsize + 2)
    norm = Normalize(vmin=vmin, vmax=vmax)
    _map = ScalarMappable(norm=norm, cmap=cmap)
    cs = _map.to_rgba(colors)

    Label_compstr = False
    if Label_Comp:
        if isinstance(Comp_Labels, list) or isinstance(Comp_Labels, np.ndarray):
            if len(Comp_Labels) == len(x): Label_compstr = True

    performance_df = init_performance_df(nsubplot)
    xmean = np.mean(x)
    for iplot in range(nsubplot):
        y = ys[iplot, :]
        if iplot == 0:
            axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1)
        else:
            axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1, sharex=axs[0])

        if plot_xx:
            tmpx = np.linspace(np.min(x), np.max(x), 20)
            axs[iplot].plot(tmpx, tmpx, linewidth=1.5, color="k", linestyle="dotted", ms=0.0)
        if ShowColor:
            if style == "line":
                axs[iplot].plot(x, y, linewidth=1.5, color="k", linestyle="-", ms=20.0, mfc=cs, mew=0.0)
            else:
                axs[iplot].scatter(x, y, s=30, marker="o", c=cs)
        else:
            if style == "line":
                axs[iplot].plot(x, y, linewidth=1.5, color="k", linestyle="-", ms=20.0, mfc=cs, mew=0.0)
            else:
                axs[iplot].scatter(x, y, s=30, marker="o", c='k')

        if Label_compstr:
            for i in range(len(x)):
                axs[iplot].text(x[i], y[i], Comp_Labels[i], fontsize=fontsize-4)

        thisdict = init_performance_dict()
        R2 = 0.0
        if compute_R2:
            yr = y - x
            diff = x - xmean

            R2 = 1.0 - np.sum(yr * yr) / (np.sum(diff * diff) + 1.0e-20)
            if not curvefit and Label_R2:
                label = "R2 = " + str(round(R2, 4))
                labelx = np.min(x) + 0.6 * (np.max(x) - np.min(x))
                labely = np.min(y) + 0.4 * (np.max(y) - np.min(y))
                axs[iplot].text(labelx, labely, label, fontsize=fontsize)

        mse = mean_squared_error(x, y)
        mse = np.sqrt(mse)
        rmse = 100.0 * mse / xmean
        mae = mean_absolute_error(x, y)
        rmae = 100.0 * mae / xmean
        print(f"iplot: {iplot}  R2: {R2}")
        print(f"MSE on test set: {mse} Relative MSE: {rmse}")
        print(f"MAE on test set: {mae} Relative MAE: {rmae}")
        thisdict["iplot"] = iplot
        thisdict["mse"] = mse
        thisdict["rmse"] = rmse
        thisdict["mae"] = mae
        thisdict["rmae"] = rmae
        thisdict["r2"] = R2
        if curvefit:
            popt, pcov = curve_fit(eval('f1'), x, y)
            xnew = np.linspace(np.min(x), np.max(x), 20)
            ynew = eval("f1(xnew,*popt)")
            axs[iplot].plot(xnew, ynew, color="k", linestyle='dashed', linewidth=1.0)
            yr = eval("f1(x,*popt)")
            yr = y - yr
            ydiff = y - np.mean(y)
            r2 = 1.0 - np.sum(yr * yr) / np.sum(ydiff * ydiff)
            if Label_R2:
                label = "R2 = " + str(round(r2, 4))
                labelx = np.min(x) + 0.6 * (np.max(x) - np.min(x))
                labely = np.min(y) + 0.3 * (np.max(y) - np.min(y))
                axs[iplot].text(labelx, labely, label, fontsize=fontsize-2)
            print(f"iplot:{iplot} popt:{popt} R2:{r2}")
            thisdict["p0_fit"] = popt[0]
            thisdict["p1_fit"] = popt[1]
            thisdict["r2_fit"] = r2

        performance_df = assign_dict2df(iplot, thisdict, performance_df)
        if iplot == nsubplot - 1:
            if Xlabel is not None: axs[iplot].set_xlabel(Xlabel, fontsize=fontsize + 2)
            if isinstance(nxticks, int):
                xticks, xlabels = get_ticks_labels(x, nxticks)
                axs[iplot].set_xticks(xticks, labels=xlabels)

        if isinstance(nyticks, int):
            yticks, ylabels = get_ticks_labels(y, nyticks)
            axs[iplot].set_yticks(yticks, labels=ylabels)

        if Ylabel is not None:
            if isinstance(Ylabel, list):
                axs[iplot].set_ylabel(Ylabel[iplot], fontsize=fontsize + 2)
            else:
                axs[iplot].set_ylabel(Ylabel, fontsize=fontsize + 2)

    plt.setp(axs[iplot].get_yticklabels(), fontsize=fontsize)
    plt.setp(axs[iplot].get_xticklabels(), fontsize=fontsize)
    if Title is not None: plt.title(Title, fontsize=fontsize + 4)
    if ShowColor and ShowColorbar:
        font = FontProperties(family='times new roman', size=fontsize)
        _map.set_array(colors)
        cbar = plt.colorbar(_map, ax=plt.gca())
        cbar.set_label("", rotation=-90, ha="left", va="center", fontproperties=font)
        cbar.ax.tick_params(labelsize=fontsize)
    return fig, performance_df


def plot_xss_yss_lines(xss, yss, style="scatter", labels=None,
                       Label_compstr=False, Comp_Labels=None,
                       savefig=False, outfile="xss_yss.png"):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fontsize = 14
    markers = ["s", "o", "d", "x", "*", "<", ">"]
    colors = ["r", "b", "c", "g", "m", "y", "k"]
    nsubplot = 1
    fig = plt.figure()
    axs = [None] * nsubplot
    iplot = 0
    nline = len(xss)
    lines = [None] * nline
    for iline in range(nline):
        x = copy.deepcopy(xss[iline])
        y = copy.deepcopy(yss[iline])
        imark = iline % len(markers)
        thismarker = markers[imark]
        icolor = iline % len(colors)
        thiscolor = colors[icolor]
        axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1)
        if style == "scatter":
            lines[iline] = axs[iplot].scatter(x, y, s=30, marker=thismarker, c=thiscolor)
        else:
            lines[iline] = axs[iplot].plot(x, y, linewidth=1.5, color=thiscolor, linestyle="-", ms=20.0, mfc=thiscolor,
                                           mew=0.0)
        if Label_compstr:
            for i in range(len(x)):
                axs[iplot].text(x[i], y[i], Comp_Labels[iline][i], fontsize=fontsize - 4)

        plt.setp(axs[iplot].get_yticklabels(), fontsize=fontsize)
        plt.setp(axs[iplot].get_xticklabels(), fontsize=fontsize)

    if isinstance(xss, np.ndarray):
        ndata = xss.shape[1]
    else:
        ndata = 4

    if ndata > 1:
        ncol = 4
    else:
        ncol = 1

    if labels is None:
        labels = ["NA"] * len(lines)
    elif len(labels) != len(lines):
        labels = ["NA"] * len(lines)
    if style == "scatter":
        plt.legend(lines, labels, scatterpoints=1, ncol=ncol, fontsize=fontsize - 4)
    else:
        plt.legend(lines, labels, ncol=ncol, fontsize=fontsize - 4)
    if savefig:
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def bar_plot(xkeys, ys, horizontal=True, fontsize=14,
             figsize=(6, 6), nyticks=4, rotation=90,
             savefig=False, outfile=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(xkeys, ys)
        ax.set_xticks(np.linspace(round(np.min(ys), 3), round(np.max(ys), 3), nyticks))
        labels = ax.get_xticklabels()
        plt.setp(labels, fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    else:
        ax.bar(xkeys, ys)
        ax.set_yticks(np.linspace(round(np.min(ys), 3), round(np.max(ys), 3), nyticks))
        labels = ax.get_yticklabels()
        plt.setp(labels, fontsize=fontsize)
        labels = ax.get_xticklabels()
        plt.setp(labels, fontsize=fontsize - 2, rotation=rotation)

    if savefig:
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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


def quartiles_plot(df, xkey, ykey, xbins=None, nbin=9,
                   vert=True, showmeans=True, meanline=True, showfliers=False):
    fontsize = 16
    ys = df[ykey].to_numpy()
    xs = df[xkey].to_numpy()
    if xbins is None:
        xmin = np.min(xs)
        xmax = np.max(xs)
        xbins = np.arange(xmin, xmax, nbin + 1)
    else:
        nbin = len(xbins) + 1

    ymeans = []
    yboxes = []
    #lolims = []
    #uplims = []
    for i in range(1, len(xbins)):
        inds = []
        for ii in range(len(df)):
            x = xs[ii]
            if x >= xbins[i - 1] and x < xbins[i]:
                inds.append(ii)
        inds = np.array(inds).astype(int)
        thisys = ys[inds]
        yboxes.append(thisys)

        ymean = np.mean(thisys)
        ymeans.append(ymean)
        print(f"ibin:{i} xrange:{xbins[i - 1], xbins[i]} length y:{len(thisys)}")

    xbins = xbins[1:len(xbins)]
    ymeans = np.array(ymeans)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    font = FontProperties(family='times new roman', size=fontsize)
    plt.boxplot(
        yboxes,
        vert=vert,
        labels=xbins,
        showmeans=showmeans,
        meanline=meanline,
        showfliers=showfliers,
    )

    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    return fig


class CSDiagram:
    def __init__(self, entries, terminal_entries,
                 isElements=True, set_terminal2entry=True):
        self.original_entries = entries
        self.terminal_entries = terminal_entries
        self.isElements = isElements
        elements = []
        for entry in terminal_entries:
            elements.append(entry.composition.elements[0])
        self.elements = elements
        elements_equivalent = {}
        if not isElements:
            for i in range(len(elements)):
                key = self.elements[i].symbol
                e = self.terminal_entries[i]
                if len(e.composition.elements) > 1:
                    value = "("
                else:
                    value = ""
                for j in range(len(e.composition.elements)):
                    el = e.composition.elements[j]
                    value += el.symbol
                if len(e.composition.elements) > 1: value += ")"
                elements_equivalent[key] = value
        else:
            for i in range(len(elements)):
                key = self.elements[i].symbol
                elements_equivalent[key] = key
        self.elements_equivalent = elements_equivalent

        if isElements:
            self.applied_entries = entries
        else:
            pentries = self.transform_entries(entries, terminal_entries)
            self.applied_entries = pentries

        self.kwargs_from_entries(self.applied_entries, set_terminal2entry=set_terminal2entry)

    def transform_entries(self, entries, terminal_entries):
        tcomps = []
        for entry in terminal_entries:
            tcomps.append(entry.composition)

        new_entries = []
        for entry in entries:
            thiscomp = entry.composition
            compstr = ""
            for it in range(len(tcomps)):
                tc = tcomps[it]
                fc = 0.0
                for ie in range(len(tc.elements)):
                    el = tc.elements[ie]
                    if ie == 0: compstr += el.symbol
                    fc += thiscomp.get_atomic_fraction(el)
                fc = round(fc * 100.0, 0)
                compstr += str(fc)
            c = Composition(compstr)
            y = entry.energy
            new_entries.append(PDEntry(c, y))
        return new_entries

    def kwargs_from_entries(self, entries, set_terminal2entry=False):
        elements = list(self.elements)
        self.dim = len(self.elements)
        entries = sorted(entries, key=lambda e: e.composition.reduced_composition)

        data = np.array(
            [[e.composition.get_atomic_fraction(el) for el in elements] + [e.energy] for e in entries]
        )

        self.all_entries = entries
        self.entry_data = data[:, 1:]
        self.entry_data = np.array(self.entry_data)
        self.minimum = np.min(self.entry_data[:, -1])
        self.maximum = np.max(self.entry_data[:, -1])

        if set_terminal2entry:
            max_ticks = []
            min_ticks = []
            for i in range(self.dim):
                ind = np.argmax(data[:, i], axis=0)
                d = data[ind]
                max_ticks.append(d[i])
                ind = np.argmin(data[:, i], axis=0)
                d = data[ind]
                min_ticks.append(d[i])

            txs = [0] * self.dim
            tys = [0.0] * self.dim
            ns = [0] * self.dim
            tds = [None] * self.dim
            for i in range(len(data)):
                d = data[i]
                for j in range(self.dim):
                    if d[j] == max_ticks[j]:
                        e = entries[i]
                        txs[j] = e.name
                        tys[j] += e.energy
                        ns[j] += 1
                        if tds[j] is None:
                            tds[j] = list(d)
                        else:
                            tds[j][-1] += e.energy

            self.terminal_entries = []
            for i in range(self.dim):
                e = PDEntry(Composition(txs[i]), tys[i] / ns[i])
                comp = e.composition
                e.name = ""
                for el in comp.elements:
                    e.name += el.symbol
                    frac = comp.get_atomic_fraction(el)
                    frac = round(frac, 2)
                    e.name += str(frac)
                if not self.isElements:
                    for k, v in self.elements_equivalent.items():
                        e.name = e.name.replace(k, v)
                self.terminal_entries.append(e)
                tds[i][-1] = tds[i][-1] / ns[i]
            tds = np.array(tds)
            self.terminal_data = tds[:, 1:]
        else:
            if not self.isElements:
                t4data = []
                for i in range(len(self.terminal_entries)):
                    c = Composition(self.elements[i].symbol)
                    y = self.terminal_entries[i].energy
                    t4data.append(PDEntry(c, y))
            else:
                t4data = self.terminal_entries[0:self.dim]

            data = np.array(
                [[e.composition.get_atomic_fraction(el) for el in elements] + [e.energy] for e in t4data]
            )
            self.terminal_data = data[:, 1:]
            self.terminal_data = np.array(self.terminal_data)

            max_ticks = []
            min_ticks = []
            for i in range(self.dim):
                ind = np.argmax(data[:, i], axis=0)
                d = data[ind]
                max_ticks.append(d[i])
                ind = np.argmin(data[:, i], axis=0)
                d = data[ind]
                min_ticks.append(d[i])

        self.max_ticks = max_ticks
        self.min_ticks = min_ticks


class CSPlotter:
    def __init__(
            self,
            phasediagram: CSDiagram,
            backend: str = "matplotlib",
            minimum: float = None,
            maximum: float = None,
            **plotkwargs,
    ):
        self._pd = phasediagram
        self._dim = len(self._pd.elements)
        if self._dim > 4: raise ValueError("Only 1-4 components supported!")

        self.backend = backend
        if minimum is not None:
            self.minimum = minimum
        else:
            self.minimum = self._pd.minimum
        self.local_minimum = self._pd.minimum
        if maximum is not None:
            self.maximum = maximum
        else:
            self.maximum = self._pd.maximum
        self.local_maximum = self._pd.maximum

        self.elements_equivalent = self._pd.elements_equivalent
        default_settings = {"label_entry": False, "cmap": "jet", "show_colorbar": True,
                            "term_ms": 15, "ms": 30, "term_mfc": "auto", "mfc": "auto",
                            "mec": "k", "mew": 0.0, "alpha": 0.6,
                            "DataAugment": True, "gridsize": 0.005}

        for key in plotkwargs:
            if key in default_settings:
                default_settings[key] = plotkwargs[key]
        self.plotkwargs = default_settings

    @property  # type: ignore
    def pd_plot_data(self):
        def get_coords(e1, e2, d1, d2):
            if self._dim < 3:
                x = [d1[0], d2[0]]
                y = [e1.energy, e2.energy]
                coord = [x, y]
            elif self._dim == 3:
                vec = np.array([d1[0:2], d2[0:2]])
                coord = triangular_coord(vec)
            else:
                vec = np.array([d1[0:3], d2[0:3]])
                coord = tet_coord(vec)
            return coord

        pd = self._pd
        lines = []
        terminal_dicts = {}
        inds = np.arange(pd.dim, dtype=int)
        combs = itertools.combinations(inds, 2)
        for comb in list(combs):
            i = comb[0]
            j = comb[1]
            e1 = pd.terminal_entries[i]
            e2 = pd.terminal_entries[j]
            d1 = pd.terminal_data[i]
            d2 = pd.terminal_data[j]
            coord = get_coords(e1, e2, d1, d2)
            lines.append(coord)
            labelcoord = list(zip(*coord))
            terminal_dicts[labelcoord[0]] = e1
            terminal_dicts[labelcoord[1]] = e2

        entry_dicts = {}
        for i, entry in enumerate(pd.all_entries):
            d = pd.entry_data[i]
            coord = get_coords(entry, entry, d, d)
            labelcoord = list(zip(*coord))
            entry_dicts[entry] = labelcoord[0]
        return lines, terminal_dicts, entry_dicts

    def get_plot(self, **kwargs):
        fig = None
        if self.backend == "plotly":
            pass
        elif self.backend == "matplotlib":
            if self._dim <= 3:
                fig = self._get_2d_plot(**self.plotkwargs)
            elif self._dim == 4:
                fig = self._get_3d_plot(**self.plotkwargs)
        return fig

    def DataAugment(self, gridsize=0.005):
        from scipy import interpolate
        if self._dim < 3 or self._dim > 4:
            raise ValueError("Only used for 3- or 4-component systems.")

        pd = self._pd
        entries = pd.all_entries
        data = np.array(pd.entry_data)
        if self._dim == 3:
            data[:, 0:2] = triangular_coord(data[:, 0:2]).transpose()
            for i, e in enumerate(entries):
                data[i, 2] = e.energy

            xnew = np.arange(0, 1.0, gridsize)
            ynew = np.arange(0, 1.0, gridsize)

            func = interpolate.LinearNDInterpolator(data[:, 0:2], data[:, 2])
            outs = np.zeros((len(ynew), len(xnew)))
            colors = []
            for (i, xval) in enumerate(xnew):
                for (j, yval) in enumerate(ynew):
                    outs[j, i] = func(xval, yval)
                    colors.append(outs[j, i])
            return xnew, ynew, outs, colors
        else:
            data[:, 0:3] = tet_coord(data[:, 0:3]).transpose()
            for i, e in enumerate(entries):
                data[i, 3] = e.energy
            xnew = np.arange(0, 1.0, gridsize)
            ynew = np.arange(0, 1.0, gridsize)
            znew = np.arange(0, 1.0, gridsize)

            #xnew = np.arange(0, 1.0, gridsize)
            #ynew = np.arange(self._pd.min_ticks[2], self._pd.max_ticks[2], gridsize)
            #znew = np.arange(self._pd.min_ticks[3], self._pd.max_ticks[3], gridsize)

            func = interpolate.LinearNDInterpolator(data[:, 0:3], data[:, 3])
            outs = np.zeros((len(znew), len(ynew), len(xnew)))
            colors = []
            for (i, xval) in enumerate(xnew):
                for (j, yval) in enumerate(ynew):
                    for (k, zval) in enumerate(znew):
                        outs[k, j, i] = func(xval, yval, zval)
                        colors.append(outs[k, j, i])
            return xnew, ynew, znew, outs, colors

    def get_terminal_newlabel(self, entry):
        thiselements = entry.composition.elements
        concs = []
        for el in thiselements:
            concs.append(entry.composition.get_atomic_fraction(el))
        concs = np.array(concs)
        iele = np.argmax(concs)
        el = thiselements[iele]
        newlabel = self.elements_equivalent[el.symbol]
        newlabel = newlabel + "*"
        return newlabel

    def _get_2d_plot(self, label_entry=False, cmap="jet", term_ms=20, ms=20,
                     term_mfc="auto", mfc="auto", mec="k", mew=0.0, alpha=0.6,
                     DataAugment=True, gridsize=0.005, show_colorbar=True):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        if mfc is None: show_colorbar = False
        fig = plt.figure()
        ax = fig.add_subplot(111)
        font = FontProperties(family='times new roman', weight="bold", size=16)
        norm = Normalize(vmin=self.minimum, vmax=self.maximum)
        _map = ScalarMappable(norm=norm, cmap=cmap)

        (lines, labels, entries) = self.pd_plot_data

        if len(self._pd.elements) == 3:
            ax.axis("equal")
            ax.set_xlim((self._pd.min_ticks[1] - 0.1, self._pd.max_ticks[1] + 0.2 * self._pd.max_ticks[2]))
            ax.set_ylim((self._pd.min_ticks[2] - 0.1, self._pd.max_ticks[2]))
            ax.axis("off")
            center = (0.5, np.sqrt(3) / 6)

        colors = [entry.energy for coord, entry in labels.items()]
        cs = _map.to_rgba(colors)

        for x, y in lines:
            ax.plot(x, y, "k-", ms=0.0)

        ii = 0
        for x, y in labels.keys():
            if term_mfc is None:
                plt.plot(x, y, "o", ms=term_ms, mfc='k', mec=mec, mew=mew, alpha=alpha)
            else:
                plt.plot(x, y, "o", ms=term_ms, mfc=cs[ii], mec=mec, mew=mew, alpha=alpha)
            ii += 1

        count = 1
        newlabels = []
        newcount = 1
        for coords in sorted(labels.keys(), key=lambda x: -x[1]):
            entry = labels[coords]
            label = entry.name
            vec = np.array(coords) - center
            vec = vec / np.linalg.norm(vec) * 10 if np.linalg.norm(vec) != 0 else vec
            valign = "bottom" if vec[1] > 0 else "top"
            if vec[0] < -0.01:
                halign = "right"
            elif vec[0] > 0.01:
                halign = "left"
            else:
                halign = "center"

            if len(entry.composition.elements) == 1:
                pass
            else:
                newlabel = self.get_terminal_newlabel(entry)
                newlabels.append("{} : {}".format(latexify(newlabel), latexify(label)))
                label = newlabel
                newcount += 1
            count += 1

            plt.annotate(latexify(label), coords, xytext=vec,
                         textcoords="offset points", horizontalalignment=halign,
                         verticalalignment=valign, fontproperties=font)
        font.set_size(14)
        plt.figtext(0.01, 0.01, "\n".join(newlabels), fontproperties=font)

        font = FontProperties()
        font.set_size(12)
        ecolors = [entry.energy for entry, coord in entries.items()]
        colors.extend(ecolors)
        cs = _map.to_rgba(ecolors)
        ii = 0
        for entry, coords in entries.items():
            vec = np.array(coords) - center
            vec = vec / np.linalg.norm(vec) * 10 if np.linalg.norm(vec) != 0 else vec
            label = entry.name
            if mfc is None:
                plt.plot(coords[0], coords[1], "o", ms=ms, mfc='k', mec=mec, mew=mew, alpha=alpha)
            else:
                plt.plot(coords[0], coords[1], "o", ms=ms, mfc=cs[ii], mec=mec, mew=mew, alpha=alpha)

            if label_entry:
                plt.annotate(latexify(label), coords, xytext=vec, textcoords="offset points",
                             horizontalalignment=halign, color="b", verticalalignment=valign,
                             fontproperties=font)
            ii += 1

        if DataAugment:
            xnew, ynew, outs, dacs = self.DataAugment(gridsize=gridsize)
            cs = _map.to_rgba(dacs)
            ii = 0
            for (i, xval) in enumerate(xnew):
                for (j, yval) in enumerate(ynew):
                    if np.isnan(dacs[ii]):
                        pass
                    else:
                        if mfc is None:
                            plt.plot(xval, yval, "o", ms=ms, mfc='k', mec=mec, mew=mew, alpha=alpha)
                        else:
                            plt.plot(xval, yval, "o", ms=ms, mfc=cs[ii], mec=mec, mew=mew, alpha=alpha)
                    ii += 1

        if show_colorbar:
            font = FontProperties(family='times new roman', size=13)
            _map.set_array(colors)
            #cbar = plt.colorbar(_map)
            cbar = plt.colorbar(_map, ax=plt.gca())
            cbar.set_label("", rotation=-90, ha="left", va="center", fontproperties=font)
            cbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.07)
        return fig, ax

    def _get_3d_plot(self, label_entry=False, cmap="jet", term_ms=20, ms=20,
                     term_mfc="auto", mfc="auto", mec="k", mew=0.0, alpha=0.6,
                     DataAugment=True, gridsize=0.02, show_colorbar=True):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        if mfc is None: show_colorbar = False

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        font = FontProperties(family='times new roman', weight="bold", size=13)
        norm = Normalize(vmin=self.minimum, vmax=self.maximum)
        _map = ScalarMappable(norm=norm, cmap=cmap)

        (lines, labels, entries) = self.pd_plot_data

        colors = [entry.energy for coord, entry in labels.items()]
        cs = _map.to_rgba(colors)
        for x, y, z in lines:
            ax.plot(x, y, z, "k-", ms=0)

        ii = 0
        for x, y, z in labels.keys():
            if term_mfc is None:
                plt.plot(x, y, z, "o", mfc='k', ms=term_ms, mec=mec, mew=mew, alpha=alpha)
            else:
                plt.plot(x, y, z, "o", mfc=cs[ii], ms=term_ms, mec=mec, mew=mew, alpha=alpha)
            ii += 1

        count = 1
        newlabels = []
        newcount = 1
        for coords in sorted(labels.keys()):
            entry = labels[coords]
            label = entry.name
            if count == 1:
                coords_shift = [-0.1, -0.07, 0.0]
            elif count == 2 or count == 4:
                coords_shift = [0.06, 0.0, 0.0]
            elif count == 3:
                coords_shift = [0.0, 0.0, 0.050]
            coords += np.array(coords_shift)
            if len(entry.composition.elements) == 1:
                pass
            else:
                newlabel = self.get_terminal_newlabel(entry)
                #ax.text(coords[0], coords[1], coords[2], latexify(newlabel), fontsize=12)
                newlabels.append("{} : {}".format(latexify(newlabel), latexify(label)))
                label = newlabel
                newcount += 1
            ax.text(coords[0], coords[1], coords[2], latexify(label), fontproperties=font)
            count += 1
        font.set_size(14)
        plt.figtext(0.01, 0.01, "\n".join(newlabels), fontproperties=font)

        ecolors = [entry.energy for entry, coord in entries.items()]
        colors.extend(ecolors)
        cs = _map.to_rgba(ecolors)
        ii = 0
        for entry, coords in entries.items():
            label = entry.name
            if mfc is None:
                plt.plot(coords[0], coords[1], coords[2], "o", ms=ms, mfc='k', mec=mec, mew=mew, alpha=alpha)
            else:
                plt.plot(coords[0], coords[1], coords[2], "o", ms=ms, mfc=cs[ii], mec=mec, mew=mew, alpha=alpha)
            ii += 1

        if DataAugment:
            xnew, ynew, znew, outs, dacs = self.DataAugment(gridsize=gridsize)
            cs = _map.to_rgba(dacs)
            ii = 0
            for (i, xval) in enumerate(xnew):
                for (j, yval) in enumerate(ynew):
                    for (k, zval) in enumerate(znew):
                        if np.isnan(dacs[ii]):
                            pass
                        else:
                            if mfc is None:
                                plt.plot(xval, yval, zval, "o", ms=ms, mfc='k', mec=mec, mew=mew, alpha=alpha)
                            else:
                                plt.plot(xval, yval, zval, "o", ms=ms, mfc=cs[ii], mec=mec, mew=mew, alpha=alpha)
                        ii += 1

        if show_colorbar:
            font = FontProperties(family='times new roman', size=13)
            _map.set_array(colors)
            cbar = plt.colorbar(_map, ax=plt.gca())
            cbar.set_label("", rotation=-90, ha="left", va="center", fontproperties=font)
            cbar.ax.tick_params(labelsize=12)

        ax.axis("off")
        ax.set_xlim(self._pd.min_ticks[1] - 0.1, self._pd.max_ticks[1] * 0.72 + 0.15)
        ax.set_ylim(self._pd.min_ticks[2], self._pd.max_ticks[2] * 0.75 + 0.10)
        ax.set_zlim(self._pd.min_ticks[3], self._pd.max_ticks[3] * 0.635 + 0.07)  # pylint: disable=E1101
        return fig, ax

    def get_contour_pd_plot(self, gridsize=0.01, inline=False, fontsize=12, show_term=True, show_ms=True):
        from matplotlib import cm
        from matplotlib.font_manager import FontProperties
        if show_ms:
            ms = 4
        else:
            ms = 0
        if show_term:
            term_ms = 4
        else:
            term_ms = 0
        thissetts = {"show_colorbar": self.plotkwargs["show_colorbar"], "term_ms": term_ms, "ms": ms,
                     "mec": "k", "mew": 1.5, "alpha": 0.5,
                     "DataAugment": False}
        fig, ax = self._get_2d_plot(**thissetts)

        xnew, ynew, outs, dacs = self.DataAugment(gridsize=gridsize)

        # pylint: disable=E1101
        font = FontProperties(family='times new roman', size=fontsize)
        if inline:
            cs = ax.contourf(xnew, ynew, outs, 10,
                             cmap=cm.jet, vmin=self.minimum, vmax=self.maximum, extend="both")

            ax.contourf(xnew, ynew, outs, 50,
                        cmap=cm.jet, vmin=self.minimum, vmax=self.maximum, extend="both")

            ax.clabel(cs, inline=True, fontsize=fontsize, colors='k')
        else:
            cs = ax.contourf(xnew, ynew, outs, 1000,
                             cmap=cm.jet, vmin=self.minimum, vmax=self.maximum, extend="both")

        '''
        font = FontProperties(family='times new roman', size=12)
        #print(self.minimum, self.maximum)
        cbar = fig.colorbar(cs)
        incr = (self.local_maximum-self.local_minimum)/9
        bounds = np.arange(self.local_minimum, self.local_maximum, incr)
        #cbar.set_ticks(bounds)
        cbar.set_label("",rotation=-90,ha="left",va="center", fontproperties=font)
        cbar.ax.tick_params(labelsize=12)
        '''
        return fig


def triangular_coord(coord):
    """
    Convert a 2D coordinate into a triangle-based coordinate system for a
    prettier phase diagram.

    Args:
        coord: coordinate used in the convex hull computation.

    Returns:
        coordinates in a triangular-based coordinate system.
    """
    unitvec = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])

    result = np.dot(np.array(coord), unitvec)
    return result.transpose()


def tet_coord(coord):
    """
    Convert a 3D coordinate into a tetrahedron based coordinate system for a
    prettier phase diagram.

    Args:
        coord: coordinate used in the convex hull computation.

    Returns:
        coordinates in a tetrahedron-based coordinate system.
    """
    unitvec = np.array(
        [
            [1, 0, 0],
            [0.5, np.sqrt(3) / 2, 0],
            [0.5, 1.0 / 3.0 * np.sqrt(3) / 2, np.sqrt(6) / 3],
        ]
    )
    result = np.dot(np.array(coord), unitvec)
    return result.transpose()


class PlotFile:
    def __init__(self, data, rgba=[["Ti", "Zr", "Hf"], ["V", "Nb", "Ta"], ["Cr", "Mo", "W"]], fontsize=16):
        self.data = data
        if isinstance(rgba, list) or isinstance(rgba, np.ndarray):
            if len(rgba) != 3:
                raise ValueError("rgba must be a 3-element array with elements of red, green and blue.")
            else:
                for i in range(3):
                    if isinstance(rgba[i], str):
                        rgba[i] = [rgba[i]]
                    elif isinstance(rgba[i], list) or isinstance(rgba[i], np.ndarray):
                        pass
                    else:
                        raise ValueError("Each element in rgba must be an element symbol or a list of symbols.")
                self.rgba = rgba
        else:
            self.rgba = [["Ti", "Zr", "Hf"], ["V", "Nb", "Ta"], ["Cr", "Mo", "W"]]

        self.fontsize = fontsize
        self.colors = ['k', 'g', 'r', 'b', 'y', 'c']
        self.ncolor = len(self.colors)
        self.markers = ['o', '+', 'd', 'x', 'v', '^']
        self.linewidth = 2
        self.nrom = 15
        self.ncurve_fit = 50

    def select_data_from_XY(self, xs, ys, condition):
        inds = np.arange(xs.shape[1], dtype=int)
        goods = np.compress(eval(condition), inds)
        bads = np.delete(inds, goods)
        xs = xs[:, goods]
        ys = ys[:, :, goods]
        badcompstrs = self.compstrs[bads]
        self.colormaps = self.colormaps[goods]
        self.alphas = self.alphas[goods]
        self.compstrs = self.compstrs[goods]
        return xs, ys, badcompstrs

    def get_thisele_concs(self, sym):
        concs = []
        el = Element(sym)
        for icomp in range(len(self.compstrs)):
            compstr = self.compstrs[icomp]
            comp = Composition(compstr)
            thisconc = comp.get_atomic_fraction(el)
            thisconc = int(round(100.0 * thisconc, 0))
            concs.append(thisconc)
        concs = np.array(concs)
        return concs

    def get_colors(self):
        colors = []
        alphas = []
        for icomp in range(len(self.compstrs)):
            compstr = self.compstrs[icomp]
            comp = Composition(compstr)
            thisc = np.array([0.0, 0.0, 0.0])
            totalf = 0.0
            for i, el in enumerate(comp):
                frac = comp.get_atomic_fraction(el)
                if el.symbol in self.rgba[0]:
                    thisc[0] += frac
                    totalf += frac
                elif el.symbol in self.rgba[1]:
                    thisc[1] += frac
                    totalf += frac
                elif el.symbol in self.rgba[2]:
                    thisc[2] += frac
                    totalf += frac
            if totalf > 1.0:
                thisc = thisc / totalf
                totalf = 1.0

            if totalf > 0:
                colors.append(thisc)
                alphas.append(totalf)
            else:
                colors.append(np.array([0.33, 0.34, 0.33]))
                alphas.append(totalf + 0.05)

        self.colormaps = np.array(colors)
        self.alphas = np.array(alphas)

    def generate2d_xys(self, thisdf, xkeys, ykeys, SameCrystal=True):
        if isinstance(xkeys, str):
            xkeys = [xkeys]
        else:
            raise ValueError("Unrecognized xkeys!")
        if isinstance(ykeys, str): ykeys = [ykeys]
        self.compstrs = thisdf["Composition"].to_numpy()

        xs = []
        ys = []
        for ix in range(len(xkeys)):
            xkey = xkeys[ix]
            if xkey in Element_negativity:
                thisxs = self.get_thisele_concs(xkey)
            elif xkey in thisdf.columns:
                thisxs = thisdf[xkey].to_numpy()
            else:
                raise ValueError(f"Unrecognized {xkey}!")

            thisys = []
            for iy in range(len(ykeys)):
                ykey = ykeys[iy]
                if ykey in thisdf.columns:
                    thisys.append(thisdf[ykey].to_numpy())
                elif "ROM" in ykey.upper():
                    thisdf = self.data.compute_ROMs_df(thisdf, [xkey], SameCrystal=SameCrystal)
                    thisys.append(thisdf[xkey + "_ROM"].to_numpy())
                elif "SCALE" in ykey.upper():
                    thisdf = self.data.compute_rescaled_elastic(thisdf, [xkey])
                    thisys.append(thisdf[xkey + "_SCALED"].to_numpy())
                else:
                    raise ValueError(f"Unrecognized {ykey}!")
            xs.append(thisxs)
            ys.append(thisys)
        xs = np.array(xs)
        ys = np.array(ys)
        self.get_colors()
        return xs, ys

    def plot2d(self, xs, ys, xkeys, ykeys, style='scatter', colormaps=True, xaxis2count=False, savefig=False,
               outfile="output.jpeg",
               isROM=False, Label_R2=False, label_abnormal=False, label_axis=False, **kwargs):
        nyticks = None
        nxticks = None
        if kwargs and "nxticks" in kwargs:
            nxticks = kwargs["nxticks"]
        if kwargs and "nyticks" in kwargs:
            nyticks = kwargs["nyticks"]

        nsubplot = len(xs)
        fig = plt.figure()
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        axs = [None] * nsubplot
        popt = None
        pcov = None
        r2 = None
        MAE = 0.0
        fontsize = self.fontsize
        print(xkeys)
        print(ykeys)
        print("===")

        for iplot in range(nsubplot):
            nline = len(ys[iplot])
            lines = [None] * nline
            thisx = xs[iplot]
            if colormaps:
                cs = self.colormaps[0:len(thisx)]
            else:
                cs = ['k'] * len(thisx)

            if xaxis2count: thisx = np.arange(len(thisx)) + 1
            if iplot == 0:
                axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1)
            else:
                axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1, sharex=axs[0])
            for iline in range(nline):
                jj = iline % self.ncolor
                thisy = ys[iplot][iline]
                if isROM:
                    tmp = thisx[0:len(thisx)]
                    thisx = thisy[0:len(thisy)]
                    thisy = tmp[0:len(tmp)]
                    tmp = None

                if style == "line":
                    lines[iline], = axs[iplot].plot(thisx, thisy, color=self.colors[jj], linewidth=self.linewidth)
                elif style == "scatter":
                    axs[iplot].scatter(thisx, thisy, s=20, c=cs, marker=self.markers[jj], alpha=self.alphas)

                if isROM:
                    if style == "line":
                        jj = nline % self.ncolor
                        thisc = self.colors[jj]
                    elif style == "scatter":
                        thisc = 'k'
                    romx = np.linspace(np.min(thisx), np.max(thisx), self.nrom)
                    axs[iplot].plot(romx, romx, color=thisc, linewidth=self.linewidth - 1.0, linestyle=':')

                if "curvefit" in kwargs and kwargs["curvefit"]:
                    popt, pcov = curve_fit(eval('f{0}'.format(kwargs['funcID'])), thisx, thisy)
                    xnew = np.linspace(np.min(thisx), np.max(thisx), self.ncurve_fit)
                    ynew = eval(f"f{kwargs['funcID']}(xnew,*popt)")
                    axs[iplot].plot(xnew, ynew, color=self.colors[jj], linestyle='dashed', linewidth=1.0)
                    yr = eval(f"f{kwargs['funcID']}(thisx,*popt)")
                    yr = thisy - yr
                    ydiff = thisy - np.mean(thisy)
                    r2 = 1.0 - np.sum(yr * yr) / np.sum(ydiff * ydiff)
                    absyr = np.absolute(yr)
                    n = len(absyr)
                    MAE = np.sum(absyr) / n
                    # for my purpose
                    if label_abnormal:
                        m = np.mean(absyr)
                        s = np.std(absyr)
                        if s > 0:
                            absyrd = np.absolute((absyr - m) / s)
                            inds = np.arange(len(thisx), dtype=int)
                            inds = np.compress(absyrd > 3.0, inds)
                            if len(inds) > 0:
                                labelx = thisx[inds]
                                labely = thisy[inds]
                                thiscompstrs = self.compstrs[inds]
                                print(f"labels:{thiscompstrs}")
                                print("========")
                                for i in range(len(labelx)):
                                    axs[iplot].text(labelx[i], labely[i], thiscompstrs[i], fontsize=fontsize - 4)
                    if Label_R2:
                        label = "R2 = " + str(round(r2, 4))
                        labelx = np.min(thisx) + 0.6 * (np.max(thisx) - np.min(thisx))
                        labely = np.min(thisy) + 0.3 * (np.max(thisy) - np.min(thisy))
                        axs[iplot].text(labelx, labely, label, fontsize=fontsize - 2)
                elif isROM:
                    absyr = np.absolute(thisy - thisx)
                    n = len(absyr)
                    MAE = np.sum(absyr) / n
                    if label_abnormal:
                        m = np.mean(absyr)
                        s = np.std(absyr)
                        if s > 0:
                            absyrd = np.absolute((absyr - m) / s)
                            inds = np.arange(len(thisx), dtype=int)
                            inds = np.compress(absyrd > 3.5, inds)
                            if len(inds) > 0:
                                labelx = thisx[inds]
                                labely = thisy[inds]
                                thiscompstrs = self.compstrs[inds]
                                print(f"labels:{thiscompstrs}")
                                print("======")
                                for i in range(len(labelx)):
                                    axs[iplot].text(labelx[i], labely[i], thiscompstrs[i], fontsize=fontsize - 4)

                print(f"iplot:{iplot} iline:{iline} popt:{popt} R2:{r2} MAE:{MAE}")

            if isinstance(nyticks, int):
                vticks, ticklabels = get_ticks_labels(ys[iplot], nyticks)
                axs[iplot].set_yticks(vticks, labels=ticklabels)
            plt.setp(axs[iplot].get_yticklabels(), fontsize=self.fontsize)
            if label_axis:
                if isROM:
                    if nsubplot > 1:
                        ylabel = xkeys[iplot]
                    else:
                        ylabel = xkeys
                else:
                    if nline > 1:
                        ylabel = ""
                    else:
                        ylabel = ykeys
                axs[iplot].set_ylabel(ylabel, fontsize=fontsize + 2)

            if nsubplot > 1:
                label = str(xkeys[iplot])
                labelx = np.min(thisx) + 0.8 * (np.max(thisx) - np.min(thisx))
                labely = np.min(thisy) + 0.5 * (np.max(thisy) - np.min(thisy))
                axs[iplot].text(labelx, labely, label, fontsize=fontsize - 2)
                if iplot == nsubplot - 1:
                    if isinstance(nxticks, int):
                        vticks, ticklabels = get_ticks_labels(thisx, nxticks)
                        axs[iplot].set_xticks(vticks, labels=ticklabels)
                    plt.setp(axs[iplot].get_xticklabels(), fontsize=self.fontsize, visible=True)
                    if label_axis:
                        if isROM:
                            xlabel = "ROM"
                        else:
                            xlabel = "Elements"
                        axs[iplot].set_xlabel(xlabel, fontsize=fontsize + 2)
                else:
                    plt.setp(axs[iplot].get_xticklabels(), visible=False)
            else:
                if isinstance(nxticks, int):
                    vticks, ticklabels = get_ticks_labels(thisx, nxticks)
                    axs[iplot].set_xticks(vticks, labels=ticklabels)
                plt.setp(axs[iplot].get_xticklabels(), fontsize=self.fontsize, visible=True)
                if label_axis:
                    if isROM:
                        xlabel = "ROM"
                    else:
                        xlabel = xkeys
                    axs[iplot].set_xlabel(xlabel, fontsize=fontsize + 2)

        if savefig:
            if ".jpeg" in outfile:
                plt.savefig(outfile, dpi=200, format='jpeg', pil_kwargs={'optimize': True})
            elif ".png" in outfile:
                plt.savefig(outfile, bbox_inches='tight')
            plt.close(fig)
            fig = None

        return fig, popt, pcov, MAE

    def plot_prop_compspace(self, df, compspace, key, key4terminals=None, screen=["inf", "inf"],
                            global_extrema=False, set_terminal2entry=True, style="contour",
                            gridsize4contour=0.01, inline=False, minimum=None, maximum=None,
                            show_term=True, show_ms=True,
                            savefig=False, outfile="output.jpeg", **kwargs):
        if style == "contour":
            if len(compspace) != 3: raise ValueError("Contour plot must be a 3-component composition space!")
        else:
            if len(compspace) > 4: raise ValueError("Composition space must be a smaller than 4!")

        outlist = self.data.get_xy4compspace(df, compspace, key, key4terminals=key4terminals,
                                             screen=screen, global_extrema=global_extrema)

        compositions = []
        for i in range(len(outlist[0])):
            compositions.append(Composition(outlist[0][i]))
        entries = []

        for i in range(len(compositions)):
            entries.append(PDEntry(compositions[i], outlist[1][i]))

        terminal_entries = []
        for i in range(len(outlist[2])):
            terminal_entries.append(PDEntry(outlist[2][i], outlist[3][i]))
        csdiagram = CSDiagram(entries, terminal_entries,
                              isElements=outlist[4], set_terminal2entry=set_terminal2entry)

        if isinstance(minimum, float):
            pass
        else:
            minimum = outlist[5]

        if isinstance(maximum, float):
            pass
        else:
            maximum = outlist[6]

        pd_plotter = CSPlotter(csdiagram, minimum=minimum, maximum=maximum, **kwargs)
        if style == "contour":
            fig = pd_plotter.get_contour_pd_plot(gridsize=gridsize4contour, inline=inline,
                                                 show_term=show_term, show_ms=show_ms)
        else:
            fig, ax = pd_plotter.get_plot()

        outlist = None
        entries = None
        terminal_entries = None
        compositions = None
        if savefig:
            if style[0:4].upper() == "AUTO":
                if ".jpeg" in outfile:
                    plt.savefig(outfile, dpi=200, format='jpeg', pil_kwargs={'optimize': True})
                elif ".png" in outfile:
                    plt.savefig(outfile, bbox_inches='tight')
                elif ".pdf" in outfile:
                    plt.savefig(outfile)
            else:
                if ".jpeg" in outfile:
                    fig.savefig(outfile, dpi=200, format='jpeg', pil_kwargs={'optimize': True})
                elif ".png" in outfile:
                    fig.savefig(outfile, bbox_inches='tight')
                elif ".pdf" in outfile:
                    fig.savefig(outfile)
            plt.close()
            fig = None

        return fig
