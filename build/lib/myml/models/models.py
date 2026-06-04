import copy
import os
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import interpolate
from sklearn import svm
from sklearn import ensemble
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import DecisionBoundaryDisplay

from myml.data.data import myData, DATA_PATH
from myml.myelements.myelements import eos_keys, hcp_eos_keys, transformkey, elastic_keys, TC_keys
from myml.myglobal import config_vars, Constants, Element_negativity
from myml.myglobal import get_sorted_compspace
from myml.plot.plot import plot_linear_xys

MODEL_PATH = config_vars["MODEL_PATH"]
float_format = Constants["float_format"]


def get_default_outfile(fname):
    fname = fname.split("/")
    fname = fname[-1]
    fname = fname.replace(".csv", "")
    fname = fname.replace("_IP", "")
    fname = fname.replace("_GBR", "")
    fname = fname.replace("_HGBR", "")
    fname = fname.replace("_MLPR", "")
    #fname = fname.replace("_NDIP", "")
    return fname


class NDInterpolate:
    def __init__(self, data, compspace):
        self.data = data
        self.compspace = get_sorted_compspace(compspace)
        self.ncompon = len(self.compspace)
        self.ndim = len(self.compspace)
        self.latname = self.data.df.iloc[0]["Crystal"]

        self.SAVE_PATH = "NDIP_Models"

        csstr = ""
        for sym in self.compspace:
            csstr += sym
        self.mname_header = self.latname + "_" + csstr + "_"
        self.funcs = {}

    def get_data(self, key, condition, df=None, screen=["inf", "inf"], extended=True, norm=False,
                 validate_data=True, undotype=1, ncompon=None, set2gooddf=False):
        if df is None:
            df = self.data.gooddf.copy()
        else:
            pass
        xs, ys, xs4eval, compids4eval = self.data.get_data4compspace(
            df, self.compspace, key, condition, key4terminals=None,
            screen=screen, extended=extended, norm=norm,
            validate_data=validate_data, undotype=undotype,
            ncompon=ncompon, set2gooddf=set2gooddf)

        return xs, ys, xs4eval, compids4eval

    def load_model(self, key):
        isValid = True
        mname = self.mname_header + key
        if os.path.isfile(os.path.join(MODEL_PATH, self.SAVE_PATH, mname)):
            with open(os.path.join(MODEL_PATH, self.SAVE_PATH, mname), 'rb') as f:
                try:
                    self.funcs[key] = pickle.load(f)
                except:
                    print(f"Fail to load {os.path.join(MODEL_PATH, self.SAVE_PATH, mname)}")
                    isValid = False
        else:
            isValid = False
        return isValid

    def save_model(self, key):
        mname = self.mname_header + key
        if not os.path.isdir(os.path.join(MODEL_PATH, self.SAVE_PATH)):
            os.mkdir(os.path.join(MODEL_PATH, self.SAVE_PATH))
        with open(os.path.join(MODEL_PATH, self.SAVE_PATH, mname), 'wb') as f:
            pickle.dump(self.funcs[key], f)

    def get_NDInterpolate_func(self, xs, ys, key, loadmodel=False, savemodel=False):
        isValid = False
        if loadmodel:
            isValid = self.load_model(key)

        if not isValid:
            thisxs = xs[:, 1:self.ndim]
            if self.ndim == 2:
                self.funcs[key] = interpolate.interp1d(thisxs.flatten(), ys)
            else:
                self.funcs[key] = interpolate.LinearNDInterpolator(thisxs, ys)
        if savemodel: self.save_model(key)

    def Interpolate_data(self, xs4eval, key, loadmodel=False):
        isValid = True
        if loadmodel:
            isValid = self.load_model(key)
        if not isValid:
            print(f"Fail to load NDIP model for {key}")
            return np.array([])

        if isinstance(xs4eval, list):
            xs4eval = np.array(xs4eval)
        elif isinstance(xs4eval, np.ndarray):
            pass
        else:
            isValid = False
        if isValid:
            if len(xs4eval.shape) == 1:
                if xs4eval.shape[0] == 0:
                    isValid = False
                else:
                    xs4eval = np.array([list(xs4eval)])
        if isValid:
            if len(xs4eval.shape) == 2:
                if xs4eval.shape[1] != self.ndim: isValid = False
            elif len(xs4eval.shape) > 2:
                raise ValueError("xs4eval must be 2 dimensional array.")

        if not isValid:
            outs4eval = np.array([])
        else:
            thisxs = xs4eval[:, 1:self.ndim]
            if self.ndim == 2:
                outs4eval = self.funcs[key](thisxs.flatten())
            else:
                outs4eval = self.funcs[key](thisxs)
        return outs4eval


def get_r2(ytest, ypred):
    yr = ytest - ypred
    ydiff = ytest - np.mean(ytest)
    r2 = 1.0 - np.sum(yr * yr) / np.sum(ydiff * ydiff)
    return r2


rename_dict4features = {"Exp_Bulk_ROM": "Bulk", "Exp_Youngs_ROM": "Youngs", "Exp_Shear_ROM": "Shear",
                        "Exp_Bulk_ROM_SCALED": "Bulk", "Exp_Youngs_ROM_SCALED": "Youngs",
                        "Exp_Shear_ROM_SCALED": "Shear",
                        "Exp_Shear_ROM_DELTA": "S_DEL", "Exp_Youngs_ROM_DELTA": "Y_DEL", "Exp_Bulk_ROM_DELTA": "B_DEL",
                        "Exp_Shear_DISTORT": "S_DIS", "Exp_Youngs_DISTORT": "Y_DIS",
                        "Eform": "Hmix_CPA", "Sconf": "Smix", "ElasticEnergy": "H_elas",
                        "radius_TC_DELTA": "r_DEL", "volume_DELTA": "V_DEL", "E_negativity": "Eneg",
                        "radius_TC_DISTORT": "r_DIS", "volume_DISTORT": "V_DIS", "temp_ratio": "T_ratio",
                        "Exp_Poisson_ROM": "Poisson", "Yield_M1": "YS_M1", "delta_Eb0": "del_Eb0", "sigma_y0": "sig_y0"}


class MLRegressor:
    def __init__(self, data, keys, modelname="GBR", mname_header=None,
                 features=None, elements=None, ncompons=None, normalization=False,
                 extended=False, valid_size=0.1, random_state=None,
                 learning_rate=0.01, loss="squared_error", tol=0.001,
                 Visualize_Results=False, Importance_mode=[0, 1], Find_PDP=False,
                 Vertical=False, features4plot=None, Find_interPDP=False, interactive_features=None, subsample=None,
                 SHAP_Plot=False, SHAP_style="beeswarm", max_display=15,
                 n_estimators=500, max_depth=None, min_samples_split=None, min_samples_leaf=None,
                 max_iter=500, max_leaf_nodes=None,
                 hidden_layer_sizes=None, learning_rate_style="adaptive", activation="relu", solver="adam",
                 early_stopping=True, max_fun=15000, warm_start=True):

        self.SAVE_PATH = "MLR_Models"

        self.data = data
        if "HGBR" in modelname:
            self.modelname = "HGBR"
        elif "GBR" in modelname:
            self.modelname = "GBR"
        else:
            self.modelname = "MLPR"

        if mname_header is None:
            outfile = get_default_outfile(self.data.fname)
            outfile += "_" + self.modelname
        else:
            outfile = mname_header + "_" + self.modelname

        self.outfig_header = outfile + "_"
        self.mname_header = outfile + "_"

        self.keys = keys
        if elements is None:
            elements = data.elements[0:len(data.elements)]
        self.elements = elements
        if ncompons is not None:
            if isinstance(ncompons, int): ncompons = [ncompons]
        self.ncompons = ncompons
        self.normalization = normalization

        self.valid_size = valid_size
        self.random_state = random_state
        self.latname = self.data.df.iloc[0]["Crystal"]

        if self.ncompons is not None:
            self.data.select_by_ncompons(self.data.gooddf, self.ncompons, set2gooddf=True)

        if features is None:
            self.feature_type = "default"
            features = []
            labels = []
            for i in range(len(self.elements)):
                ii = Element_negativity.index(self.elements[i])
                features.append("CONC" + str(ii))
                labels.append(self.elements[i])
        else:
            self.feature_type = "custom"
            labels = []
            for feature in features:
                if "CONC" in feature:
                    label = feature.replace("CONC", "")
                    i = int(label)
                    label = self.elements[i]
                elif "_norm" in feature:
                    label = feature.replace("_norm", "")
                else:
                    label = feature
                labels.append(label)
        self.features = features
        self.labels = labels
        self.nfeature = len(self.features)
        for i in range(len(self.labels)):
            label = self.labels[i]
            try:
                self.labels[i] = rename_dict4features[label]
            except:
                pass

        if features4plot is None: features4plot = np.arange(self.nfeature, dtype=int)
        self.features4plot = features4plot

        self.nsplit = 1
        if len(self.keys) == 1 and "Yield_EXP" in self.keys[0]:
            if len(self.features) > 1 and self.features[0] == "temp_ratio": self.nsplit = 2

        if self.normalization:
            self.data.gooddf = self.data.normalization(self.data.gooddf, keys=self.keys,
                                                       extended=False, ele_ref="ADD",
                                                       savefile=False, outfile=None, Add_norm=False)
            self.data.gooddf = self.data.normalization(self.data.gooddf, keys=self.features,
                                                       extended=False, ele_ref="ADD",
                                                       savefile=False, outfile=None, Add_norm=False)

        for key in self.keys:
            if key in self.features:
                raise ValueError(f"The {key} in both features and target keys!")

        self.keyapps = ["_reg", "_mse", "_rmse", "_mae", "_rmae"]
        self.init_models()

        self.Visualize_Results = Visualize_Results
        if isinstance(Importance_mode, bool):
            if Importance_mode:
                Importance_mode = [0, 1]
            else:
                Importance_mode = None
        elif isinstance(Importance_mode, str):
            if Importance_mode.upper() == "BOTH":
                Importance_mode = [0, 1]
            elif "PERMU" in Importance_mode.upper():
                Importance_mode = [1]
            else:
                Importance_mode = [0]
        elif isinstance(Importance_mode, list):
            m = []
            l = [0, 1]
            for i in Importance_mode:
                if i in l: m.append(i)
            if len(m) == 0:
                Importance_mode = None
            else:
                Importance_mode = m[0:len(m)]
        else:
            Importance_mode = None

        self.Importance_mode = Importance_mode
        self.Find_PDP = Find_PDP
        self.Find_interPDP = Find_interPDP
        self.interactive_features = interactive_features
        if self.Find_interPDP:
            if interactive_features is None:
                self.inter_features = []
                for i in range(self.nfeature):
                    for j in range(i + 1, self.nfeature):
                        self.inter_features.append((i, j))
                self.ninter_plot = len(self.inter_features)
            else:
                interactive_features = np.array(interactive_features)
                if len(interactive_features.shape) == 1:
                    self.inter_features = []
                    ninter_plot = int(len(interactive_features) / 2)
                    for i in range(ninter_plot):
                        ii = self.features.index(interactive_features[i * 2])
                        jj = self.features.index(interactive_features[2 * i + 1])
                        self.inter_features.append((ii, jj))
                else:
                    for i in range(interactive_features.shape[0]):
                        ii = self.features.index(interactive_features[i][0])
                        jj = self.features.index(interactive_features[i][1])
                        self.inter_features.append((ii, jj))
                self.ninter_plot = len(self.inter_features)
        else:
            self.inter_features = None
            self.ninter_plot = 0

        self.Vertical = Vertical
        ndata = len(self.data.gooddf)
        self.ndata = ndata
        if subsample is None:
            subsample = min(int(ndata / 30), 100)
        self.subsample = subsample
        if self.modelname == "HGBR" or self.modelname == "MLPR":
            self.Visualize_Results = False
            self.Importance_mode = None

        self.SHAP_Plot = SHAP_Plot
        self.SHAP_style = SHAP_style
        self.max_display = max_display

        self.nele = len(self.elements)
        if max_depth is None: max_depth = self.nfeature
        if min_samples_leaf is None:
            if self.data.Source[0:3].upper() == "INT":
                combs = combinations(np.arange(self.nele - 1), 2)
                ncombs = len(list(combs))
                min_samples_leaf = int(ncombs / 2)
                if min_samples_leaf < 6: min_samples_leaf = 6
            else:
                min_samples_leaf = 3

        if self.modelname == "HGBR":
            if max_leaf_nodes is None: max_leaf_nodes = 8 * 3
            self.params = {
                "loss": loss,
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "max_leaf_nodes": max_leaf_nodes,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "validation_fraction": valid_size,
                "tol": tol,
                "random_state": random_state,
                "warm_start": warm_start,
            }
        elif self.modelname == "GBR":
            if min_samples_split is None:
                if self.data.Source[0:3] == "Int":
                    if self.ncompons is None:
                        min_samples_split = 7
                    else:
                        if 5 in self.ncompons:
                            min_samples_split = 6
                        elif 4 in self.ncompons:
                            min_samples_split = 7
                        else:
                            min_samples_split = 8
                else:
                    min_samples_split = 3
            self.params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "learning_rate": learning_rate,
                "loss": loss,
                "tol": tol,
                "warm_start": warm_start,
            }
        else:
            if hidden_layer_sizes is None:
                nlayer = max(int(ndata / 64), 1)
                nlayer = min(nlayer, 100)
                nlayer += 2
                nsize = 2.0 * int(np.sqrt(ndata / (nlayer - 2)))
                nint = round(np.log(nsize) / np.log(2), 0)
                nsize = int(np.power(2, nint))
                if nsize < 4: nsize = 4
                if nsize > 128: nsize = 128
                hidden_layer_sizes = (nlayer, nsize)
            self.hidden_layer_sizes = hidden_layer_sizes

            if early_stopping: max_iter = 100000
            self.params = {
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "activation": activation,
                "solver": solver,
                "learning_rate": learning_rate_style,
                "learning_rate_init": learning_rate,
                "max_iter": max_iter,
                "random_state": random_state,
                "early_stopping": early_stopping,
                "validation_fraction": valid_size,
                "tol": tol,
                "max_fun": max_fun,
            }

        self.performance_keys = ["key", "mse_test", "rmse_test", "mae_test", "rmae_test", "r2_test",
                                 "mse_all", "rmse_all", "mae_all", "rmae_all", "r2_all"]
        self.performance_file = self.mname_header + "performance.csv"
        self.performance_df = pd.DataFrame(columns=self.performance_keys)
        self.performance_df["key"] = self.keys
        for key in self.performance_keys[1:len(self.performance_keys)]:
            self.performance_df[key] = np.zeros(len(self.keys))
        self.performance_df = self.performance_df.set_index("key")

    def init_models(self):
        self.models = {}
        for key in self.keys:
            for app in self.keyapps[0:1]:
                thiskey = key + app
                self.models[thiskey] = None

    def load_model(self, key):
        isValid = True
        mname = key + self.keyapps[0]
        fname = self.mname_header + mname
        if os.path.isfile(os.path.join(MODEL_PATH, self.SAVE_PATH, fname)):
            with open(os.path.join(MODEL_PATH, self.SAVE_PATH, fname), 'rb') as f:
                try:
                    self.models[mname] = pickle.load(f)
                except:
                    print(f"Fail to load {os.path.join(MODEL_PATH, self.SAVE_PATH, fname)}")
                    isValid = False
        else:
            print(f"File {os.path.join(MODEL_PATH, self.SAVE_PATH, fname)} is not existed!")
            isValid = False
        return isValid

    def save_model(self, key):
        mname = key + self.keyapps[0]
        fname = self.mname_header + mname
        if not os.path.isdir(os.path.join(MODEL_PATH, self.SAVE_PATH)):
            os.mkdir(os.path.join(MODEL_PATH, self.SAVE_PATH))
        with open(os.path.join(MODEL_PATH, self.SAVE_PATH, fname), 'wb') as f:
            pickle.dump(self.models[mname], f)

    def save_to_summary(self, performdict):
        fname = self.performance_file
        if os.path.isfile(os.path.join(os.getcwd(), self.SAVE_PATH, fname)):
            df = pd.read_csv(os.path.join(os.getcwd(), self.SAVE_PATH, fname))
            self.performance_df = df.copy()
            self.performance_df = self.performance_df.set_index("key")
        if "key" not in performdict: raise ValueError("performance dict must have 'key' keyword.")
        for key in performdict:
            if key in self.performance_keys[1:len(self.performance_keys)]:
                self.performance_df.loc[performdict["key"], key] = performdict[key]
        self.performance_df.to_csv(os.path.join(os.getcwd(), self.SAVE_PATH, fname), index=True,
                                   float_format=float_format)

    def generate_X(self):
        X = []
        for feature in self.features:
            thisx = self.data.gooddf[feature].to_numpy()
            X.append(thisx)
        X = np.array(X)
        X = X.T
        return X

    def generate_data(self, key, test_size=None, random_state=None):
        if random_state is None: random_state = self.random_state
        if test_size is None: test_size = self.valid_size

        X = self.generate_X()
        y = self.data.gooddf[key].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test, X, y

    def get_predictions(self, key, x):
        mname = key + self.keyapps[0]
        model = self.models[mname]
        preds = model.predict(x)
        return preds

    def plot_predictions(self, ytest, ypred, colorkey=None):
        if colorkey is None:
            colors = None
        else:
            colors = self.data.gooddf[colorkey].to_numpy()
            if len(colors) != len(ytest): colors = None

        fig = plot_linear_xys(ytest, ypred, cmap="jet", colors=colors, style="scatter", plot_xx=True, ShowColorbar=True)
        return fig

    def get_mse_mae(self, X_test, y_test, key, plot_predict=True, savefig=False, colorkey=None,
                    App="_test", loadmodel=False):
        if loadmodel:
            isValid = self.load_model(key)
            if not isValid:
                print("Cannot load " + self.mname_header + key + " model!")
                return
        performdict = {}
        performdict["key"] = key
        y_pred = self.get_predictions(key, X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse = np.sqrt(mse)
        rmse = 100.0 * mse / np.mean(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmae = 100.0 * mae / np.mean(y_pred)
        r2 = get_r2(y_test, y_pred)

        performdict["mse" + App] = mse
        performdict["rmse" + App] = rmse
        performdict["mae" + App] = mae
        performdict["rmae" + App] = rmae
        performdict["r2" + App] = r2
        print("*******************************************")
        print(f"Evaluation of {self.modelname} Gradient Boost regressor _ {key}:")
        print(f"MSE on {App} set: {mse} Relative MSE: {rmse}")
        print(f"MAE on {App} set: {mae} Relative MAE: {rmae} R2:{r2}")
        self.save_to_summary(performdict)
        if plot_predict:
            self.plot_predictions(y_test, y_pred, colorkey=colorkey)
            if savefig:
                outfile = self.outfig_header + key + App + "_predict.png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        print("*******************************************")

    def save_predictions2data(self, key, predictions2data=True, plot_predict_all=False, savefig=False, colorkey=None):
        preds = self.get_predictions(key, self.generate_X())
        outkey = "Predicted_" + key
        if predictions2data:
            df = self.data.gooddf
            df[outkey] = preds
            fname = self.data.fname
            if df.index.name == "CompID":
                df.to_csv(fname, index=True, float_format=float_format)
            else:
                df.to_csv(fname, index=False, float_format=float_format)

        if plot_predict_all:
            y_test = self.data.gooddf[key].to_numpy()
            self.get_mse_mae(self.generate_X(), y_test, key, plot_predict=True, savefig=savefig, colorkey=colorkey,
                             App="_all")

    def visualize_results(self, X_test, y_test, key, **kwargs):
        figsize=(6,4)
        if kwargs and "figsize" in kwargs:
            figsize=kwargs["figsize"]

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        mname = key + self.keyapps[0]
        model = self.models[mname]
        test_score = np.zeros((self.params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_predict(X_test)):
            test_score[i] = mean_squared_error(y_test, y_pred)

        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 1, 1)
        plt.title("Deviance")
        plt.plot(
            np.arange(self.params["n_estimators"]) + 1,
            model.train_score_,
            "b-",
            label="Training Set Deviance",
        )
        plt.plot(
            np.arange(self.params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Boosting Iterations")
        plt.ylabel("Deviance")
        fig.tight_layout()
        plt.show()

    def shap_plot(self, X_test, key, style="beeswarm", max_display=15, figapp="",
                  savefig=True, loadmodel=False, output=False, **kwargs):
        figsize=(6,4)
        fontsize=14
        color_bar = True
        if kwargs and "figsize" in kwargs:
            figsize=kwargs["figsize"]
        if kwargs and "fontsize" in kwargs:
            fontsize=kwargs["fontsize"]
        if kwargs and "color_bar" in kwargs:
            color_bar = kwargs["color_bar"]

        if loadmodel:
            isValid = self.load_model(key)
            if not isValid:
                print("Cannot load " + self.mname_header + key + " model!")
                return
        mname = key + self.keyapps[0]
        model = self.models[mname]

        Xdf = pd.DataFrame(X_test, columns=self.labels)
        explainer = shap.Explainer(model, Xdf)
        shap_values = explainer(Xdf, check_additivity=False)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams.update({'figure.autolayout': True})

        if style.lower() == "bar":
            ax = shap.plots.bar(shap_values, max_display=max_display, show=False)
        elif style.lower() == "heatmap":
            ax = shap.plots.heatmap(shap_values, max_display=max_display, show=False)
        elif style.lower() == "waterfall":
            ax = shap.plots.waterfall(shap_values[0], max_display=str(max_display), show=False)
        else:
            ax = shap.plots.beeswarm(shap_values, max_display=max_display, plot_size=figsize,
                                     color_bar_label='', color_bar=color_bar, show=False)

        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_xlabel("SHAP value", fontsize=fontsize + 2)

        if output:
            nes = shap_values.data.shape[0]
            shap_X = np.hstack([shap_values.data, shap_values.values, shap_values.base_values.reshape([nes, 1])])

            dy_labels = []
            for i in range(len(self.labels)):
                dy_labels.append("DY_D" + self.labels[i])

            cols = self.labels + dy_labels + ["Y_base"]
            shap_df = pd.DataFrame(shap_X, columns=cols)
            shap_df.to_csv(self.mname_header + mname + "_SHAP.csv")

        if color_bar:
            plt.colorbar().ax.tick_params(direction="in", labelsize=0)

        if savefig:
            outfile = self.outfig_header + key
            if isinstance(figapp, str) and len(figapp) > 0:
                outfile += "_" + figapp
            outfile += "_shap.png"
            outfile = os.path.join(self.SAVE_PATH, outfile)
            plt.savefig(outfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def find_importance(self, X_test, y_test, key,
                        n_repeats=10, random_state=None, n_jobs=1,
                        Vertical=False, savefig=False, figapp=None, **kwargs):
        fontsize=14
        if kwargs and "fontsize" in kwargs:
            fontsize=kwargs["fontsize"]

        mname = key + self.keyapps[0]
        model = self.models[mname]
        rank = model.feature_importances_

        if self.Importance_mode is None:
            return rank
        else:
            nsubplot = len(self.Importance_mode)
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"]

            fig = plt.figure(figsize=(6 * nsubplot, 6))
            for iplot in range(len(self.Importance_mode)):
                plotid = self.Importance_mode[iplot]
                ax = plt.subplot(1, nsubplot, iplot + 1)
                if plotid == 0:
                    sorted_idx = np.argsort(rank)
                    pos = np.arange(sorted_idx.shape[0]) + 0.5
                    if Vertical:
                        plt.bar(pos, rank[sorted_idx], align="center")
                        plt.xticks(pos, np.array(self.labels)[sorted_idx])
                    else:
                        plt.barh(pos, rank[sorted_idx], align="center")
                        plt.yticks(pos, np.array(self.labels)[sorted_idx])
                    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
                    #plt.title("Feature Importance(MDI)_" + key, fontsize=fontsize+2)
                else:
                    result = permutation_importance(
                        model, X_test, y_test,
                        n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
                    sorted_idx = result.importances_mean.argsort()
                    plt.boxplot(
                        result.importances[sorted_idx].T,
                        vert=Vertical,
                        labels=np.array(self.labels)[sorted_idx],
                    )
                    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
                    #plt.title("Permutation Importance_" + key, fontsize=fontsize+2)
            fig.tight_layout()

            if savefig:
                outfile = self.outfig_header + key + self.keyapps[0]
                if figapp is None:
                    outfile = outfile + ".png"
                else:
                    outfile = outfile + "_" + figapp + ".png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
                fig = None
            else:
                plt.show()
        return rank

    def pdp_plot_settings(self, features):
        subfeatures = []
        nfeature = len(features)
        finfo = copy.deepcopy(features)
        if nfeature % 2 == 1 and nfeature > 3 and nfeature != 9:
            finfo = np.append(finfo, finfo[0])
            nfeature += 1
        if nfeature <= 3:
            ncols = [1]
            subfeatures.append(finfo)
        elif nfeature % 2 == 0 and nfeature < 13:
            if nfeature >= 12:
                ncols = [4]
            elif nfeature >= 10:
                ncols = [5]
            elif nfeature >= 8:
                ncols = [4]
            elif nfeature >= 6:
                ncols = [2]
            elif nfeature > 3:
                ncols = [2]
            subfeatures.append(finfo)
        else:
            if nfeature > 18:
                raise ValueError("Too many feature to plot!")
            elif nfeature == 18:
                ncols = [3, 3]
                iend = 9
            elif nfeature == 17:
                ncols = [3, 4]
                iend = 9
            elif nfeature == 16:
                ncols = [4, 4]
                iend = 8
            elif nfeature == 15:
                ncols = [3, 2]
                iend = 9
            elif nfeature == 14:
                ncols = [4, 2]
                iend = 8
            elif nfeature >= 13:
                ncols = [3, 2]
                iend = 9
            elif nfeature >= 11:
                ncols = [1, 4]
                iend = 3
            elif nfeature >= 9:
                ncols = [3]
                iend = nfeature
            elif nfeature >= 7:
                ncols = [1, 2]
                iend = 3
            elif nfeature >= 5:
                ncols = [1, 1]
                iend = 3

            if len(ncols) == 1:
                subfeatures.append(finfo)
            else:
                subfeatures.append(finfo[0:iend])
                subfeatures.append(finfo[iend:nfeature])

        nrows = []
        for i in range(len(subfeatures)):
            nsub = len(subfeatures[i])
            ncol = ncols[i]
            nrows.append(int(nsub / ncol))
        return subfeatures, ncols, nrows

    def PDP_settings(self, ifeatures):
        self.common_params = {
            "subsample": self.subsample,
            "n_jobs": 1,
            "grid_resolution": 20,
            "random_state": 0,
        }

        self.features_info = {
            # features of interest
            "features": ifeatures,
            "feature_names": self.labels,
            # type of partial dependence plot
            "kind": "average",
            # information regarding categorical features
            "categorical_features": None,
        }

    def find_partialdependence(self, X_train, key, savefig=False, figapp=None, PDP_output=False, **kwargs):
        fontsize=14
        if kwargs and "fontsize" in kwargs:
            fontsize=kwargs["fontsize"]

        mname = key + self.keyapps[0]
        model = self.models[mname]

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        basesize = 2.67

        if self.feature_type == "default":
            line_kw = {"linewidth": 2, "color": "tab:blue"}
        else:
            if X_train.shape[0] < 10000:
                line_kw = {"linewidth": 0, "color": "tab:blue", "marker": "o", "markersize": 8}
            else:
                line_kw = {"linewidth": 2, "color": "tab:blue"}

        if self.nsplit == 1:
            features4plot = []
            features4plot.append(self.features4plot)
        else:
            features4plot = []
            features4plot.append(np.arange(1))
            features4plot.append(np.arange(1, len(self.features4plot)))

        cols = []
        Xdf = []
        for isplit in range(len(features4plot)):
            feature4plot = features4plot[isplit]
            subfeatures, ncolss, nrowss = self.pdp_plot_settings(feature4plot)
            for iplot in range(len(subfeatures)):
                subfeature = subfeatures[iplot]
                nrows = nrowss[iplot]
                ncols = ncolss[iplot]
                self.PDP_settings(subfeature)

                _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(1.25 * basesize * ncols, basesize * nrows),
                                     constrained_layout=True)

                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_train,
                    **self.features_info,
                    line_kw=line_kw,
                    ax=ax,
                    **self.common_params,
                )
                if self.feature_type == "default":
                    #xmaxs = 0.8 * np.ones(len(self.features))
                    #xmins = 0.1 * np.ones(len(self.features))
                    xmaxs = np.max(X_train, axis=0)
                    xmins = np.min(X_train, axis=0)
                else:
                    xmaxs = np.max(X_train, axis=0)
                    xmins = np.min(X_train, axis=0)

                thisax = display.axes_
                thisax = thisax.reshape([nrows, ncols])
                for i in range(len(subfeature)):
                    irow = int(i / ncols)
                    icol = i - irow * ncols
                    ifeat = subfeature[i]
                    if "CONC" in self.features[ifeat].upper():
                        xmin = 0.1
                        xmax = xmaxs[ifeat]
                    else:
                        xmin = xmins[ifeat]
                        xmax = xmaxs[ifeat]
                    thisax[irow, icol].set_xlim(xmin, xmax)
                    thisax[irow, icol].set_xlabel(self.labels[ifeat], fontsize=fontsize + 2)
                    thisax[irow, icol].yaxis.label.set_visible(False)
                    plt.setp(thisax[irow, icol].get_yticklabels(), fontsize=fontsize)
                    plt.setp(thisax[irow, icol].get_xticklabels(), fontsize=fontsize)

                    if PDP_output:
                        cols.append(self.labels[ifeat])
                        cols.append("DY_D" + self.labels[ifeat])
                        thislines = thisax[irow, icol].get_lines()
                        xdata = thislines[0].get_xdata()
                        ydata = thislines[0].get_ydata()
                        Xdf.append(xdata)
                        Xdf.append(ydata)

                '''
                display.axes_
                _ = display.figure_.suptitle(
                    ("Partial dependence of " + key + " on each feature"),
                    fontsize=fontsize+4,
                    )
                '''
                if savefig:
                    outfile = self.outfig_header + key + "_PDP" + str(isplit) + "_" + str(iplot)
                    if figapp is None:
                        outfile = outfile + ".png"
                    else:
                        outfile = outfile + "_" + figapp + ".png"
                    outfile = os.path.join(self.SAVE_PATH, outfile)
                    plt.savefig(outfile, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
        if PDP_output:
            Xdf = np.array(Xdf)
            #print(f"cols:{cols} shape:{Xdf.shape}")
            thisdf = pd.DataFrame(Xdf.T, columns=cols)
            thisdf.to_csv(self.mname_header + mname + "_PDP.csv")

    def find_interactive_pdp(self, X_train, key, savefig=False, figapp=None, **kwargs):
        fontsize=14
        if kwargs and "fontsize" in kwargs:
            fontsize=kwargs["fontsize"]

        mname = key + self.keyapps[0]
        model = self.models[mname]

        subfeatures, ncolss, nrowss = self.pdp_plot_settings(self.inter_features)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        basesize = 2.67
        if self.feature_type == "default":
            line_kw = {"linewidth": 2, "color": "tab:blue"}
        else:
            if X_train.shape[0] < 10000:
                line_kw = {"linewidth": 0, "color": "tab:blue", "marker": "o", "markersize": 8}
            else:
                line_kw = {"linewidth": 2, "color": "tab:blue"}

        for iplot in range(len(subfeatures)):
            subfeature = subfeatures[iplot]
            nrows = nrowss[iplot]
            ncols = ncolss[iplot]

            thisinter = []
            for ipair in subfeature:
                thisinter.append(self.inter_features[ipair])
            self.PDP_settings(thisinter)

            _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(1.25 * basesize * ncols, basesize * nrows),
                                 constrained_layout=True)

            display = PartialDependenceDisplay.from_estimator(
                model,
                X_train,
                **self.features_info,
                line_kw=line_kw,
                ax=ax,
                **self.common_params,
            )

            if self.feature_type == "default":
                xmaxs = 0.8 * np.ones(len(self.features))
                xmins = 0.1 * np.ones(len(self.features))
            else:
                xmaxs = np.max(X_train, axis=0)
                xmins = np.min(X_train, axis=0)
            thisax = display.axes_
            thisax = thisax.reshape([nrows, ncols])
            for i in range(len(subfeature)):
                irow = int(i / ncols)
                icol = i - irow * ncols
                isub = subfeature[i]
                ix = self.inter_features[isub][0]
                iy = self.inter_features[isub][1]
                if "CONC" in self.features[ix].upper():
                    xmin = 0.1
                    xmax = xmaxs[ix]
                else:
                    xmin = xmins[ix]
                    xmax = xmaxs[ix]
                thisax[irow, icol].set_xlim(xmin, xmax)
                thisax[irow, icol].set_xlabel(self.labels[ix], fontsize=fontsize + 2)
                if "CONC" in self.features[iy].upper():
                    xmin = 0.1
                    xmax = xmaxs[iy]
                else:
                    xmin = xmins[iy]
                    xmax = xmaxs[iy]
                thisax[irow, icol].set_ylim(xmin, xmax)
                thisax[irow, icol].set_ylabel(self.labels[iy], fontsize=fontsize + 2)
                #thisax[irow, icol].yaxis.label.set_visible(True)
                plt.setp(thisax[irow, icol].get_yticklabels(), fontsize=fontsize)
                plt.setp(thisax[irow, icol].get_xticklabels(), fontsize=fontsize)

            '''
            display.axes_
            _ = display.figure_.suptitle(
                ("Interactive partial dependence of " + key),
                fontsize=fontsize+4,
                )
            '''
            if savefig:
                outfile = self.outfig_header + key + "_interPDP" + str(iplot)
                if figapp is None:
                    outfile = outfile + ".png"
                else:
                    outfile = outfile + "_" + figapp + ".png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def get_regression_model(self, X_train, y_train, key, savemodel=False):
        mname = key + self.keyapps[0]
        if self.modelname == "HGBR":
            model = ensemble.HistGradientBoostingRegressor(**self.params)
            model.fit(X_train, y_train)
        elif self.modelname == "GBR":
            model = ensemble.GradientBoostingRegressor(**self.params)
            model.fit(X_train, y_train)
        else:
            model = MLPRegressor(**self.params).fit(X_train, y_train)

        self.models[mname] = model
        if savemodel: self.save_model(key)
        return model

    def regression_models(self, random_state=None, n_repeats=10,
                          loadmodel=False, savemodel=False, savefig=False,
                          plot_predict=False, colorkey=None, figapp="",
                          Save_Pred2Data=False, plot_predict_all=False, SHAP_data="TEST",
                          SHAP_output=False, PDP_output=False, **kwargs):

        if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
            os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))

        for key in self.keys:
            X_train, X_test, y_train, y_test, X, y = self.generate_data(key, random_state=random_state)
            if loadmodel:
                isValid = self.load_model(key)
            else:
                isValid = False
            if isValid:
                self.get_mse_mae(X_test, y_test, key,
                                 plot_predict=plot_predict, savefig=savefig, colorkey=colorkey, App="_test")
            else:
                self.get_regression_model(X_train, y_train, key, savemodel=savemodel)
                self.get_mse_mae(X_test, y_test, key,
                                 plot_predict=plot_predict, savefig=savefig, colorkey=colorkey, App="_test")

            if self.Visualize_Results:
                self.visualize_results(X_test, y_test, key, **kwargs)

            if self.SHAP_Plot:
                if SHAP_data.upper() == "ALL":
                    self.shap_plot(X, key, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output, **kwargs)
                elif SHAP_data.upper() == "TRAIN":
                    self.shap_plot(X_train, key, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output, **kwargs)
                else:
                    self.shap_plot(X_test, key, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output, **kwargs)

            if self.Importance_mode is None:
                pass
            else:
                self.find_importance(X_test, y_test, key,
                                     n_repeats=n_repeats, random_state=random_state,
                                     Vertical=self.Vertical, savefig=savefig, figapp=figapp)
            if self.Find_PDP:
                self.find_partialdependence(X_train, key, savefig=savefig, figapp=figapp, PDP_output=PDP_output,
                                            **kwargs)
            if self.Find_interPDP:
                self.find_interactive_pdp(X_train, key, savefig=savefig, figapp=figapp, **kwargs)
            if Save_Pred2Data or plot_predict_all:
                self.save_predictions2data(key,
                                           predictions2data=Save_Pred2Data, plot_predict_all=plot_predict_all,
                                           savefig=savefig, colorkey=colorkey)

    def kfold_crossvalidation(self, n_splits, n_repeats=1, style="KFold", random_state=None, find_outfliers=False):
        from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
        if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
            os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))

        kfold_key_keys = ["imodel", "score",
                          "imax_abs", "abs_error", "compstr_abs",
                          "imax_rabs", "relative_error", "compstr_rabs",
                          "imax_std", "standardized_error", "compstr_std"
                          ]

        kfold_keys = ["key", "R2_AVG_ALL", "R2_STDEV"]
        kfold_df = pd.DataFrame(columns=kfold_keys)
        keys = []
        R2s = []
        R2stds = []
        print(f"length of original data:{len(self.data.gooddf)}")
        for key in self.keys:
            X = self.generate_X()
            y = self.data.gooddf[key].to_numpy()

            if style[0:3].upper() == "STR":
                rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                y4split = np.ones(len(y))
            else:
                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                y4split = copy.deepcopy(y)

            imax_abs = 0
            abs_error = "INF"
            compstr_abs = "NA"

            imax_rabs = 0
            relative_error = "INF"
            compstr_rabs = "NA"

            imax_std = 0
            standardized_error = "INF"
            compstr_std = "NA"

            imodel = 0
            thisR2s = []
            df = pd.DataFrame(columns=kfold_key_keys)
            for train, test in rkf.split(X, y4split):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                model = self.get_regression_model(X_train, y_train, key, savemodel=False)
                thisscore = model.score(X_test, y_test)
                if find_outfliers:
                    preds = model.predict(X_test)


                    resids = y_test - preds
                    resid_std = np.std(resids)
                    relative_resids = resids / y_test
                    standardized_resids = resids / resid_std

                    iabs = np.argmax(np.abs(resids))
                    abs_error = resids[iabs]
                    test_abs = y_test[iabs]
                    pred_abs = preds[iabs]
                    imax_abs = test[iabs]
                    compstr_abs = self.data.gooddf.iloc[imax_abs]["Composition"]

                    irabs = np.argmax(np.abs(relative_resids))
                    relative_error = relative_resids[irabs]
                    test_rabs = y_test[irabs]
                    pred_rabs = preds[irabs]
                    imax_rabs = test[irabs]
                    compstr_rabs = self.data.gooddf.iloc[imax_rabs]["Composition"]

                    istd = np.argmax(np.abs(standardized_resids))
                    standardized_error = standardized_resids[istd]
                    test_std = y_test[istd]
                    pred_std = preds[istd]
                    imax_std = test[istd]
                    compstr_std = self.data.gooddf.iloc[imax_std]["Composition"]

                    if imodel % 50 == 0:
                        print(f"iabs:{iabs} abs_error:{abs_error}")
                        print(f"test_abs:{test_abs} pred_abs:{pred_abs}")
                        print(f"imax_abs:{imax_abs} compstr_abs:{compstr_abs}")
                        print(f"irabs:{irabs} relative_error:{relative_error}")
                        print(f"test_rabs:{test_rabs} pred_rabs:{pred_rabs}")
                        print(f"imax_rabs:{imax_rabs} compstr_rabs:{compstr_rabs}")
                        print(f"istd:{istd} test_std:{test_std} pred_std:{pred_std}")
                        print(f"imax_std:{imax_std} compstr_std:{compstr_std}")
                        print(f"--- {imodel} --- \n")

                thisdict = {
                            "imodel": imodel, "score": thisscore,
                            "imax_abs": imax_abs, "abs_error": abs_error, "compstr_abs": compstr_abs,
                            "imax_rabs": imax_rabs, "relative_error": relative_error, "compstr_rabs": compstr_rabs,
                            "imax_std": imax_std, "standardized_error": standardized_error, "compstr_std": compstr_std
                             }

                df.loc[len(df)] = thisdict
                thisR2s.append(thisscore)
                if imodel % 10 == 0:
                    print(f"key:{key} xtrain:{X_train.shape} xtest:{X_test.shape} imodel:{imodel} score:{thisscore}")
                imodel += 1

            fname = "KFOLD_" + self.mname_header + key + ".csv"
            df.to_csv(os.path.join(self.SAVE_PATH, fname), index=False)

            thisR2s = np.array(thisR2s)
            thismean = np.mean(thisR2s)
            thisstd = np.std(thisR2s)
            keys.append(key)
            R2s.append(thismean)
            R2stds.append(thisstd)
            print(f"key:{key}  R2_AVG_ALL:{thismean} Stdev:{thisstd}")
            print(f"--- finished Kfold for {key}! ---")
        kfold_df["key"] = keys
        kfold_df["R2_AVG_ALL"] = R2s
        kfold_df["R2_STDEV"] = R2stds
        fname = "KFOLD_" + self.mname_header + "ALL.csv"
        kfold_df.to_csv(os.path.join(self.SAVE_PATH, fname), index=False)
        return kfold_df


class SVMs:
    def __init__(self, data, key, features, thres4cla=[0.0],
                 modelname="SVC", mname_header=None, kernel="Linear", valid_size=0.1,
                 extended=True, norm=False):

        self.SAVE_PATH = "MLR_Models"

        self.data = data
        self.key = key
        self.features = features
        self.thres4cla = thres4cla
        self.nfeatures = len(features)
        norm_features = []
        for i in range(len(self.features)):
            norm_features.append(self.features[i]+"_norm")

        self.modelname = modelname
        if "SVR" in modelname:
            self.style = "REG"
            self.thres4cla = None
            self.nclasses = 0
        else:
            self.style = "CLA"
            try:
                self.nclasses = len(self.thres4cla)
            except:
                print("ERROR: For classification, thres4cla must be a list or 1-d np.array!")
                exit()

        if mname_header is None:
            outfile = get_default_outfile(self.data.fname)
            outfile += "_" + self.modelname
        else:
            outfile = mname_header + "_" + self.modelname

        self.outfig_header = outfile + "_"
        self.mname_header = outfile + "_"

        self.kernel = kernel
        self.valid_size = valid_size

        self.norm = norm


        if self.norm:
            self.norm_features = norm_features
            self.data.normalization(self.data.gooddf, keys=features, extended=extended, Add_norm=True)
        else:
            self.norm_features = self.features[0:len(self.features)]

        self.keyapps = ["_svm", "_mse", "_rmse", "_mae", "_rmae"]
        self.init_models()

        self.performance_keys = ["key", "mse_test", "rmse_test", "mae_test", "rmae_test", "r2_test",
                                 "mse_all", "rmse_all", "mae_all", "rmae_all", "r2_all"]
        self.performance_file = self.mname_header + "performance.csv"
        self.performance_df = pd.DataFrame(columns=self.performance_keys)
        self.performance_df["key"] = [self.key]
        for pkey in self.performance_keys[1:len(self.performance_keys)]:
            self.performance_df[pkey] = [0.0]
        self.performance_df = self.performance_df.set_index("key")

    def init_models(self):
        self.models = {}
        for app in self.keyapps[0:1]:
            thiskey = self.key + app
            self.models[thiskey] = None

    def load_model(self):
        isValid = True
        mname = self.key + self.keyapps[0]
        fname = self.mname_header + mname
        if os.path.isfile(os.path.join(MODEL_PATH, self.SAVE_PATH, fname)):
            with open(os.path.join(MODEL_PATH, self.SAVE_PATH, fname), 'rb') as f:
                try:
                    self.models[mname] = pickle.load(f)
                except:
                    print(f"Fail to load {os.path.join(MODEL_PATH, self.SAVE_PATH, fname)}")
                    isValid = False
        else:
            print(f"File {os.path.join(MODEL_PATH, self.SAVE_PATH, fname)} is not existed!")
            isValid = False
        return isValid

    def save_model(self):
        mname = self.key + self.keyapps[0]
        fname = self.mname_header + mname
        if not os.path.isdir(os.path.join(MODEL_PATH, self.SAVE_PATH)):
            os.mkdir(os.path.join(MODEL_PATH, self.SAVE_PATH))
        with open(os.path.join(MODEL_PATH, self.SAVE_PATH, fname), 'wb') as f:
            pickle.dump(self.models[mname], f)

    def save_to_summary(self, performdict):
        fname = self.performance_file
        if os.path.isfile(os.path.join(os.getcwd(), self.SAVE_PATH, fname)):
            df = pd.read_csv(os.path.join(os.getcwd(), self.SAVE_PATH, fname))
            self.performance_df = df.copy()
            self.performance_df = self.performance_df.set_index("key")
        if "key" not in performdict: raise ValueError("performance dict must have 'key' keyword.")
        for key in performdict:
            if key in self.performance_keys[1:len(self.performance_keys)]:
                self.performance_df.loc[performdict["key"], key] = performdict[key]
        self.performance_df.to_csv(os.path.join(os.getcwd(), self.SAVE_PATH, fname), index=True,
                                   float_format=float_format)

    def generate_X(self):
        X = []
        for feature in self.norm_features:
            thisx = self.data.gooddf[feature].to_numpy()
            X.append(thisx)
        X = np.array(X)
        X = X.T
        return X

    def generate_data(self, key, test_size=None, random_state=None):
        if test_size is None: test_size = self.valid_size

        X = self.generate_X()
        y = self.data.gooddf[key].to_numpy()
        if self.style == "CLA":
            y = np.select([y<self.thres4cla[0], y>=self.thres4cla[0]], [-1, 1])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test, X, y

    def get_predictions(self, x):
        mname = self.key + self.keyapps[0]
        model = self.models[mname]
        preds = model.predict(x)
        return preds

    def plot_predictions(self, ytest, ypred, colorkey=None):
        if colorkey is None:
            colors = None
        else:
            colors = self.data.gooddf[colorkey].to_numpy()
            if len(colors) != len(ytest): colors = None

        fig = plot_linear_xys(ytest, ypred, cmap="jet", colors=colors, style="scatter", plot_xx=True, ShowColorbar=True)
        return fig

    def get_mse_mae(self, X_test, y_test, plot_predict=True, savefig=False, colorkey=None,
                    App="_test", loadmodel=False):
        if loadmodel:
            isValid = self.load_model(self.key)
            if not isValid:
                print("Cannot load " + self.mname_header + self.key + " model!")
                return
        performdict = {}
        performdict["key"] = self.key
        y_pred = self.get_predictions(self.key, X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse = np.sqrt(mse)
        rmse = 100.0 * mse / np.mean(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmae = 100.0 * mae / np.mean(y_pred)
        r2 = get_r2(y_test, y_pred)

        performdict["mse" + App] = mse
        performdict["rmse" + App] = rmse
        performdict["mae" + App] = mae
        performdict["rmae" + App] = rmae
        performdict["r2" + App] = r2
        print("*******************************************")
        print(f"Evaluation of {self.modelname} Gradient Boost regressor _ {self.key}:")
        print(f"MSE on {App} set: {mse} Relative MSE: {rmse}")
        print(f"MAE on {App} set: {mae} Relative MAE: {rmae} R2:{r2}")
        self.save_to_summary(performdict)
        if plot_predict:
            self.plot_predictions(y_test, y_pred, colorkey=colorkey)
            if savefig:
                outfile = self.outfig_header + self.key + App + "_predict.png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        print("*******************************************")

    def save_predictions2data(self, predictions2data=True, plot_predict_all=False, savefig=False, colorkey=None):
        preds = self.get_predictions(self.key, self.generate_X())
        outkey = "Predicted_" + self.key
        if predictions2data:
            df = self.data.gooddf
            df[outkey] = preds
            fname = self.data.fname
            if df.index.name == "CompID":
                df.to_csv(fname, index=True, float_format=float_format)
            else:
                df.to_csv(fname, index=False, float_format=float_format)

        if plot_predict_all:
            y_test = self.data.gooddf[self.key].to_numpy()
            self.get_mse_mae(self.generate_X(), y_test, plot_predict=True, savefig=savefig, colorkey=colorkey,
                             App="_all")

    def shap_plot(self, X_test, style="beeswarm", max_display=15, figapp="", savefig=True, loadmodel=False,
                  output=False, **kwargs):
        figsize=(6,4)
        fontsize=14
        if kwargs and "figsize" in kwargs:
            figsize=kwargs["figsize"]
        if kwargs and "fontsize" in kwargs:
            fontsize=kwargs["fontsize"]

        if loadmodel:
            isValid = self.load_model(self.key)
            if not isValid:
                print("Cannot load " + self.mname_header + self.key + " model!")
                return
        mname = self.key + self.keyapps[0]
        model = self.models[mname]

        Xdf = pd.DataFrame(X_test, columns=self.labels)
        explainer = shap.Explainer(model, Xdf)
        shap_values = explainer(Xdf, check_additivity=False)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams.update({'figure.autolayout': True})
        color_bar = True
        if style.lower() == "bar":
            ax = shap.plots.bar(shap_values, max_display=max_display, show=False)
        elif style.lower() == "heatmap":
            ax = shap.plots.heatmap(shap_values, max_display=max_display, show=False)
        elif style.lower() == "waterfall":
            ax = shap.plots.waterfall(shap_values[0], max_display=str(max_display), show=False)
        else:
            ax = shap.plots.beeswarm(shap_values, max_display=max_display, plot_size=figsize,
                                     color_bar_label='', color_bar=color_bar, show=False)

        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_xlabel("SHAP value", fontsize=fontsize + 2)

        if output:
            nes = shap_values.data.shape[0]
            shap_X = np.hstack([shap_values.data, shap_values.values, shap_values.base_values.reshape([nes, 1])])

            dy_labels = []
            for i in range(len(self.labels)):
                dy_labels.append("DY_D" + self.labels[i])

            cols = self.labels + dy_labels + ["Y_base"]
            shap_df = pd.DataFrame(shap_X, columns=cols)
            shap_df.to_csv(self.mname_header + mname + "_SHAP.csv")

        if not color_bar:
            plt.colorbar().ax.tick_params(direction="in", labelsize=0)

        if savefig:
            outfile = self.outfig_header + self.key
            if isinstance(figapp, str) and len(figapp) > 0:
                outfile += "_" + figapp
            outfile += "_shap.png"
            outfile = os.path.join(self.SAVE_PATH, outfile)
            plt.savefig(outfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_support_vectors(self, X_test, y_test, savefig=True, loadmodel=False,
                  figapp="", output=False, **kwargs):

        fontsize=14
        if kwargs and "fontsize" in kwargs:
            fontsize=kwargs["fontsize"]

        if loadmodel:
            isValid = self.load_model(self.key)
            if not isValid:
                print("Cannot load " + self.mname_header + self.key + " model!")
                return
        mname = self.key + self.keyapps[0]
        model = self.models[mname]

        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            model,
            X_test,
            plot_method="contour",
            colors="k",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot support vectors
        if self.modelname == "LinearSVC":
            decision_function = model.decision_function(X_test)
            support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
            support_vectors = X_test[support_vector_indices]
        else:
            support_vectors = copy.deepcopy(model.support_vectors_)

        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )

        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_xlabel(self.features[0], fontsize=fontsize + 2)
        ax.set_ylabel(self.features[1], fontsize=fontsize + 2)

        if savefig:
            outfile = self.outfig_header + self.key
            if isinstance(figapp, str) and len(figapp) > 0:
                outfile += "_" + figapp
            outfile += ".png"
            outfile = os.path.join(self.SAVE_PATH, outfile)
            plt.savefig(outfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def get_SVM_model(self, X_train, y_train, savemodel=False):
        mname = self.key + self.keyapps[0]
        if self.modelname == "SVC":
            model = svm.SVC(**self.params)
            model.fit(X_train, y_train)
        elif self.modelname == "LinearSVC":
            model = svm.LinearSVC(**self.params)
            model.fit(X_train, y_train)
        else:
            raise ValueError("Unrecognized modelname!")

        self.models[mname] = model
        if savemodel: self.save_model()
        return model

    def SVM_models(self, random_state=None,
                          loadmodel=False, savemodel=False, savefig=False,
                          plot_predict=False, colorkey=None, figapp="",
                          Save_Pred2Data=False, plot_predict_all=False, SHAP_data="TEST",
                          SHAP_output=False):

        if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
            os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))

        for key in self.keys:
            X_train, X_test, y_train, y_test, X, y = self.generate_data(random_state=random_state)
            if loadmodel:
                isValid = self.load_model()
            else:
                isValid = False
            if isValid:
                self.get_mse_mae(X_test, y_test,
                                 plot_predict=plot_predict, savefig=savefig, colorkey=colorkey, App="_test")
            else:
                self.get_SVM_model(X_train, y_train, savemodel=savemodel)
                self.get_mse_mae(X_test, y_test,
                                 plot_predict=plot_predict, savefig=savefig, colorkey=colorkey, App="_test")

            if self.SHAP_Plot:
                if SHAP_data.upper() == "ALL":
                    self.shap_plot(X, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output)
                elif SHAP_data.upper() == "TRAIN":
                    self.shap_plot(X_train, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output)
                else:
                    self.shap_plot(X_test, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output)


            if Save_Pred2Data or plot_predict_all:
                self.save_predictions2data(predictions2data=Save_Pred2Data, plot_predict_all=plot_predict_all,
                                           savefig=savefig, colorkey=colorkey)
class DataFill_NDIP:
    def __init__(self, data, ncompon):
        self.SAVE_PATH = "DATA_FILL"
        self.data = data
        self.ncompon = ncompon
        self.latname = self.data.df.iloc[0]["Crystal"]
        outfile = get_default_outfile(self.data.fname)
        outfile = outfile + "_IP.csv"
        self.default_outfile = outfile

    def fill_one_compspace(self, compspace, condition, keys,
                           validate_data=True, undotype=1, extended=True,
                           screen=["inf", "inf"], xs4eval=None, compid4eval=None,
                           loadmodel=False, savemodel=False):
        screen = np.array(screen)
        if len(screen.shape) == 1:
            onescr = copy.deepcopy(screen)
            screen = []
            for i in range(len(keys)):
                screen.append(onescr)

        gooddf, baddf = self.data.select_by_CS(self.data.df, compspace, condition,
                                               ncompon=self.ncompon, set2gooddf=True)
        thisIP = NDInterpolate(self.data, compspace)
        for ikey in range(len(keys)):
            key = keys[ikey]
            xs, ys, this_xs4eval, this_compid4eval = thisIP.get_data(key, condition,
                                                                     df=gooddf, screen=screen[ikey], extended=extended,
                                                                     norm=False,
                                                                     validate_data=validate_data, undotype=undotype,
                                                                     ncompon=self.ncompon, set2gooddf=False)

            if loadmodel:
                isValid = thisIP.load_model(key)
            else:
                isValid = False
            if not isValid:
                thisIP.get_NDInterpolate_func(xs, ys, key, savemodel=savemodel)
            if xs4eval is None:
                outs = thisIP.Interpolate_data(this_xs4eval, key)
                self.data.modified_value_by_compids(self.data.df, key, this_compid4eval, outs)
            else:
                outs = thisIP.Interpolate_data(xs4eval, key)
                self.data.modified_value_by_compids(self.data.df, key, compid4eval, outs)

    def fill_compspaces(self, compspaces, keys=None,
                        fill_transform=False, update_transform=False, fname4update_trans=None,
                        undotype=1, extended=True, screen=["inf", "inf"],
                        loadmodel=False, savemodel=False, savefile=False, outfile=None):
        if keys == None:
            if undotype == 1:
                keys = eos_keys[0:len(eos_keys)]
                if "hcp" in self.data.fname:
                    keys += hcp_eos_keys
                if "Hmix_TC" in self.data.df.columns:
                    keys += TC_keys
            if undotype == 3:
                update_transform = False
                keys = elastic_keys[0:len(elastic_keys)]
            if "hcp" in self.data.fname:
                fill_transform = False
                update_transform = False
            else:
                if fill_transform:
                    keys += ["bcc2hcp"]
                    update_transform = False
        if "bcc2hcp" in keys:
            update_transform = False

        screen = np.array(screen)
        if len(screen.shape) == 1:
            onescr = copy.deepcopy(screen)
            screen = []
            for i in range(len(keys)):
                screen.append(onescr)

        condition = "exact"
        ics = 0
        for compspace in compspaces:
            self.fill_one_compspace(compspace, condition, keys,
                                    validate_data=True, undotype=undotype, extended=extended,
                                    screen=screen, xs4eval=None, compid4eval=None,
                                    loadmodel=loadmodel, savemodel=savemodel)
            if ics % 50 == 0:
                print(f"Finished {ics} compspaces with {compspace}!")
            ics += 1

        if undotype == 1:
            self.data.df = self.data.compute_lattice_parameter(savefile=False)
            #self.data.df = self.data.update_gmix_cpa(Temp=300.0, savefile=False)

        if update_transform:
            if "bcc" in self.data.fname:
                if fname4update_trans is None:
                    fname = self.default_outfile.replace("bcc", "hcp")
                    fname = os.path.join(DATA_PATH, fname)
                else:
                    fname = fname4update_trans
                self.data.df = self.data.update_transform_from(fname, from_DATA=False, savefile=False)

        if undotype == 3 and "bcc" in self.data.fname:
            self.data.df = self.data.compute_elastic_props(savefile=False)

        if savefile:
            if outfile is None:
                if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
                    os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))
                outfile = os.path.join(self.SAVE_PATH, self.default_outfile)
            self.data.save_to(outfile, df=None)
        return self.data.df


class DataFill_MLR:
    def __init__(self, data, ncompon, modelname="GBR", modeldata=None, mode="eos",
                 fill_transform=True, elements=None, normalization=False):
        self.SAVE_PATH = "DATA_FILL"
        self.data = data
        self.ncompon = ncompon
        self.latname = self.data.df.iloc[0]["Crystal"]
        if "HGBR" in modelname:
            modelname = "HGBR"
        elif "GBR" in modelname:
            modelname = "GBR"
        else:
            modelname = "MLPR"
        self.modelname = modelname
        self.mode = mode

        if self.mode == "eos":
            self.keys = eos_keys[0:len(eos_keys)]
            if self.latname == "hcp":
                self.keys += hcp_eos_keys
                self.undotype = 1
            elif self.latname == "bcc":
                if fill_transform: self.keys += [transformkey]
                self.undotype = 2
        else:
            self.keys = elastic_keys[0:len(elastic_keys)]
            self.undotype = 3

        outfile = get_default_outfile(self.data.fname)
        outfile = outfile + "_" + self.modelname + ".csv"
        self.default_outfile = outfile

        df1, df2, df3 = self.data.validate_dataframe(df=self.data.df, undotype=self.undotype, set2data=True)
        if modeldata is None:
            self.data.gooddf.to_csv(self.default_outfile, index=True, float_format=float_format)
            modeldata = myData(self.default_outfile, from_DATA=False, Source="INTERNAL")
            os.remove(self.default_outfile)
        self.modeldata = modeldata
        if elements is None:
            elements = data.elements[0:len(data.elements)]
        self.elements = elements
        self.normalization = normalization

        self.model = MLRegressor(self.modeldata, self.keys, mname_header=None,
                                 modelname=self.modelname, features=None, SHAP_Plot=False,
                                 elements=self.elements, normalization=self.normalization)

        undodf = pd.concat([df1, df2, df3])
        undodf.to_csv(outfile, index=True, float_format=float_format)
        self.outdata = myData(outfile, from_DATA=False, Source="INTERNAL")
        self.compstrs = undodf["Composition"].to_numpy()
        self.X_test = self.outdata.get_concs(self.outdata.df)

    def fill_data(self, update_transform=False, fname4update_trans=None, loadmodel=False, savemodel=False,
                  savefile=False, outfile=None):
        for key in self.keys:
            mname = key + self.model.keyapps[0]
            if loadmodel:
                isValid = self.model.load_model(key)
            else:
                isValid = False
            if not isValid:
                X_train, X_test, y_train, y_test, X, y = self.model.generate_data(key, random_state=None)
                self.model.get_regression_model(X_train, y_train, key, savemodel=savemodel)

            #thismodel = self.model.models[mname]
            if self.X_test.shape[0] > 0:
                outs = self.model.get_predictions(key, self.X_test)
                self.outdata.df[key] = outs

        if self.mode == "eos":
            self.outdata.df = self.outdata.compute_lattice_parameter(savefile=False)
            #self.outdata.df = self.outdata.update_gmix_cpa(Temp=300.0, savefile=False)
            if update_transform:
                if fname4update_trans is None:
                    fname4update_trans = self.default_outfile
                    fname4update_trans = fname4update_trans.replace("bcc", "hcp")
                    fname4update_trans = os.path.join(DATA_PATH, fname4update_trans)
                self.outdata.update_transform_from(fname4update_trans, from_DATA=False, savefile=False)
        else:
            if self.latname == "bcc":
                self.outdata.df = self.outdata.compute_elastic_props(savefile=False)

        if savefile:
            if outfile is None:
                if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
                    os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))
                outfile = os.path.join(self.SAVE_PATH, self.default_outfile)
            outdf = pd.concat([self.data.gooddf, self.outdata.df])
            if outdf.index.name == "CompID":
                outdf.to_csv(outfile, index=True, float_format=float_format)
            else:
                outdf.to_csv(outfile, index=False, float_format=float_format)