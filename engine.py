"""Auditable SLR calculations. Always includes an intercept. No Streamlit dependency."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Fit:
    x: np.ndarray
    y: np.ndarray
    alpha: float
    beta: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    covariance: np.ndarray
    hc3_covariance: np.ndarray
    hac_covariance: np.ndarray
    leverage: np.ndarray
    sse: float
    ssr: float
    sst: float
    mse: float
    r2: float
    adj_r2: float
    f: float
    f_p: float
    n: int
    df: int
    hac_lags: int


def validate_pairs(frame, xcol, ycol, labelcol=None):
    if xcol == ycol:
        raise ValueError("Choose different columns for X and Y.")
    if len(frame) > 20000:
        raise ValueError("Use at most 20,000 rows per analysis.")
    # Numeric strings are accepted; invalid strings, booleans and infinities are flagged.
    values = frame[[xcol, ycol]].apply(pd.to_numeric, errors="coerce")
    bools = frame[[xcol, ycol]].map(lambda v: isinstance(v, (bool, np.bool_))).any(axis=1)
    good = np.isfinite(values.to_numpy(dtype=float)).all(axis=1) & ~bools.to_numpy()
    clean = values.loc[good].copy()
    clean.columns = ["X", "Y"]
    clean.insert(0, "Observation", frame.loc[good, labelcol].astype(str).to_numpy()
                 if labelcol else (np.flatnonzero(good) + 1).astype(str))
    clean.insert(0, "Source row", np.flatnonzero(good) + 1)
    rejected = frame.loc[~good].copy()
    rejected.insert(0, "Source row", np.flatnonzero(~good) + 1)
    return clean.reset_index(drop=True), rejected


def fit_slr(x, y, alpha=0.05, hac_lags=1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 4:
        raise ValueError("Provide at least four paired observations.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X and Y must contain finite numbers only.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one.")
    n = len(x)
    mx, my = x.mean(), y.mean()
    xc, yc = x-mx, y-my
    sx = float(np.linalg.norm(xc))
    if sx <= np.finfo(float).eps * max(np.linalg.norm(x), 1) * 10:
        raise ValueError("X is constant or numerically indistinguishable from constant.")
    sst = float(yc @ yc)
    if sst <= np.finfo(float).eps**2 * max(float(y@y), 1):
        raise ValueError("Y is constant; correlation, R² and standard inference are undefined.")
    # Center and scale X to avoid ill-conditioned GDP-sized design matrices.
    z = xc/sx
    Z = np.column_stack([np.ones(n), z])
    theta = np.linalg.lstsq(Z, y, rcond=None)[0]
    residual = y - Z @ theta
    sse = float(residual @ residual)
    if sse <= np.finfo(float).eps**2 * max(float(y@y), 1) * 100:
        raise ValueError("The fit is effectively perfect. Inferential diagnostics are undefined; use data with measurable residual variation.")
    df = n-2
    mse = sse/df
    A = np.array([[1, -mx/sx], [0, 1/sx]])
    beta = A @ theta
    inv = np.diag([1/n, 1.0])
    covz = mse*inv
    leverage = 1/n + z*z
    if np.any(1-leverage < 1e-10):
        raise ValueError("An observation has leverage effectively equal to one. Add more distinct X values.")
    hc = inv @ (Z.T @ ((residual/(1-leverage))[:, None]**2 * Z)) @ inv
    lags = int(np.clip(hac_lags, 0, n-3))
    scores = Z*residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags+1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1-lag/(lags+1))*(gamma+gamma.T)
    hac = (n/df) * inv @ meat @ inv
    ssr = float((Z@theta-my) @ (Z@theta-my))
    r2 = 1-sse/sst
    f = ssr/mse
    return Fit(x, y, alpha, beta, Z@theta, residual, A@covz@A.T,
               A@hc@A.T, A@hac@A.T, leverage, sse, ssr, sst, mse,
               r2, 1-(1-r2)*(n-1)/df, f, float(stats.f.sf(f, 1, df)), n, df, lags)


def coefficient_table(fit, method="Classical"):
    cov = {"Classical": fit.covariance, "HC3": fit.hc3_covariance, "HAC": fit.hac_covariance}[method]
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    statistic = fit.beta/se
    if method == "Classical":
        p = 2*stats.t.sf(abs(statistic), fit.df)
        critical = stats.t.ppf(1-fit.alpha/2, fit.df)
        label = "t statistic"
    else:
        p = 2*stats.norm.sf(abs(statistic))
        critical = stats.norm.ppf(1-fit.alpha/2)
        label = "z statistic (asymptotic)"
    return pd.DataFrame({"Term": ["Intercept", "Slope"], "Estimate": fit.beta,
                         "Standard error": se, label: statistic, "p-value": p,
                         "CI lower": fit.beta-critical*se, "CI upper": fit.beta+critical*se})


def anova_table(fit):
    return pd.DataFrame({"Source": ["Regression", "Residual", "Total"],
        "df": [1, fit.df, fit.n-1], "Sum of squares": [fit.ssr, fit.sse, fit.sst],
        "Mean square": [fit.ssr, fit.mse, np.nan], "F": [fit.f, np.nan, np.nan],
        "Significance F": [fit.f_p, np.nan, np.nan]})


def descriptive_table(x, y):
    rows = []
    for name, a in [("X", np.asarray(x)), ("Y", np.asarray(y))]:
        q1, med, q3 = np.quantile(a, [.25, .5, .75])
        rows.append({"Variable": name, "Count": len(a), "Mean": a.mean(), "Median": med,
          "Std deviation (sample)": a.std(ddof=1), "Variance (sample)": a.var(ddof=1),
          "Min": a.min(), "Q1": q1, "Q3": q3, "Max": a.max(), "IQR": q3-q1,
          "Skewness": stats.skew(a, bias=False), "Excess kurtosis": stats.kurtosis(a, bias=False),
          "SE of mean (iid)": stats.sem(a)})
    return pd.DataFrame(rows)


def predict(fit, xnew, method="Classical"):
    xn = np.atleast_1d(np.asarray(xnew, float))
    if not np.isfinite(xn).all():
        raise ValueError("Prediction inputs must be finite.")
    D = np.column_stack([np.ones(len(xn)), xn])
    cov = {"Classical": fit.covariance, "HC3": fit.hc3_covariance, "HAC": fit.hac_covariance}[method]
    var_mean = np.maximum(np.einsum("ij,jk,ik->i", D, cov, D), 0)
    crit = (stats.t.ppf(1-fit.alpha/2, fit.df) if method == "Classical"
            else stats.norm.ppf(1-fit.alpha/2))
    mean = D@fit.beta
    result = pd.DataFrame({"X": xn, "Predicted Y": mean, "Mean CI lower": mean-crit*np.sqrt(var_mean),
           "Mean CI upper": mean+crit*np.sqrt(var_mean),
           "Extrapolation": (xn < fit.x.min()) | (xn > fit.x.max())})
    if method == "Classical":
        width = crit*np.sqrt(fit.mse+var_mean)
        result["Individual PI lower"] = mean-width
        result["Individual PI upper"] = mean+width
    return result


def residual_table(fit, labels=None):
    h, e = fit.leverage, fit.residual
    student = e/np.sqrt(fit.mse*(1-h))
    deleted_mse = (fit.sse-e*e/(1-h))/(fit.n-3)
    external = np.divide(e, np.sqrt(np.maximum(deleted_mse, 0)*(1-h)),
                         out=np.full(fit.n, np.nan), where=deleted_mse > 0)
    cook = e*e/(2*fit.mse)*h/(1-h)**2
    result = pd.DataFrame({"Observation": labels if labels is not None else np.arange(1,fit.n+1),
       "X": fit.x, "Y": fit.y, "Fitted Y": fit.fitted, "Residual": e,
       "Standardized residual": student, "Externally studentized residual": external,
       "Leverage": h, "Cook distance": cook,
       "Large residual": abs(student)>2, "High leverage": h>4/fit.n,
       "Influential": cook>4/fit.n, "LOOCV error": e/(1-h)})
    return result


def _auxiliary_r2(y, design):
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    total = np.sum((y-np.mean(y))**2)
    if total <= np.finfo(float).eps**2 * max(float(y@y),1):
        return np.nan
    return max(0, 1-np.sum((y-fitted)**2)/total)


def diagnostics(fit, ordered=False):
    e, x, n = fit.residual, fit.x, fit.n
    z = (x-x.mean())/x.std()
    result = []
    def add(test, null, statistic=np.nan, p=np.nan, note=""):
        result.append({"Diagnostic": test, "Null / reference": null, "Statistic": statistic,
                       "p-value": p, "Reading": ("Evidence against null" if p < fit.alpha else
                          "Insufficient evidence against null") if np.isfinite(p) else "Not assessed",
                       "Qualification": note})
    if n >= 8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample = e if n <= 5000 else np.random.default_rng(42).choice(e, 5000, replace=False)
            sh = stats.shapiro(sample)
        add("Shapiro–Wilk", "Normally distributed errors", sh.statistic, sh.pvalue,
            "Residual-based approximation; low power in small samples. Fixed random 5,000-row subsample if n>5,000.")
        jb = stats.jarque_bera(e)
        add("Jarque–Bera", "Normal skewness and kurtosis", jb.statistic, jb.pvalue,
            "Asymptotic test; unreliable with small samples.")
        bp = n*_auxiliary_r2(e*e, np.column_stack([np.ones(n),z]))
        add("Koenker–Breusch–Pagan LM", "Constant conditional error variance", bp, stats.chi2.sf(bp,1),
            "Studentized LM = n × auxiliary R²; asymptotic chi-square(1).")
    else:
        add("Normality / variance tests", "Adequate sample required", note="Suppressed below 8 observations.")
    if n >= 10 and len(np.unique(x)) >= 4 and abs(fit.beta[1]) > 1e-12:
        # Equivalent span to intercept, X, squared/cubed fitted values for nonzero slope.
        design = np.column_stack([np.ones(n), z, z*z, z*z*z])
        if np.linalg.matrix_rank(design) == 4:
            er = fit.y-design@np.linalg.lstsq(design, fit.y, rcond=None)[0]
            full_sse = er@er
            if full_sse > 0:
                f = max(0,(fit.sse-full_sse)/2)/(full_sse/(n-4))
                add("Ramsey RESET (powers 2, 3)", "No additional fitted-power terms", f, stats.f.sf(f,2,n-4),
                    "Classical F test; may indicate nonlinearity or omitted structure. Not proof of correct specification.")
        else:
            add("Ramsey RESET", "Correct functional form", note="Auxiliary design is rank deficient.")
    else:
        add("Ramsey RESET", "Correct functional form", note="Needs n≥10, ≥4 unique X values and a nonzero fitted slope.")
    if ordered:
        dw = float(np.sum(np.diff(e)**2)/fit.sse)
        add("Durbin–Watson", "Around 2 under no first-order correlation", dw, note="No universal pass cutoff. Inspect ordered residuals.")
        if n >= 10:
            yy = e[1:]
            aux = np.column_stack([np.ones(n-1), z[1:], e[:-1]])
            if np.linalg.matrix_rank(aux) == 3:
                lm = (n-1)*_auxiliary_r2(yy, aux)
                add("Breusch–Godfrey LM (lag 1)", "No first-order residual autocorrelation", lm,
                    stats.chi2.sf(lm,1), "Drops first observation; asymptotic LM on remaining rows. Requires meaningful order.")
    else:
        add("Serial dependence", "Independent errors", note="Not assessed for unordered cross-sectional rows. Study design still matters.")
    return pd.DataFrame(result)


def holdout(x, y, fraction=.25, ordered=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<12:
        raise ValueError("Holdout needs at least 12 observations.")
    count=max(3,int(np.ceil(len(x)*fraction)))
    indices=np.arange(len(x)) if ordered else np.random.default_rng(42).permutation(len(x))
    train,test=indices[:-count],indices[-count:]
    fit=fit_slr(x[train],y[train])
    predicted=fit.beta[0]+fit.beta[1]*x[test]
    errors=y[test]-predicted
    benchmark=np.full(count,y[train].mean())
    denom=np.sum((y[test]-y[test].mean())**2)
    return {"Training n":len(train),"Test n":count,"Test RMSE":float(np.sqrt(np.mean(errors**2))),
            "Test MAE":float(np.mean(abs(errors))),"Test R²":float(1-errors@errors/denom) if denom>0 else np.nan,
            "Train-mean benchmark RMSE":float(np.sqrt(np.mean((y[test]-benchmark)**2)))}, pd.DataFrame({
               "Original row position":test+1,"Actual Y":y[test],"Predicted Y":predicted,"Error":errors})
