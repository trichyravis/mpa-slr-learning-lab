# Simple Linear Regression Learning Lab

A generic, Excel-first Streamlit teaching tool by **The Mountain Path Academy**.
Use any numerical explanatory variable X and response Y. OLS always includes an intercept.

## Run locally

Python 3.12 recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy from GitHub

Open https://share.streamlit.io/ and choose Create app, then deploy from GitHub.
Choose this repository, branch `main`, entry point `app.py`, and Python 3.12.
No API keys or secrets are needed. GitHub Pages cannot run a Streamlit Python server.
Streamlit Community Cloud redeploys when the selected GitHub branch changes.

## Classroom workflow

1. Download the formatted Excel input template in the sidebar. It includes Data and Instructions worksheets.
2. Replace the synthetic X/Y sample with your paired observations. Keep headers in row 1.
3. Upload .xlsx or CSV. Select worksheet, header row, X, Y and optional labels.
4. Add units and identify time-ordered data. Confirm chronology and regular spacing.
5. Explore descriptives, fitted relationship, coefficient inference, ANOVA, diagnostics and predictions.
6. Download a formatted Excel analysis and printable HTML report (Print → Save as PDF).

The app generates its template at runtime. A matching SLR_Input_Template.xlsx is included beside app.py.

## Analysis included

- Descriptives: count, mean, median, sample standard deviation/variance, quartiles, IQR, min/max, skewness, excess kurtosis and SE of mean.
- Pearson correlation, covariance, slope/intercept, R²/adjusted R², residual standard error, RMSE/MAE and PRESS/LOOCV.
- Classical coefficient SE, t tests, two-sided p-values and 90/95/99% confidence intervals.
- Full classical ANOVA: df, SS, MS, F and Significance F; explanation of F=t² in SLR.
- HC3 and Bartlett-kernel Newey–West HAC covariance, asymptotic z tests and Wald χ². ANOVA stays explicitly classical.
- Interactive scatter, fitted line/mean CI, distributions, residual-vs-fitted, Q–Q, scale-location, residual sequence and influence charts.
- Shapiro–Wilk, Jarque–Bera, Koenker–Breusch–Pagan, Ramsey RESET; ordered data add Durbin–Watson and lag-1 Breusch–Godfrey.
- ADF and KPSS time-series stationarity screens when n≥20.
- Standardized/externally studentized residuals, leverage, Cook distance and transparent heuristic flags.
- Single/batch prediction, mean CI, classical individual PI and extrapolation warnings.
- Reproducible random holdout for independent observations; chronological holdout for time series.
- Extensive teaching explanations, assumptions, limitations, test nulls, Excel equivalents and classroom questions.

## Statistical conventions and limitations

Classical inference assumes a correctly specified linear mean, zero conditional mean errors,
constant error variance, independent errors and conditional normality for exact finite-sample t/F inference.
OLS calculation does not require normally distributed X or Y. Robust inference is asymptotic.
Robust SEs do not cure bias, omitted variables, clustering, nonstationarity or wrong functional form.
Individual prediction intervals are classical only: robust coefficient covariance does not identify new-error variance.
Non-rejection of diagnostic nulls never verifies assumptions. Exogeneity and sampling validity require study-design judgement.

Analysis uses original, untransformed values. No automatic differencing, seasonal adjustment, log transformation,
sorting or base-year conversion occurs. For trending data, justify and prepare growth/difference models before upload.
Contemporaneous holdout predictions condition on test X; X must be available at forecast time for a usable forecast.
The app blocks constant X/Y, effectively perfect fits, leverage near one and fewer than four observations because
standard inference is undefined or numerically unreliable. Some diagnostics require more data.

Missing/invalid pairs require explicit exclusion acknowledgement. Exclusions in time-series mode block fitting
until gaps are corrected. Duplicates are retained. No outlier deletion or zero imputation occurs.
Uploads are limited to 10 MB, expanded Excel to 60 MB, analysis to 20,000 rows and batch prediction to 1,000 inputs.
Application code reads uploads in server memory and does not save them. Hosting providers still process those files;
use an appropriate hosting arrangement for sensitive information.

### Methods

OLS uses centered/scaled X and is audited against scipy.stats.linregress.
HC3 uses e²/(1−h)²; HAC uses Bartlett weights and the n/(n−2) covariance correction.
Robust tests use standard normal z and chi-square(1) Wald references.
Koenker–Breusch–Pagan uses n times auxiliary R² from squared residuals on intercept and X.
RESET adds squared/cubed fitted powers using an equivalent standardized-X polynomial span for nonzero slopes.
BG drops the first row, regresses residuals on intercept, X and lag-1 residuals, and uses auxiliary-n × R².
Shapiro uses a reproducible random 5,000-residual subset when larger. JB, LM and robust tests are asymptotic.

## Examples and provenance

Six synthetic teaching examples (seed 2026) illustrate positive/negative/weak association, curvature,
unequal variance and influence. Default: Advertising and sales. Synthetic data are not observed business data.

Optional official GDP–PFCE snapshot: 16 quarters FY2022–23 to FY2025–26, constant 2022–23 prices, ₹ crore.
Source: NSO via RBI Table 158, 31 July 2026 publication:
https://rbi.org.in/Scripts/PublicationsView.aspx?id=23983
This is a fixed vintage, not a live feed. PFCE is part of GDP; the relationship is partly mechanical.
Upload newer data as needed and record its source/vintage.

## Tests

```bash
python -m unittest test_engine test_app -v
```

Tests cover independent SciPy estimates, hand covariance calculations, ANOVA/F=t² identities,
explicit leave-one-out refits, intervals, chronological holdout, invalid data, all examples,
Excel round-trip/formula safety and Streamlit navigation. GitHub Actions runs the suite.

## References

- NIST: https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm
- SciPy: https://docs.scipy.org/doc/scipy/reference/stats.html
- Statsmodels: https://www.statsmodels.org/stable/diagnostic.html
- Streamlit: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
