from html import escape
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from engine import (fit_slr, validate_pairs, coefficient_table, anova_table,
                    descriptive_table, residual_table, diagnostics, predict, holdout)
from samples import sample_data, SAMPLE_NAMES, RBI_SOURCE
from exports import excel_bytes, report_html, TEMPLATE_INSTRUCTIONS
from content import ASSUMPTIONS, LIMITATIONS, GLOSSARY

st.set_page_config(page_title="SLR Learning Lab | The Mountain Path Academy", page_icon="📊",layout="wide")
st.markdown("""<style>
.stApp {background:#f7f9fc} .block-container {padding-top:2rem;max-width:1450px}
h1,h2,h3 {color:#142d4e} h1{font-weight:750!important;letter-spacing:-1.3px}
[data-testid="stSidebar"] {background:#f1f5f9;border-right:2px solid #b5c6d5;color:#152d49}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#152d49;font-weight:700}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#41566b}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.35rem}
[data-testid="stSidebar"] [role="radiogroup"] label {
  background:white;color:#152d49;border:1px solid #8699aa;border-radius:7px;
  padding:.42rem .6rem;margin:0;align-items:center;min-width:0
}
[data-testid="stSidebar"] [role="radiogroup"] label p {color:inherit;white-space:normal;overflow-wrap:anywhere}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#e1edf4}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:#173955;color:white;border-color:#173955;box-shadow:inset 4px 0 #f3cd76
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:white;font-weight:700}
[data-testid="stSidebar"] label:has(input:focus-visible),
[data-testid="stSidebar"] :is(button,input,[role="slider"]):focus-visible {
  outline:3px solid #096888;outline-offset:2px
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:white;color:#152d49;border-color:#8699aa}
[data-testid="stSidebar"] [data-baseweb="select"]:has(input:disabled) > div {background:#e4eaf0;color:#41566b;opacity:1}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {background:#173955;color:white;border-color:#173955}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button p {color:white}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {background:#245171}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce5ef;border-top:3px solid #c4a04c;border-radius:12px;padding:15px}
[data-testid="stMetricValue"]{font-size:1.8rem}
.hero{padding:27px 32px;background:linear-gradient(120deg,#132b48,#1c5575);border-radius:16px;color:white;margin-bottom:22px}
.hero h1{color:white!important;font-size:2.5rem!important;margin:4px 0 8px}.hero p{color:#dceaf7;margin:0}
.eyebrow{font-size:.78rem;letter-spacing:2px;color:#f0d28a;font-weight:700}
.result-panel{background:#132f4c;color:white;border-radius:14px;padding:22px 26px;margin:16px 0}.result-panel strong{display:block;color:#ffda88;font-size:clamp(1.3rem,3vw,2rem);margin:6px 0}.result-panel small{color:#d9e9f3}.variance-bar{display:flex;height:15px;background:#dae2eb;border-radius:8px;overflow:hidden;margin:8px 0}.variance-bar span{background:#176b87}.lesson{background:#eef5fa;border-left:4px solid #176b87;padding:16px 20px;border-radius:0 9px 9px 0;margin:14px 0}
.stButton button,.stDownloadButton button{border-radius:8px}
@media(max-width:700px){.hero h1{font-size:1.8rem!important}.hero{padding:20px}.block-container{padding:1rem}}
</style>""",unsafe_allow_html=True)


def fmt(value, digits=4):
    if value is None or not np.isfinite(value): return "Not available"
    if value != 0 and abs(value) < .0001: return f"{value:.3e}"
    return f"{value:,.{digits}f}"


def show_table(frame):
    config = {c:st.column_config.NumberColumn(format="%.4e" if "p-value" in str(c) or c=="Significance F" else "%.5g")
              for c in frame.select_dtypes(include="number").columns}
    st.dataframe(frame,hide_index=True,use_container_width=True,column_config=config)


def chart_style(fig, height=450):
    fig.update_layout(template="plotly_white",height=height,font=dict(family="Arial",size=14,color="#183651"),
        colorway=["#176b87","#d3a53b","#ad5144","#5b7191"],margin=dict(l=30,r=20,t=45,b=40),
        paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",legend=dict(orientation="h",y=1.13))
    return fig


def check_xlsx(payload):
    try:
        with ZipFile(BytesIO(payload)) as z:
            if sum(i.file_size for i in z.infolist())>60_000_000:
                raise ValueError("The expanded workbook exceeds 60 MB. Upload a smaller file.")
            if any(i.filename.endswith("vbaProject.bin") for i in z.infolist()):
                raise ValueError("Macro-enabled content is not supported. Save a plain .xlsx file.")
    except BadZipFile as e:
        raise ValueError("This is not a valid .xlsx workbook.") from e


@st.cache_data(show_spinner=False)
def template_download():
    frame,*_=sample_data("Advertising and sales")
    frame=frame.rename(columns={"Advertising spend":"X","Sales":"Y"})
    return excel_bytes({"Data":frame},TEMPLATE_INSTRUCTIONS)


with st.sidebar:
    st.markdown("### ▲ The Mountain Path\n**ACADEMY**")
    st.caption("STATISTICS LEARNING STUDIO")
    source_mode=st.radio("1 · Choose your data",["Built-in example","Upload Excel / CSV"])
    sample_name=st.selectbox("Learning example",SAMPLE_NAMES,disabled=source_mode!="Built-in example")
    st.download_button("↓ Excel input template",template_download(),"SLR_Input_Template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    st.divider()
    section=st.radio("2 · Explore your analysis",["Overview","Data & descriptives","Regression & ANOVA",
        "Assumptions & diagnostics","Prediction & validation","Learning guide","Download report"])
    st.divider()
    confidence=st.select_slider("Confidence level",options=[90,95,99],value=95)
    alpha=1-confidence/100
    inference=st.selectbox("Coefficient inference",["Classical","HC3","HAC"],
        help="Classical: exact t/F under normal iid errors. HC3: heteroscedasticity-robust asymptotic inference. HAC: also allows weak serial dependence in correctly ordered data.")
    st.caption("All models include an intercept. Results recalculate automatically.")

st.markdown("""<div class='hero'><div class='eyebrow'>THE MOUNTAIN PATH ACADEMY · APPLIED STATISTICS</div>
<h1>Simple Linear Regression Lab</h1><p>Bring two variables. Understand the relationship, test the assumptions, and explain the result.</p></div>""",unsafe_allow_html=True)

frame, default_x, default_y, default_units, default_order=sample_data(sample_name)
provenance="Synthetic teaching data; fixed random seed 2026."
if sample_name=="GDP and PFCE — official snapshot":
    provenance=f"NSO via RBI Table 158, 31 July 2026 publication. FY2022–23 to FY2025–26. Source: {RBI_SOURCE}"
    if source_mode=="Built-in example":
        st.info("Official GDP–PFCE snapshot · ₹ crore, constant 2022–23 prices · 16 quarters. PFCE is a component of GDP. High fit partly reflects accounting overlap. No live data refresh is implied.")

with st.expander("Data setup — upload, select variables and describe the observations",expanded=source_mode=="Upload Excel / CSV"):
    if source_mode=="Upload Excel / CSV":
        upload=st.file_uploader("Excel (.xlsx) or CSV · one header row, one paired observation per row",type=["xlsx","csv"])
        if upload is None:
            st.info("Upload your data to begin, or choose a built-in example in the sidebar.")
            st.stop()
        if upload.size>10_000_000:
            st.error("Please upload a file smaller than 10 MB.");st.stop()
        payload=upload.getvalue()
        try:
            if upload.name.lower().endswith("xlsx"):
                check_xlsx(payload)
                book=pd.ExcelFile(BytesIO(payload),engine="openpyxl")
                sheets=book.sheet_names
                sheet=st.selectbox("Worksheet",sheets,index=sheets.index("Data") if "Data" in sheets else 0)
                header=st.number_input("Header row number",1,100,1)
                frame=pd.read_excel(book,sheet_name=sheet,header=int(header)-1,nrows=20001)
            else:
                frame=pd.read_csv(BytesIO(payload),nrows=20001)
            if len(frame)<4 or len(frame.columns)<2: raise ValueError("Need at least two columns and four observations.")
            frame.columns=[str(c) for c in frame.columns]
            if not frame.columns.is_unique: raise ValueError("Column names must be unique.")
            candidates=[c for c in frame if pd.to_numeric(frame[c],errors="coerce").notna().sum()>=4]
            if len(candidates)<2: raise ValueError("Need at least two columns with four numeric values each.")
            default_x,default_y=candidates[:2]
            if "X" in candidates and "Y" in candidates: default_x,default_y="X","Y"
            default_order=False
            default_units="user-defined units"
            provenance=f"User upload: {upload.name}. SHA-256: {hashlib.sha256(payload).hexdigest()[:16]}. Worksheet: {sheet if upload.name.lower().endswith('xlsx') else 'CSV'}."
        except (ValueError,KeyError,TypeError,OSError,ImportError) as e:
            st.error(f"Cannot read this dataset: {e}");st.stop()
    else:
        st.caption("Change a sample or upload any numeric X and Y dataset. Input data are not sent to a third-party analytics service.")
    cols=list(frame.columns)
    a,b,c=st.columns(3)
    xcol=a.selectbox("Explanatory variable (X)",cols,index=cols.index(default_x))
    ycol=b.selectbox("Response variable (Y)",cols,index=cols.index(default_y))
    laboptions=["(row number)"]+cols
    labelcol=c.selectbox("Observation label",laboptions,index=laboptions.index("Observation") if "Observation" in cols else 0)
    units=st.text_input("Units / measurement notes",value=default_units)
    time_mode=st.checkbox("These are time-ordered observations",value=default_order)
    order_confirmed=False
    if time_mode:
        st.warning("Keep rows in true chronological order at regular intervals. The app does not automatically sort dates or fiscal-quarter labels. Gaps, duplicates and mixed frequencies require correction before serial tests.")
        order_confirmed=st.checkbox("I confirm rows are chronological and regularly spaced",value=default_order)
    st.caption("Inference: "+inference+". Missing, invalid and infinite numeric values are flagged; zeros and negative values are valid.")
    st.dataframe(frame.head(12),hide_index=True,use_container_width=True)

try:
    data,rejected=validate_pairs(frame,xcol,ycol,None if labelcol=="(row number)" else labelcol)
except ValueError as e:
    st.error(str(e));st.stop()
if len(rejected):
    st.warning(f"{len(rejected)} of {len(frame)} rows have missing or invalid X/Y pairs. No imputation is performed.")
    st.dataframe(rejected,hide_index=True,use_container_width=True)
    if not st.checkbox("Exclude the displayed invalid pairs and analyse the remaining rows",key=f"exclude_{hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()[:12]}_{xcol}_{ycol}"):
        st.stop()
    if time_mode:
        st.error("Excluding rows may create time gaps. Correct and re-upload regularly spaced data, or switch off time-series diagnostics. No serial tests or HAC are run across gaps.")
        st.stop()
ordered=time_mode and order_confirmed
if time_mode and not order_confirmed:
    st.warning("Confirm row order in Data setup before using time-series analysis.")
    st.stop()
if inference=="HAC" and not ordered:
    st.error("HAC needs meaningful chronological order. Enable and confirm time order, or choose Classical / HC3.")
    st.stop()
hac_lags=1
if inference=="HAC":
    hac_lags=st.sidebar.number_input("HAC maximum lag",0,max(0,len(data)-3),min(1,max(0,len(data)-3)))
try:
    fit=fit_slr(data.X,data.Y,alpha=alpha,hac_lags=hac_lags)
except ValueError as e:
    st.error(str(e));st.stop()
coeff=coefficient_table(fit,inference)
anova=anova_table(fit)
desc=descriptive_table(fit.x,fit.y)
detail=residual_table(fit,data.Observation)
diag=diagnostics(fit,ordered)
rho=float(np.corrcoef(fit.x,fit.y)[0,1])
slope_p=float(coeff.loc[1,"p-value"])
direction="higher" if fit.beta[1]>0 else "lower"
metrics={"Observations":fit.n,"R²":fit.r2,"Adjusted R²":fit.adj_r2,"Correlation r":rho,
         "Residual standard error":np.sqrt(fit.mse),"In-sample RMSE":np.sqrt(fit.sse/fit.n),
         "In-sample MAE":np.mean(abs(fit.residual)),"Classical F":fit.f,"Classical F p-value":fit.f_p,
         "PRESS":np.sum(detail["LOOCV error"]**2),"LOOCV RMSE":np.sqrt(np.mean(detail["LOOCV error"]**2))}
cards=st.columns(4)
cards[0].metric("Paired observations",fit.n)
cards[1].metric("R² · sample fit",f"{fit.r2:.1%}")
cards[2].metric("Slope",fmt(fit.beta[1]))
cards[3].metric(f"Slope p-value · {inference}",fmt(slope_p))
if fit.n<30:
    st.caption("Small sample: coefficient uncertainty and diagnostic power deserve particular attention.")

if section=="Overview":
    sign = "+" if fit.beta[1] >= 0 else "−"
    st.markdown(f"""<div class="result-panel"><small>Fitted equation</small>
    <strong>Predicted {escape(str(ycol))} = {fmt(fit.beta[0])} {sign} {fmt(abs(fit.beta[1]))} × {escape(str(xcol))}</strong>
    A one-unit increase in {escape(str(xcol))} corresponds to {fmt(abs(fit.beta[1]))} units {direction} fitted {escape(str(ycol))}.
    <br><small>{escape(str(units))}</small></div>""", unsafe_allow_html=True)
    if not fit.x.min()<=0<=fit.x.max():
        st.caption("X=0 is outside the observed range: interpret the intercept cautiously.")
    grid=np.linspace(fit.x.min(),fit.x.max(),160)
    pred=predict(fit,grid,inference)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=grid,y=pred["Mean CI lower"],mode="lines",line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=grid,y=pred["Mean CI upper"],mode="lines",fill="tonexty",fillcolor="rgba(23,107,135,.12)",line=dict(width=0),name=f"{confidence}% mean CI"))
    fig.add_trace(go.Scatter(x=fit.x,y=fit.y,mode="markers",text=data.Observation,name="Observed",marker=dict(size=9,color="#176b87"),hovertemplate="%{text}<br>X=%{x}<br>Y=%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid,y=pred["Predicted Y"],mode="lines",name="OLS fitted line",line=dict(color="#c49a34",width=3)))
    fig.update_layout(xaxis_title=xcol,yaxis_title=ycol)
    st.plotly_chart(chart_style(fig),use_container_width=True)
    st.subheader("How much variation does the line explain?")
    share = float(np.clip(fit.r2, 0, 1))
    st.markdown(f'<div class="variance-bar" aria-hidden="true"><span style="width:{100*share:.4f}%"></span></div>', unsafe_allow_html=True)
    st.caption(f"Explained by the line  {share:.1%}   ·   Residual variation  {1-share:.1%}")
    st.write(f"The fitted line accounts for **{fit.r2:.2%} of observed Y variation**. The slope test "+
             (f"rejects H₀: β=0 at α={alpha:.2f}." if slope_p<alpha else f"does not reject H₀: β=0 at α={alpha:.2f}."))
    st.info("Statistical significance does not establish causation or practical importance. Inspect assumptions and held-out prediction before drawing a substantive conclusion.")
    if ordered:
        st.warning("These are time series: trends and seasonality can inflate fit. Contemporaneous X does not become a usable forecast unless it is available when the prediction must be made.")
    st.caption("Next: Data & descriptives → Regression & ANOVA → Assumptions & diagnostics → Prediction & validation.")

elif section=="Data & descriptives":
    st.subheader("Know the data before fitting the line")
    st.caption(provenance)
    show_table(desc)
    st.write(f"**Pearson correlation:** {rho:.5f} · **Sample covariance:** {np.cov(fit.x,fit.y,ddof=1)[0,1]:,.5g}")
    st.caption("Quartiles use linear interpolation; standard deviation and variance use n−1. Skewness and excess kurtosis are bias corrected. SE of mean assumes independent observations.")
    fig=make_subplots(rows=1,cols=2,subplot_titles=(f"Distribution of X: {xcol}",f"Distribution of Y: {ycol}"))
    fig.add_trace(go.Histogram(x=fit.x,name="X",marker_color="#176b87"),row=1,col=1)
    fig.add_trace(go.Histogram(x=fit.y,name="Y",marker_color="#c49a34"),row=1,col=2)
    st.plotly_chart(chart_style(fig,350),use_container_width=True)
    st.subheader("Data quality review")
    st.write(f"Uploaded/sample rows: **{len(frame)}** · Analysed pairs: **{fit.n}** · Excluded pairs: **{len(rejected)}** · Duplicate X/Y pairs: **{data.duplicated(['X','Y']).sum()}**")
    st.caption("Duplicates are retained: repeated measurements can be legitimate. Source row refers to data-row position, excluding the header.")
    st.dataframe(data,hide_index=True,use_container_width=True)

elif section=="Regression & ANOVA":
    st.subheader("What does the slope tell us?")
    lower, upper = coeff.loc[1, ["CI lower", "CI upper"]]
    st.markdown(f"**{confidence}% slope interval**  ·  {fmt(lower)} to {fmt(upper)}")
    decision = "Reject the zero-slope hypothesis" if slope_p < alpha else "Insufficient evidence to reject the zero-slope hypothesis"
    st.write(f"{decision} at the {alpha:.0%} significance level using {inference} inference.")
    with st.expander("Model fit and prediction error statistics"):
        show_table(pd.DataFrame(metrics.items(),columns=["Statistic","Value"]))
    st.subheader(f"Coefficient estimates · {inference} inference · {confidence}% intervals")
    show_table(coeff)
    st.caption("All coefficient tests are two-sided against zero. Classical t tests use n−2 degrees of freedom. HC3/HAC use asymptotic normal z tests; small-sample accuracy is not guaranteed.")
    st.write(f"Slope CI: **[{fmt(coeff.loc[1,'CI lower'])}, {fmt(coeff.loc[1,'CI upper'])}]**. "+
             ("It excludes zero." if coeff.loc[1,'CI lower']*coeff.loc[1,'CI upper']>0 else "It includes zero."))
    st.subheader("Analysis of variance · classical OLS")
    show_table(anova)
    st.latex(r"SST=SSR+SSE,\quad F=\frac{SSR/1}{SSE/(n-2)}=t_{\mathrm{slope}}^2")
    st.write(f"**F(1, {fit.df}) = {fmt(fit.f)}**, Significance F = **{fmt(fit.f_p)}**. H₀: the population slope is zero.")
    st.caption("ANOVA always displays the classical F decomposition. It is not silently replaced when robust coefficient inference is selected. The F=t² identity applies to classical inference.")
    if inference!="Classical":
        z=fit.beta[1]/coeff.loc[1,"Standard error"]
        st.info(f"{inference} robust Wald χ²(1) = {z*z:,.5g}; asymptotic p-value = {fmt(stats.chi2.sf(z*z,1))}. Robust covariance changes uncertainty, not the fitted line or R².")
    with st.expander("Compare Classical, HC3 and HAC standard errors"):
        comparisons=[]
        for method in (["Classical","HC3","HAC"] if ordered else ["Classical","HC3"]):
            table=coefficient_table(fit,method);table.insert(0,"Method",method);comparisons.append(table)
        show_table(pd.concat(comparisons,ignore_index=True))
    st.subheader("How OLS finds the line")
    st.latex(r"b=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2},\qquad a=\bar{y}-b\bar{x}")
    st.write(f"Mean X = {fit.x.mean():,.5g}; mean Y = {fit.y.mean():,.5g}; Sxx = {np.sum((fit.x-fit.x.mean())**2):,.5g}; Sxy = {np.sum((fit.x-fit.x.mean())*(fit.y-fit.y.mean())):,.5g}.")
    st.caption("Residual mean and residual–X covariance are zero by OLS construction. That does not prove exogeneity or the absence of omitted-variable bias.")

elif section=="Assumptions & diagnostics":
    st.subheader("Assumptions: assess the design and the residuals")
    st.info("A diagnostic p-value above alpha means insufficient evidence against its null. It never certifies an assumption. Exogeneity and representative sampling require substantive judgement.")
    for title,meaning,assessment,consequence in ASSUMPTIONS:
        with st.expander(title):
            st.write("**Meaning:** "+meaning);st.write("**Assessment:** "+assessment);st.write("**If violated:** "+consequence)
    st.subheader("Diagnostic tests")
    show_table(diag)
    fig=make_subplots(rows=2,cols=2,subplot_titles=("Residuals vs fitted","Normal Q–Q plot","Scale–location","Residuals in row order"))
    fig.add_trace(go.Scatter(x=fit.fitted,y=fit.residual,mode="markers",name="Residuals"),row=1,col=1)
    fig.add_hline(y=0,line_dash="dash",row=1,col=1)
    qq=stats.probplot(fit.residual,dist="norm")
    fig.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",name="Residual quantiles"),row=1,col=2)
    fig.add_trace(go.Scatter(x=qq[0][0],y=qq[1][0]*qq[0][0]+qq[1][1],mode="lines",name="Q–Q reference"),row=1,col=2)
    fig.add_trace(go.Scatter(x=fit.fitted,y=np.sqrt(abs(detail["Standardized residual"])),mode="markers",name="√|standardized residual|"),row=2,col=1)
    fig.add_trace(go.Scatter(x=np.arange(1,fit.n+1),y=fit.residual,mode="lines+markers",name="Ordered residuals"),row=2,col=2)
    fig.update_xaxes(title_text="Fitted Y",row=1,col=1);fig.update_xaxes(title_text="Theoretical normal quantile",row=1,col=2)
    fig.update_xaxes(title_text="Fitted Y",row=2,col=1);fig.update_xaxes(title_text="Row position",row=2,col=2)
    st.plotly_chart(chart_style(fig,680),use_container_width=True)
    st.caption("Look for random scatter around zero, an approximately straight Q–Q pattern, stable spread and no ordered runs. Row-order plots are not time-series evidence unless order is meaningful.")
    if ordered:
        with st.expander("Stationarity screening for time series"):
            st.write("ADF H₀: unit root. KPSS H₀: level stationarity. Results are complementary screens, not a proof against spurious regression; deterministic trends and seasonality need explicit modelling.")
            if fit.n<20:
                st.info("At least 20 observations are required by this app for stationarity screens. This teaching threshold does not guarantee reliable power.")
            else:
                try:
                    from statsmodels.tsa.stattools import adfuller,kpss
                    rows=[]
                    for name,a in [("X",fit.x),("Y",fit.y)]:
                        adf=adfuller(a,regression="c",autolag="AIC")
                        kp=kpss(a,regression="c",nlags="auto")
                        rows.extend([{"Series":name,"Test":"ADF (constant)","Statistic":adf[0],"p-value":adf[1],"Lags":adf[2]},
                            {"Series":name,"Test":"KPSS (level)","Statistic":kp[0],"p-value":kp[1],"Lags":kp[2]}])
                    show_table(pd.DataFrame(rows));st.caption("KPSS tabulated p-values can be bounds (0.01 / 0.10). No automatic differencing is performed.")
                except (ImportError,ValueError,OverflowError) as e:
                    st.warning(f"Stationarity screen unavailable: {e}")
    st.subheader("Outliers, leverage and influence")
    flagged=detail[detail[["Large residual","High leverage","Influential"]].any(axis=1)]
    st.write(f"**{len(flagged)} observations flagged** using |standardized residual|>2, leverage>4/n, or Cook distance>4/n. These are review thresholds, not formal multiple-testing-adjusted decisions.")
    show_table(flagged if len(flagged) else detail.head(8))
    fig=go.Figure(go.Scatter(x=detail.Leverage,y=detail["Standardized residual"],text=detail.Observation,mode="markers",marker=dict(size=10,color=detail["Cook distance"],colorscale="YlOrRd",showscale=True,colorbar=dict(title="Cook D"))))
    fig.update_layout(xaxis_title="Leverage",yaxis_title="Standardized residual")
    st.plotly_chart(chart_style(fig,350),use_container_width=True)
    st.caption("Review source errors and domain context. If exclusions are justified, document them and compare results. This app never automatically removes flagged observations.")
    st.subheader("Limitations that remain even after diagnostics")
    for item in LIMITATIONS: st.markdown("- "+item)

elif section=="Prediction & validation":
    st.subheader("Predict a mean response and a new observation")
    st.caption(f"Observed X range: {fit.x.min():,.6g} to {fit.x.max():,.6g}. Enter X in the same units as your inputs.")
    newx=st.number_input("X for prediction",value=float(fit.x.mean()),format="%.6f")
    prediction=predict(fit,[newx],inference)
    show_table(prediction)
    if bool(prediction.Extrapolation.iloc[0]):
        st.warning("This X is outside the training range. Model intervals do not account for the extra risk of an invalid extrapolated relationship.")
    if inference=="Classical":
        st.info(f"The {confidence}% mean CI describes uncertainty in the average Y. The wider individual prediction interval also includes new-observation noise. Both require the classical model assumptions.")
    else:
        st.info("Robust mean confidence intervals are asymptotic. Individual prediction intervals are intentionally unavailable: robust coefficient covariance does not estimate the conditional variance of a new error.")
    with st.expander("Batch predictions"):
        text=st.text_area("Enter X values separated by commas or line breaks",placeholder="10, 20, 30")
        if text.strip():
            try:
                tokens=text.replace("\n",",").split(",")
                if len(tokens)>1000: raise ValueError("Maximum 1,000 batch inputs.")
                batch=predict(fit,[float(t.strip()) for t in tokens if t.strip()],inference)
                show_table(batch)
                st.download_button("Download batch predictions",excel_bytes({"Predictions":batch}),"SLR_Batch_Predictions.xlsx")
            except ValueError as e: st.error(f"Check your X values: {e}")
    st.subheader("Does the model predict held-out data?")
    st.caption("This evaluation refits on the training subset only. The dashboard fit uses all valid data. A holdout is one noisy estimate, not a guarantee.")
    if ordered:
        st.write("**Chronological holdout:** the latest rows are held out. Earlier observations train the model. This evaluates prediction conditional on observed test X, not an unconditional future forecast.")
    else:
        st.write("**Random holdout:** reproducible shuffle (seed 42). Suitable only when independent rows can be randomly split; use chronological order for time series.")
    fraction=st.slider("Test fraction",.15,.40,.25,.05)
    if fit.n>=12:
        try:
            hm,hrows=holdout(fit.x,fit.y,fraction,ordered)
            show_table(pd.DataFrame(hm.items(),columns=["Holdout statistic","Value"]))
            st.caption("Negative test R² means predictions are worse than the test-set mean reference in that metric. The separate train-mean RMSE is a feasible constant benchmark.")
            show_table(hrows)
        except ValueError as e: st.warning(str(e))
    else: st.info("Holdout evaluation requires at least 12 observations in this app.")
    st.write(f"**LOOCV RMSE:** {metrics['LOOCV RMSE']:,.5g}. Uses leave-one-out refitting identity eᵢ/(1−hᵢ).")
    if ordered: st.warning("LOOCV uses future observations when predicting earlier rows. Use the chronological holdout for time-series validation.")

elif section=="Learning guide":
    st.subheader("A clear route through simple linear regression")
    st.markdown("""1. **Define the question.** Specify Y, X, units, observation and intended population.
2. **Inspect the data.** Check missingness, distributions, timing and the scatter plot.
3. **Fit the line.** OLS chooses intercept and slope to minimize squared residuals.
4. **Measure uncertainty.** Read slope, its confidence interval, t/z test and p-value together.
5. **Read ANOVA.** Separate explained and residual variation; interpret the classical F test.
6. **Assess assumptions.** Combine residual plots, diagnostics and research design.
7. **Validate predictions.** Keep test data separate and respect time order.
8. **Communicate limitations.** Explain what the result can and cannot establish.""")
    for title,text in GLOSSARY.items():
        with st.expander(title): st.write(text)
    st.subheader("Excel equivalents")
    st.code('=SLOPE(Y_range,X_range)\n=INTERCEPT(Y_range,X_range)\n=RSQ(Y_range,X_range)\n=CORREL(X_range,Y_range)\n=STEYX(Y_range,X_range)\n=LINEST(Y_range,X_range,TRUE,TRUE)',language="text")
    st.caption("Data → Data Analysis → Regression reproduces classical OLS with an intercept. Excel LINEST's coefficient ordering differs from this app's intercept-first table.")
    st.subheader("Practice questions")
    st.markdown("""- Why can R² be high while a model has poor causal interpretation?
- Why does a prediction interval exceed a mean confidence interval?
- In SLR, why do the classical slope t test and regression F test have the same p-value?
- If the variance test rejects, what does HC3 change? What does it leave unchanged?
- Why does normality concern errors rather than X and Y?
- How can one influential observation change the slope?""")
    st.subheader("Assumptions and limitations")
    for title,meaning,assessment,consequence in ASSUMPTIONS:
        with st.expander(title): st.write(meaning+"\n\n"+assessment+"\n\n"+consequence)
    for item in LIMITATIONS: st.markdown("- "+item)
    st.subheader("References")
    st.markdown("[NIST regression handbook](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm) · [SciPy statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html) · [Statsmodels diagnostics](https://www.statsmodels.org/stable/diagnostic.html)")

else:
    st.subheader("Take the analysis into your classroom")
    metadata={"X":xcol,"Y":ycol,"Units":units,"Inference":inference,"Alpha":alpha,
        "Confidence level":confidence/100,"Time ordered":ordered,"HAC lags":fit.hac_lags if inference=="HAC" else "Not used",
        "Source rows":len(frame),"Excluded pairs":len(rejected),"Source":provenance,
        "Model":"OLS with intercept, untransformed levels. No outliers automatically removed."}
    tables={"Settings":pd.DataFrame(metadata.items(),columns=["Setting","Value"]),
        "Model statistics":pd.DataFrame(metrics.items(),columns=["Statistic","Value"]),
        "Coefficients":coeff,"ANOVA classical":anova,"Descriptives":desc,"Diagnostics":diag,
        "Residuals and influence":detail,"Data":data,
        "Assumptions":pd.DataFrame(ASSUMPTIONS,columns=["Assumption","Meaning","Assessment","Consequence"]),
        "Limitations":pd.DataFrame({"Limitation":LIMITATIONS})}
    if len(rejected): tables["Excluded rows"]=rejected
    if fit.n>=12:
        try:
            hm,hrows=holdout(fit.x,fit.y,.25,ordered)
            tables["Holdout 25 percent"]=pd.DataFrame(hm.items(),columns=["Statistic","Value"])
            tables["Holdout predictions"]=hrows
        except ValueError: pass
    a,b=st.columns(2)
    a.download_button("↓ Complete Excel analysis",excel_bytes(tables),"SLR_Analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    # Report includes the same tables; assumptions/limitations appended separately by renderer.
    report_tables={k:v for k,v in tables.items() if k not in ["Settings","Assumptions","Limitations"]}
    b.download_button("↓ Printable HTML report",report_html(report_tables,metadata).encode(),"SLR_Report.html","text/html",use_container_width=True)
    st.caption("Open the HTML report in your browser and choose Print → Save as PDF. Excel results are computed snapshots, with settings and methodology included.")
    st.download_button("↓ Formatted Excel input template",template_download(),"SLR_Input_Template.xlsx")
    st.info("The exported holdout uses 25% of rows and records its split separately. Changes to inputs or inference settings require downloading a fresh report.")

st.divider()
st.caption("The Mountain Path Academy · Generic SLR learning tool · All results are conditional on data quality, study design and model assumptions.")
