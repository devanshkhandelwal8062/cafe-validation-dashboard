"""
Restro Café Idea Validation: Streamlit dashboard
Descriptive and diagnostic analytics on the synthetic customer survey, ending in a sales pipeline.

Run:  pip install streamlit pandas numpy plotly scipy
      streamlit run app.py          (keep cafe_survey_synthetic_data.csv in the same folder)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Restro Café Validation", page_icon="☕", layout="wide")
PRIMARY, ACCENT, MUTED, GOOD, BAD = "#3B6EA5", "#E07B39", "#A7AFB8", "#3A8F5C", "#C2453D"
BANDS = ["Below 10,000", "10,000-20,000", "20,000-35,000", "35,000-50,000", "50,000-75,000", "Above 75,000"]
AGE_MID = {"18-24": 21, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5, "55+": 60}
CALIB = {5: 0.50, 4: 0.25, 3: 0.05, 2: 0.0, 1: 0.0}
FAMILY_ORDER = ["Single / other", "Couple, no children", "Family with children"]


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_raw(file):
    return pd.read_csv(file, keep_default_na=False, na_values=[""])


@st.cache_data
def clean_and_transform(raw):
    HC = [c for c in raw.columns if c.startswith("HC") and c[2].isdigit()]
    IMP = [c for c in raw.columns if c.startswith("Imp_")]
    LIKERT = HC + IMP
    df, log = raw.copy(), []

    def note(step, before, after, action):
        log.append({"Step": step, "Rows removed": before - after, "Rows left": after, "Action": action})

    n = len(df); df = df.drop_duplicates("Respondent_ID")
    note("Duplicates", n, len(df), "Kept first submission per Respondent_ID")
    fix = {"dubai": "Dubai", "dxb": "Dubai", "abu dhabi": "Abu Dhabi", "auh": "Abu Dhabi", "sharjah": "Sharjah", "shj": "Sharjah",
           "ajman": "Ajman", "rak": "Ras Al Khaimah", "ras al khaimah": "Ras Al Khaimah", "fujairah": "Fujairah",
           "uaq": "Umm Al Quwain", "umm al quwain": "Umm Al Quwain"}
    df["Emirate"] = df["Emirate"].str.strip().str.lower().map(fix)
    df["Gender"] = df["Gender"].replace({"F": "Female", "M": "Male", "N/A": "Prefer not to say"})
    note("Inconsistent labels", len(df), len(df), "Standardised Emirate and Gender spellings")
    n = len(df); df = df[df["Survey_Duration_Sec"] >= 120]
    note("Speeders", n, len(df), "Dropped surveys finished in under 2 minutes")
    n = len(df); df = df[df[LIKERT].std(axis=1) > 0]
    note("Straight-liners", n, len(df), "Dropped identical answers on all 15 ratings")
    n = len(df)
    df = df[~((df["Household_Size"] < 1) | (df["Children_Under_12"] >= df["Household_Size"]) | (df["Dining_Out_Freq_Month"] > 31))]
    note("Impossible values", n, len(df), "Household 0, children >= household, > 31 meals out")

    def iqr_out(s, k=3.0):
        q1, q3 = s.quantile([.25, .75]); i = q3 - q1
        return (s < q1 - k * i) | (s > q3 + k * i)
    n = len(df)
    df = df[~(iqr_out(df["Expected_Spend_AED"] / df["Usual_Group_Size"])
              | iqr_out(df["Current_Avg_Spend_AED"] / df["Usual_Group_Size"]).fillna(False))]
    note("Spend outliers", n, len(df), "Per-person spend outside 3 x IQR")
    cap = df["Delivery_Orders_Month"].quantile(.99)
    df["Delivery_Orders_Month"] = df["Delivery_Orders_Month"].clip(upper=cap)
    note("Delivery outliers", len(df), len(df), f"Capped at 99th percentile ({cap:.0f})")
    miss = int(df.isna().sum().sum())
    for c in df.columns[df.isna().any()]:
        df[c] = df[c].fillna(df[c].median() if df[c].dtype.kind in "if" else df[c].mode()[0])
    df["Income_Band_Code"] = df["Income_Band_Code"].round().astype(int)
    df["Monthly_Household_Income_AED"] = df["Income_Band_Code"].map(lambda b: BANDS[b - 1])
    note("Missing values", len(df), len(df), f"Imputed {miss} blank cells (median / mode)")

    t = df.reset_index(drop=True)
    t["Health_Index"] = t[HC].mean(axis=1).round(2)
    t["Current_Spend_Per_Person"] = (t["Current_Avg_Spend_AED"] / t["Usual_Group_Size"]).round(1)
    t["Expected_Spend_Per_Person"] = (t["Expected_Spend_AED"] / t["Usual_Group_Size"]).round(1)
    t["Spend_Uplift_Pct"] = ((t["Expected_Spend_AED"] / t["Current_Avg_Spend_AED"] - 1) * 100).round(1)
    t["Age_Mid"] = t["Age_Group"].map(AGE_MID)
    t["Health_Tier"] = pd.cut(t["Health_Index"], [0, 2.5, 3.75, 5], labels=["Low", "Medium", "High"]).astype(str)
    t["Income_Tier"] = pd.cut(t["Income_Band_Code"], [0, 2, 4, 6], labels=["Below 20k", "20k-50k", "Above 50k"]).astype(str)
    t["Family_Type"] = np.select([t["Children_Under_12"] > 0, t["Marital_Status"] == "Married"],
                                 ["Family with children", "Couple, no children"], "Single / other")
    t["Intent"] = (t["Visit_Likelihood_1to5"] >= 4).astype(int)
    t["Interested"] = (t["Visit_Likelihood_1to5"] >= 3).astype(int)
    t["Accepts_Premium"] = ((t["Intent"] == 1) & (t["Premium_WTP_Pct"] >= 5)).astype(int)
    t["Trial_Prob"] = t["Visit_Likelihood_1to5"].map(CALIB)
    t["Repeat_2plus"] = (t["Expected_Visits_Month"] >= 2).astype(int)
    t["Expected_Monthly_Value_AED"] = (t["Trial_Prob"] * t["Expected_Visits_Month"] * t["Expected_Spend_AED"]).round(1)
    return t, pd.DataFrame(log)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("☕ Restro Café")
up = st.sidebar.file_uploader("Survey CSV (optional)", type="csv", help="Leave empty to use cafe_survey_synthetic_data.csv")
try:
    raw = load_raw(up if up is not None else "cafe_survey_synthetic_data.csv")
except FileNotFoundError:
    st.error("Put cafe_survey_synthetic_data.csv next to app.py, or upload it in the sidebar.")
    st.stop()
data, clean_log = clean_and_transform(raw)

st.sidebar.markdown("### Filters")
def ms(label, col, order=None):
    opts = order or sorted(data[col].unique())
    return st.sidebar.multiselect(label, opts, default=opts)
f_emi = ms("Emirate", "Emirate")
f_fam = ms("Household type", "Family_Type", FAMILY_ORDER)
f_inc = ms("Income tier", "Income_Tier", ["Below 20k", "20k-50k", "Above 50k"])
f_hea = ms("Health tier", "Health_Tier", ["Low", "Medium", "High"])
t = data[data["Emirate"].isin(f_emi) & data["Family_Type"].isin(f_fam) & data["Income_Tier"].isin(f_inc) & data["Health_Tier"].isin(f_hea)]
st.sidebar.caption(f"{len(t):,} of {len(data):,} respondents selected")
if len(t) < 30:
    st.warning("Fewer than 30 respondents match these filters; widen them for reliable results.")
    if len(t) == 0:
        st.stop()

IMP = [c for c in t.columns if c.startswith("Imp_")]
ITEMS = [c for c in t.columns if c.startswith("Item_")]
OFFERS = [c for c in t.columns if c.startswith("Offer_")]
nice = lambda c: c.replace("Imp_", "").replace("Item_", "").replace("Offer_", "").replace("_", " ")

# ----------------------------------------------------------------------------- header
st.title("Restro café: idea validation dashboard")
st.caption("Synthetic survey of UAE residents · family-focused, health-led, mild-luxury café · descriptive and diagnostic analytics ending in a sales pipeline")
k = st.columns(6)
k[0].metric("Respondents (clean)", f"{len(t):,}", f"of {len(raw):,} raw", delta_color="off")
k[1].metric("Would visit (F2 4-5)", f"{t['Intent'].mean():.0%}")
k[2].metric("Median bill / visit", f"AED {t['Expected_Spend_AED'].median():,.0f}",
            f"{t['Expected_Spend_AED'].median() / t['Current_Avg_Spend_AED'].median() - 1:+.0%} vs today")
k[3].metric("Median spend / person", f"AED {t['Expected_Spend_Per_Person'].median():,.0f}",
            f"{t['Expected_Spend_Per_Person'].median() - t['Current_Spend_Per_Person'].median():+.0f} vs today")
nps = ((t["Recommend_0to10"] >= 9).mean() - (t["Recommend_0to10"] <= 6).mean()) * 100
k[4].metric("NPS", f"{nps:+.0f}")
k[5].metric("Expected real trial", f"{t['Trial_Prob'].mean():.0%}", help="Calibrated: 50% of 5s, 25% of 4s, 5% of 3s visit")

tab_q, tab_d, tab_g, tab_p, tab_x = st.tabs(["1 · Data quality", "2 · Descriptive", "3 · Diagnostic", "4 · Sales pipeline", "5 · Data"])

# ----------------------------------------------------------------------------- 1 data quality
with tab_q:
    st.subheader("From raw responses to the analysis set")
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(clean_log, hide_index=True, width="stretch")
    miss = raw.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values()
    fig = px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x": "% missing", "y": ""}, text=miss.round(1).astype(str) + "%",
                 title="Missing values in the raw file", color_discrete_sequence=[PRIMARY])
    c2.plotly_chart(fig, width="stretch")
    removed = len(raw) - len(data)
    st.info(f"**{removed} of {len(raw):,} rows ({removed/len(raw):.1%}) were removed** as duplicates, speeders, straight-liners, impossible values "
            f"or spend typos. Labels, delivery outliers and blanks were fixed instead, because the rest of those answers were valid.")
    c1, c2 = st.columns(2)
    fig = px.histogram(raw, x=raw["Survey_Duration_Sec"] / 60, nbins=60, title="Survey completion time (minutes)",
                       color_discrete_sequence=[PRIMARY], labels={"x": "minutes"})
    fig.add_vline(x=2, line_dash="dash", line_color=BAD, annotation_text="2-min cut-off")
    c1.plotly_chart(fig, width="stretch")
    c1.caption("A small cluster finished in under 2 minutes, too fast to read 41 questions, so they are removed as speeders.")
    ph = (raw["Expected_Spend_AED"] / raw["Usual_Group_Size"]).dropna()
    fig = px.box(x=ph, log_x=True, points="outliers", title="Expected spend per person (AED, log scale)", color_discrete_sequence=[ACCENT])
    c2.plotly_chart(fig, width="stretch")
    c2.caption("Points far right (AED 500+ per person) are mostly extra-zero typos and are removed with a 3 x IQR rule.")

# ----------------------------------------------------------------------------- 2 descriptive
with tab_d:
    st.subheader("Who answered")
    def share_bar(col, title, order=None, color=PRIMARY):
        s = t[col].value_counts(normalize=True).mul(100)
        s = s.reindex([o for o in order if o in s.index]) if order else s.sort_values()
        return px.bar(x=s.values, y=s.index.astype(str), orientation="h", text=s.round(1).astype(str) + "%", title=title,
                      labels={"x": "% of respondents", "y": ""}, color_discrete_sequence=[color])
    c = st.columns(4)
    c[0].plotly_chart(share_bar("Age_Group", "Age", list(AGE_MID)[::-1]), width="stretch")
    c[1].plotly_chart(share_bar("Emirate", "Emirate"), width="stretch")
    c[2].plotly_chart(share_bar("Monthly_Household_Income_AED", "Monthly income (AED)", BANDS[::-1]), width="stretch")
    c[3].plotly_chart(share_bar("Family_Type", "Household type", None, ACCENT), width="stretch")
    st.caption(f"{(t['Age_Group'].isin(['25-34','35-44'])).mean():.0%} are aged 25-44 and {(t['Family_Type']=='Family with children').mean():.0%} "
               f"have children under 12: the core market for a family café.")

    st.subheader("Spend and intent")
    c1, c2 = st.columns(2)
    sp = pd.concat([t[["Current_Spend_Per_Person"]].rename(columns={"Current_Spend_Per_Person": "AED"}).assign(When="Today, at cafés"),
                    t[["Expected_Spend_Per_Person"]].rename(columns={"Expected_Spend_Per_Person": "AED"}).assign(When="Expected at our café")])
    fig = px.histogram(sp, x="AED", color="When", barmode="overlay", nbins=50, opacity=.6, title="Spend per person per visit",
                       color_discrete_map={"Today, at cafés": MUTED, "Expected at our café": PRIMARY})
    c1.plotly_chart(fig, width="stretch")
    c1.caption(f"Median spend per person rises from AED {t['Current_Spend_Per_Person'].median():.0f} today to "
               f"AED {t['Expected_Spend_Per_Person'].median():.0f} at the concept: room for a mild-luxury price point.")
    vc = t["Visit_Likelihood_1to5"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig = px.bar(x=["1 Very unlikely", "2 Unlikely", "3 Unsure", "4 Likely", "5 Very likely"], y=vc.values,
                 text=(vc / vc.sum() * 100).round(0).astype(int).astype(str) + "%", title="Likelihood to visit (F2)",
                 color=["No", "No", "No", "Yes", "Yes"], color_discrete_map={"No": MUTED, "Yes": PRIMARY}, labels={"x": "", "y": "Respondents", "color": "Would visit"})
    c2.plotly_chart(fig, width="stretch")
    c2.caption("Answers are polarised: people mostly either want the concept or reject it; very few are unsure.")

    c1, c2 = st.columns(2)
    m = t[IMP].mean().rename(nice).sort_values()
    fig = px.bar(x=m.values, y=m.index, orientation="h", text=m.round(2), range_x=[1, 5], title="Importance when choosing a café (1-5)",
                 labels={"x": "mean rating", "y": ""}, color_discrete_sequence=[PRIMARY])
    c1.plotly_chart(fig, width="stretch")
    c1.caption(f"Top priority: **{m.index[-1]}** ({m.iloc[-1]:.2f}). Healthy menu scores {m['Healthy Menu']:.2f}, so food must be healthy *and* taste great.")
    fig = px.histogram(t, x="Health_Index", nbins=20, color="Health_Tier", title="Health-consciousness index (C1-C5)",
                       category_orders={"Health_Tier": ["Low", "Medium", "High"]}, color_discrete_map={"Low": MUTED, "Medium": ACCENT, "High": PRIMARY})
    c2.plotly_chart(fig, width="stretch")
    tiers = t["Health_Tier"].value_counts(normalize=True)
    c2.caption(f"High {tiers.get('High',0):.0%} · Medium {tiers.get('Medium',0):.0%} · Low {tiers.get('Low',0):.0%} health-conscious.")

    c1, c2 = st.columns(2)
    ip = t[ITEMS].mean().mul(100).rename(nice).sort_values()
    c1.plotly_chart(px.bar(x=ip.values, y=ip.index, orientation="h", text=ip.round(0).astype(int).astype(str) + "%",
                           title="Menu items usually ordered", labels={"x": "% of respondents", "y": ""}, color_discrete_sequence=[PRIMARY]), width="stretch")
    op = t[OFFERS].mean().mul(100).rename(nice).sort_values()
    c2.plotly_chart(px.bar(x=op.values, y=op.index, orientation="h", text=op.round(0).astype(int).astype(str) + "%",
                           title="Offers that would increase visits", labels={"x": "% of respondents", "y": ""}, color_discrete_sequence=[ACCENT]), width="stretch")
    st.caption(f"{ip.index[-1]} is the anchor item ({ip.iloc[-1]:.0f}%); {op.index[-1].lower()} is the most wanted offer ({op.iloc[-1]:.0f}%).")
    c1, c2 = st.columns(2)
    c1.plotly_chart(share_bar("Preferred_Location", "Preferred location (F1)"), width="stretch")
    c2.plotly_chart(share_bar("Preferred_Channel", "Preferred channel (F6)", None, ACCENT), width="stretch")

# ----------------------------------------------------------------------------- 3 diagnostic
with tab_g:
    st.subheader("Correlations")
    num = {"Health_Index": "Health index", "Income_Band_Code": "Income band", "Children_Under_12": "Children",
           "Dining_Out_Freq_Month": "Dining-out freq", "Usual_Group_Size": "Group size", "Current_Avg_Spend_AED": "Current bill",
           "Imp_Healthy_Menu": "Imp: healthy menu", "Imp_Ambience": "Imp: ambience", "Imp_Kids_Friendly": "Imp: kids-friendly",
           "Imp_Value_For_Money": "Imp: value", "Imp_Offers": "Imp: offers", "Imp_Instagrammable": "Imp: instagram",
           "Premium_WTP_Pct": "Premium WTP %", "Visit_Likelihood_1to5": "Visit likelihood", "Expected_Visits_Month": "Visits / month",
           "Expected_Spend_AED": "Expected bill"}
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, help="Spearman ranks the values; better for 1-5 scales")
    corr = t[list(num)].rename(columns=num).corr(method=method)
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", height=640,
                    title=f"{method.title()} correlation matrix")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    for col, target in [(c1, "Visit likelihood"), (c2, "Expected bill")]:
        r = corr[target].drop(["Visit likelihood", "Expected bill", "Visits / month"]).sort_values()
        fig = px.bar(x=r.values, y=r.index, orientation="h", text=r.round(2), title=f"What moves with {target.lower()}?",
                     color=np.where(r.values > 0, "positive", "negative"), color_discrete_map={"positive": GOOD, "negative": BAD},
                     labels={"x": f"{method} r", "y": "", "color": ""})
        col.plotly_chart(fig, width="stretch")
        col.caption(f"Strongest link: **{r.index[-1]}** (r = {r.iloc[-1]:+.2f}); strongest negative: **{r.index[0]}** (r = {r.iloc[0]:+.2f}).")

    st.subheader("Who wants to visit? Group comparisons")
    c = st.columns(3)
    for col, (var, order) in zip(c, {"Health_Tier": ["Low", "Medium", "High"], "Income_Tier": ["Below 20k", "20k-50k", "Above 50k"],
                                     "Family_Type": FAMILY_ORDER}.items()):
        g = t.groupby(var)["Intent"].agg(["mean", "size"]).reindex([o for o in order if o in t[var].unique()])
        fig = px.bar(x=g.index, y=g["mean"] * 100, text=[f"{v:.0%} (n={n})" for v, n in zip(g["mean"], g["size"])], range_y=[0, 105],
                     title=var.replace("_", " "), labels={"x": "", "y": "% would visit"}, color_discrete_sequence=[PRIMARY])
        col.plotly_chart(fig, width="stretch")
        ct = pd.crosstab(t[var], t["Intent"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            c2v, p, dof, _ = stats.chi2_contingency(ct)
            cv = np.sqrt(c2v / (len(t) * (min(ct.shape) - 1)))
            col.caption(f"Chi-square = {c2v:.1f}, p = {p:.2g}, Cramér's V = {cv:.2f} " + ("(significant)" if p < 0.05 else "(not significant)"))

    c1, c2 = st.columns(2)
    fig = px.box(t, x="Family_Type", y="Expected_Spend_AED", category_orders={"Family_Type": FAMILY_ORDER}, points=False,
                 title="Expected bill by household type", color_discrete_sequence=[PRIMARY], labels={"Family_Type": "", "Expected_Spend_AED": "AED per visit"})
    c1.plotly_chart(fig, width="stretch")
    med = t.groupby("Family_Type")[["Expected_Spend_AED", "Expected_Spend_Per_Person"]].median()
    if "Family with children" in med.index:
        c1.caption(f"Families' median bill: AED {med.loc['Family with children','Expected_Spend_AED']:.0f}, but only "
                   f"AED {med.loc['Family with children','Expected_Spend_Per_Person']:.0f} per person: revenue through volume.")
    fig = px.box(t, x="Health_Tier", y="Premium_WTP_Pct", category_orders={"Health_Tier": ["Low", "Medium", "High"]}, points=False,
                 title="Premium willingness to pay by health tier", color_discrete_sequence=[ACCENT], labels={"Health_Tier": "", "Premium_WTP_Pct": "% premium"})
    c2.plotly_chart(fig, width="stretch")
    wm = t.groupby("Health_Tier")["Premium_WTP_Pct"].mean()
    c2.caption(" · ".join(f"{k}: {wm[k]:.1f}% avg premium" for k in ["Low", "Medium", "High"] if k in wm.index))

    fig = px.scatter(t, x="Current_Avg_Spend_AED", y="Expected_Spend_AED", color="Health_Tier", opacity=.5,
                     category_orders={"Health_Tier": ["Low", "Medium", "High"]}, color_discrete_map={"Low": MUTED, "Medium": ACCENT, "High": PRIMARY},
                     title="Current vs expected bill", labels={"Current_Avg_Spend_AED": "Current bill (AED)", "Expected_Spend_AED": "Expected bill (AED)"})
    mx = max(t["Current_Avg_Spend_AED"].max(), t["Expected_Spend_AED"].max())
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="#333"), name="No change"))
    b1, b0 = np.polyfit(t["Current_Avg_Spend_AED"], t["Expected_Spend_AED"], 1)
    r_ = np.corrcoef(t["Current_Avg_Spend_AED"], t["Expected_Spend_AED"])[0, 1]
    fig.add_trace(go.Scatter(x=[0, mx], y=[b0, b0 + b1 * mx], mode="lines", line=dict(color=BAD, width=2),
                             name=f"Fit: y = {b0:.0f} + {b1:.2f}x (r = {r_:.2f})"))
    st.plotly_chart(fig, width="stretch")
    up_ = t.groupby("Health_Tier")["Spend_Uplift_Pct"].median()
    st.caption("Median change vs today: " + " · ".join(f"{k} health {up_[k]:+.0f}%" for k in ["Low", "Medium", "High"] if k in up_.index)
               + ". Health-conscious customers expect to pay more; low-health customers expect to pay less.")

    items = t[ITEMS].rename(columns=nice)
    fig = px.imshow(items.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-.5, zmax=.5, height=600,
                    title="Menu items ordered together (phi correlation)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Red pairs are ordered together more than chance: a healthy brunch basket (coffee + avocado toast, salad + healthy dessert) "
               "and an indulgent family basket (burger + classic dessert, pasta + mocktail).")

# ----------------------------------------------------------------------------- 4 sales pipeline
with tab_p:
    st.subheader("From survey answers to a sales pipeline")
    stages = pd.Series({
        "Target respondents": len(t), "Interested (F2 ≥ 3)": t["Interested"].sum(), "Intend to visit (F2 ≥ 4)": t["Intent"].sum(),
        "Accept premium (WTP ≥ 5%)": t["Accepts_Premium"].sum(), "Expected to try (calibrated)": t["Trial_Prob"].sum(),
        "Expected repeat (2+ visits/mo)": (t["Trial_Prob"] * t["Repeat_2plus"]).sum()})
    fig = go.Figure(go.Funnel(y=stages.index, x=stages.values.round(0), textinfo="value+percent initial+percent previous",
                              marker=dict(color=[PRIMARY] * 4 + [ACCENT] * 2)))
    fig.update_layout(title="Sales pipeline (blue = stated, orange = calibrated to real behaviour)", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{t['Intent'].mean():.0%} say they will visit, but only about {t['Trial_Prob'].mean():.0%} are expected to really try it: "
               "the biggest leak is between intent and first visit, so launch marketing must drive trial.")

    seg = t.groupby("Family_Type").apply(lambda s: pd.Series({
        "Intent %": s["Intent"].mean() * 100, "Trial %": s["Trial_Prob"].mean() * 100,
        "Repeat %": (s["Trial_Prob"] * s["Repeat_2plus"]).mean() * 100,
        "Share of people %": len(s) / len(t) * 100,
        "Share of value %": s["Expected_Monthly_Value_AED"].sum() / max(t["Expected_Monthly_Value_AED"].sum(), 1) * 100})).reindex(
        [f for f in FAMILY_ORDER if f in t["Family_Type"].unique()])
    c1, c2 = st.columns(2)
    long = seg[["Intent %", "Trial %", "Repeat %"]].reset_index().melt(id_vars="Family_Type", var_name="Stage", value_name="%")
    c1.plotly_chart(px.bar(long, x="Family_Type", y="%", color="Stage", barmode="group", title="Conversion by household type",
                           color_discrete_sequence=[PRIMARY, ACCENT, GOOD], labels={"Family_Type": ""}), width="stretch")
    long2 = seg[["Share of people %", "Share of value %"]].reset_index().melt(id_vars="Family_Type", var_name="Measure", value_name="%")
    c2.plotly_chart(px.bar(long2, x="Family_Type", y="%", color="Measure", barmode="group", title="Share of people vs share of expected revenue",
                           color_discrete_sequence=[MUTED, PRIMARY], labels={"Family_Type": ""}), width="stretch")

    st.subheader("Does one outlet break even?")
    st.caption("Cost figures are estimates for a ~3,000 sq ft, 90-seat community café in Dubai; change them to test scenarios.")
    c = st.columns(4)
    households = c[0].slider("Households within ~15 min", 4000, 30000, 15000, 1000)
    awareness = c[1].slider("Awareness of the café", 0.10, 0.90, 0.50, 0.05, format="%.2f")
    freq_real = c[2].slider("Share of stated visits that happen", 0.2, 1.0, 0.5, 0.05, format="%.2f")
    rent = c[3].number_input("Rent (AED / month)", 10000, 200000, 40000, 5000)
    fixed = rent + 156000 + 18000 + 25000 + 12000 + 15000
    vis = t["Trial_Prob"] * t["Expected_Visits_Month"]
    vph = vis.mean(); bill = (vis * t["Expected_Spend_AED"]).sum() / max(vis.sum(), 1e-9)
    dshare = (vis * t["Preferred_Channel"].map({"Delivery": 1, "Mix of all": 1 / 3}).fillna(0)).sum() / max(vis.sum(), 1e-9)
    var_pct = 0.32 + 0.03 + 0.02 + 0.28 * dshare
    sales = households * awareness * vph * freq_real * bill / 1.05
    ebitda = sales * (1 - var_pct) - fixed
    be = fixed / (1 - var_pct)
    m = st.columns(4)
    m[0].metric("Net sales / month", f"AED {sales:,.0f}")
    m[1].metric("Break-even sales / month", f"AED {be:,.0f}")
    m[2].metric("EBITDA / month", f"AED {ebitda:,.0f}", f"{ebitda / sales:+.0%} margin" if sales else None)
    m[3].metric("Group visits / day", f"{households * awareness * vph * freq_real / 30:,.0f}",
                f"need {be * 1.05 / bill / 30:,.0f}", delta_color="off")
    hh = np.arange(4000, 30001, 1000)
    curve = hh * awareness * vph * freq_real * bill / 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hh, y=curve, name="Net sales", line=dict(color=PRIMARY, width=3)))
    fig.add_hline(y=be, line_dash="dash", line_color=BAD, annotation_text="Break-even")
    fig.add_vline(x=households, line_dash="dot", line_color="#555")
    fig.update_layout(title="Net sales vs catchment size", xaxis_title="Households within ~15 min", yaxis_title="AED / month", height=380)
    st.plotly_chart(fig, width="stretch")
    need = be / (awareness * vph * freq_real * bill / 1.05)
    (st.success if ebitda > 0 else st.error)(
        f"At {awareness:.0%} awareness this café needs about **{need:,.0f} households** nearby to break even. "
        f"With {households:,} households it {'makes' if ebitda > 0 else 'loses'} **AED {abs(ebitda):,.0f} a month**.")

# ----------------------------------------------------------------------------- 5 data
with tab_x:
    st.subheader("Cleaned and transformed data (filtered)")
    st.dataframe(t, width="stretch", height=420)
    st.download_button("Download filtered data (CSV)", t.to_csv(index=False).encode(), "cafe_survey_filtered.csv", "text/csv")
