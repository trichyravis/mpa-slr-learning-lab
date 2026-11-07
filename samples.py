from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RBI_SOURCE = "https://rbi.org.in/Scripts/PublicationsView.aspx?id=23983"


def sample_data(name):
    if name == "GDP and PFCE — official snapshot":
        return pd.read_csv(ROOT/"gdp_pfce.csv"), "PFCE", "GDP", "₹ crore at constant 2022–23 prices", True
    rng = np.random.default_rng(2026)
    x = np.linspace(5, 100, 48)
    noise = rng.normal(0, 9, len(x))
    if name == "Advertising and sales":
        y = 35+2.4*x+noise
        cols = ["Advertising spend", "Sales"]
    elif name == "Curved relationship":
        y = 15+0.04*x*x+noise
        cols = ["Input", "Response"]
    elif name == "Unequal error variance":
        y = 35+2.4*x+rng.normal(size=len(x))*(2+0.5*x)
        cols = ["Input", "Response"]
    elif name == "Influential observation":
        y = 35+2.4*x+noise
        x[-1], y[-1] = 210, 80
        cols = ["Input", "Response"]
    elif name == "Negative relationship":
        y = 350-2.2*x+noise
        cols = ["Price", "Demand"]
    elif name == "No clear relationship":
        y = rng.normal(100, 25, len(x))
        cols = ["Input", "Response"]
    else:
        raise ValueError("Unknown sample")
    frame = pd.DataFrame({"Observation": [f"Case {i+1:02}" for i in range(len(x))],
                          cols[0]: np.round(x,3), cols[1]: np.round(y,3)})
    return frame, *cols, "illustrative units", False


SAMPLE_NAMES = ["Advertising and sales", "Negative relationship", "No clear relationship",
                "Curved relationship", "Unequal error variance", "Influential observation",
                "GDP and PFCE — official snapshot"]
