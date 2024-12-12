import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


def main():
    df = pd.read_csv(
        Path("data/area_history.csv"),
        index_col="time_index",
        parse_dates=True,
        dtype={
            "percent": "float32",
            "area": "float64",
            "hash": "string",
            "area_type": "category",
        },
    )

    area_dinamic = (
        df.dropna()
        .groupby([pd.Grouper(freq="D"), "area_type"])[["area", "percent"]]
        .mean()
        .reset_index()
        .set_index("time_index")
    )

    last_date = df.index.max().strftime("%Y-%m-%d %X")

    occupied_by_ua = (
        df[(df["area_type"] == "other_territories") & (df["hash"] == "#01579b")]
        .groupby(pd.Grouper(freq="D"))[["area", "percent"]]
        .mean()
        .fillna(0)
    )

    aa = area_dinamic[area_dinamic["area_type"] == "occupied_after_24_02_2022"][
        "area"
    ]  # ["percent"]
    aa = pd.DataFrame(
        index=pd.date_range(start=aa.index.min(), end=aa.index.max(), freq="D")
    ).join(aa)
    aa.interpolate(inplace=True)
    aa["prefix"] = 0
    aa.loc[:"2022-11-11", "prefix"] = 1

    aa["area"] = aa["area"].subtract(occupied_by_ua["area"], fill_value=0)

    best_model = None
    for p in range(6):
        for q in range(6):
            mod = ARIMA(aa["area"], order=(p, 1, q)).fit()
            if best_model is None or best_model.aic > mod.aic:
                best_model = mod

    print(best_model.summary())

    fh = 120  # int(365.25 * 5)
    fcst = best_model.get_forecast(fh, alpha=0.05).summary_frame()
    fcst.index = pd.date_range(
        start=aa.index.max() + pd.DateOffset(days=1), periods=fcst.shape[0], freq="D"
    )

    svo_end_alpha = 99.95
    svo_end_km_ration = 0.01
    hh = fcst[["mean", "mean_se"]].diff() / fcst[["mean", "mean_se"]].abs()
    end_svo = hh[
        (hh["mean_se"] <= 1 - svo_end_alpha / 100)
        & (hh["mean"] <= svo_end_km_ration / 100)
    ].index.min()
    fcst = fcst[:end_svo]
    fh = 90

    offset = 40
    max_idx = aa.iloc[-365 - offset : -offset]["area"].idxmax()
    max_val = round(aa.iloc[-365 - offset : -offset]["area"].max() / 1000, 1)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax = axs[0]
    sns.lineplot(aa["area"] / 1000, ax=ax, label="Факт")
    sns.lineplot(fcst["mean"] / 1000, ls="--", ax=ax, label="Ожидание")
    fill_95p = ax.fill_between(
        fcst.index,
        fcst["mean_ci_lower"] / 1000,
        fcst["mean_ci_upper"] / 1000,
        alpha=0.2,
        color="grey",
    )
    fill_95p.set_label("95% дов. интервал")
    ax.legend()
    ax.set(
        xlabel=None,
        ylabel="тыс. км\u00b2",
        title="Территория подконтрольная РФ с начала СВО",
    )

    ax.text(max_idx, max_val, f"{max_val:.1f}", ha="center", va="bottom")

    ax.text(
        aa["area"].index.max(),
        aa["area"].iloc[-1] / 1000,
        f'{aa['area'].iloc[-1]/1000:.1f}',
        ha="center",
        va="bottom",
    )

    ax = axs[1]
    ax.set(
        xlabel=None,
        ylabel="км\u00b2/сутки",
        title="Среднесуточное изменение",
    )
    day_din_area = (
        aa.diff()["2022-11-23":].rolling(5, center=True, min_periods=3).mean()
    )
    sns.lineplot(
        day_din_area["area"],
        ax=ax,
        legend=None,
    )
    bbox = {"boxstyle": "larrow", "fc": "0.8", "alpha": 0.4}
    dy = day_din_area.iloc[-1].values[0]
    dx = day_din_area.index.max()
    ax.annotate(
        f"{dy:.2f}",
        (dx + datetime.timedelta(days=16 + int(fh / 60)), dy),
        # xytext=(-2, 1),
        bbox=bbox,
        va="center",
        ha="left",
    )
    for ax in axs:
        ax.grid(ls=":", lw=0.5)
    fig.tight_layout()
    fig.text(
        0,
        0,
        f"Источник: deepstatemap от {last_date}",
        fontdict={"size": 8},
        alpha=0.45,
    )
    Path("img/").mkdir(exist_ok=True)
    fig.savefig(Path("img/area.png"), format="png", dpi=300)


if __name__ == "__main__":
    sys.exit(main())
