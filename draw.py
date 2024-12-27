import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.tsatools import add_trend


def append_trend(X: pd.DataFrame, date, trend_name: str) -> pd.DataFrame:
    if X.index.min() <= date <= X.index.max():
        X[trend_name] = 0
        X.loc[date:, trend_name] = np.arange(len(X.loc[date:, trend_name]))
        return X
    else:
        raise ValueError(
            f"Date {date} is out of range [{X.index.min()}, {X.index.max()}]"
        )


def main() -> None:
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

    area_dynamic = (
        df.dropna()
        .groupby([pd.Grouper(freq="D"), "area_type"])[["area", "percent"]]
        .mean()
    )

    area_dynamic.reset_index(inplace=True)
    area_dynamic.set_index("time_index", inplace=True)

    last_date = df.index.max().strftime("%Y-%m-%d %X")

    occupied_by_ua = (
        df[(df["area_type"] == "other_territories") & (df["hash"] == "#01579b")]
        .groupby(pd.Grouper(freq="D"))[["area", "percent"]]
        .mean()
    )
    occupied_by_ua.interpolate(inplace=True)

    occupied_by_ru = area_dynamic[
        area_dynamic["area_type"] == "occupied_after_24_02_2022"
    ]["area"]  # ["percent"]
    occupied_by_ru = pd.DataFrame(
        index=pd.date_range(
            start=occupied_by_ru.index.min(), end=occupied_by_ru.index.max(), freq="D"
        )
    ).join(occupied_by_ru)
    occupied_by_ru.interpolate(inplace=True)
    occupied_by_ru["prefix"] = 0
    occupied_by_ru.loc[:"2022-11-11", "prefix"] = 1

    occupied_by_ru["area"] = occupied_by_ru["area"].subtract(
        occupied_by_ua["area"], fill_value=0
    )

    fh = 120  # int(365.25 * 5)
    y = occupied_by_ru.loc["2022-11-12":, "area"]
    X = pd.DataFrame(
        index=pd.date_range(
            y.index.min(), y.index.max() + pd.DateOffset(days=fh), freq="D"
        )
    )
    X = add_trend(X, "ct")

    # Добавляем моментум
    X = append_trend(X, y.index.max() - pd.DateOffset(days=365), "momentum_y")
    X = append_trend(X, y.index.max() - pd.DateOffset(days=31), "momentum_m")
    best_model = None
    bound_factor = 0.995
    pq = [(p, q) for p in range(6) for q in range(6)]
    pq.sort(key=lambda x: x[0] + x[1])
    for p, q in pq:
        mod = ARIMA(y, exog=X.loc[: y.index.max()], order=(p, 0, q), trend="n").fit()
        if best_model is None or (best_model.aic * bound_factor) > mod.aic:
            best_model = mod

    print(best_model.summary())
    fcst = best_model.get_forecast(
        fh, alpha=0.01, exog=X.loc[y.index.max() + pd.DateOffset(days=1) :]
    ).summary_frame()
    fcst.index = pd.date_range(
        start=y.index.max() + pd.DateOffset(days=1),
        periods=fcst.shape[0],
        freq="D",
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

    offset = 60
    max_idx = occupied_by_ru.iloc[-365 - offset : -offset]["area"].idxmax()
    max_val = round(
        occupied_by_ru.iloc[-365 - offset : -offset]["area"].max() / 1000, 1
    )

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax = axs[0]
    sns.lineplot(occupied_by_ru["area"] / 1000, ax=ax, label="Факт")
    sns.lineplot(fcst["mean"] / 1000, ls="--", ax=ax, label="Ожидание")
    fill_95p = ax.fill_between(
        fcst.index,
        fcst["mean_ci_lower"] / 1000,
        fcst["mean_ci_upper"] / 1000,
        alpha=0.2,
        color="grey",
    )
    fill_95p.set_label("99% дов. интервал")
    ax.legend()
    ax.set(
        xlabel=None,
        ylabel="тыс. км\u00b2",
        title="Территория подконтрольная РФ с начала СВО",
    )

    ax.text(max_idx, max_val, f"{max_val:.1f}", ha="center", va="bottom")

    ax.text(
        occupied_by_ru["area"].index.max(),
        occupied_by_ru["area"].iloc[-1] / 1000,
        f'{occupied_by_ru['area'].iloc[-1]/1000:.1f}',
        ha="center",
        va="bottom",
    )

    ax.text(
        fcst.index.max(),
        fcst["mean"].iloc[-1] / 1000,
        f"{fcst['mean'].iloc[-1]/1000:.1f}",
        ha="left",
        va="center",
        color="darkorange",
    )

    ax = axs[1]
    ax.set(
        xlabel=None,
        ylabel="км\u00b2/сутки",
        title="Среднесуточное изменение",
    )
    day_din_area = (
        occupied_by_ru.diff()["2022-11-23":]
        .rolling(5, center=True, min_periods=3)
        .mean()
    )
    sns.lineplot(
        day_din_area["area"],
        ax=ax,
        legend=None,
    )
    ax.fill_between(
        day_din_area["area"].index,
        0,
        day_din_area["area"],
        color="royalblue",
        alpha=0.1,
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
        f"Источник: DeepStateMap от {last_date}",
        fontdict={"size": 8},
        alpha=0.45,
    )
    Path("img/").mkdir(exist_ok=True)
    fig.savefig(Path("img/area.png"), format="png", dpi=300)


if __name__ == "__main__":
    sys.exit(main())
