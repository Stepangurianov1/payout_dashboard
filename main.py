import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
from sqlalchemy import create_engine
import plotly.graph_objs as go
from flask_caching import Cache

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
]

app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Payout Dashboard"

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300,
    },
)

DWH = {
    "host": "49.12.21.243",
    "port": 6432,
    "db": "postgres",
    "user": "second_bi_user",
    "pwd": "sdsdGVGYJ12",
    "database": "postgres",
    "password": "sdsdGVGYJ12"
}

engine_dwh = create_engine(
    f"postgresql://{DWH['user']}:{DWH['pwd']}@{DWH['host']}:{DWH['port']}/{DWH['db']}"
)

@cache.memoize(timeout=300)
def load_payout_engine_info_cached():
    query = """
    SELECT *
    FROM cascade.payout_engine_info
    WHERE order_status <> 'new' and order_status is not null
    """
    return pd.read_sql_query(query, engine_dwh)


df = load_payout_engine_info_cached()

ENGINE_COL = "engine"
TIME_COL = "period_start"
CURRENCY_COL = "currency"

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

time_options = [
    {"label": f"{hour:02d}:{minute:02d}", "value": f"{hour:02d}:{minute:02d}"}
    for hour in range(24)
    for minute in (0, 30)
]
default_start_time = "00:00"
default_end_time = "23:30"

if df[TIME_COL].notna().any():
    min_dt = df[TIME_COL].min()
    max_dt = df[TIME_COL].max()
    default_start_date = min_dt.date().isoformat()
    default_end_date = max_dt.date().isoformat()
else:
    today = datetime.utcnow().date().isoformat()
    default_start_date = today
    default_end_date = today


# ------------------------
# Вспомогательные функции агрегации
# ------------------------

def apply_dashboard_filters(
    df_,
    selected_gateways=None,
    selected_currencies=None,
    start_date=None,
    end_date=None,
    start_time="00:00",
    end_time="23:30",
):
    filtered = df_

    if selected_gateways:
        filtered = filtered[filtered[ENGINE_COL].astype(str).isin(selected_gateways)]

    if selected_currencies:
        filtered = filtered[filtered[CURRENCY_COL].astype(str).isin(selected_currencies)]

    if start_date and end_date:
        start_dt = pd.to_datetime(f"{start_date} {start_time}")
        end_dt = pd.to_datetime(f"{end_date} {end_time}")
        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
        filtered = filtered[(filtered[TIME_COL] >= start_dt) & (filtered[TIME_COL] <= end_dt)]

    return filtered

def make_time_agg(df_):
    agg = (
        df_.groupby("slot_15m")
        .size()
        .rename("orders_count")
        .reset_index()
        .sort_values("slot_15m")
    )
    return agg


def make_gateway_conv(df_, metric_mode="take"):
    if df_.empty:
        return pd.DataFrame(
            columns=[
                "engine",
                "status_group",
                "orders",
                "total_orders",
                "pct",
                "conversion_pct",
            ]
        )

    work_df = df_.copy()
    if pd.api.types.is_categorical_dtype(work_df["engine"]):
        work_df["engine"] = work_df["engine"].cat.remove_unused_categories()

    grouped = (
        work_df.groupby("engine", as_index=False, observed=True)
        .agg(
            total_orders=("orders_count_wo_new", "sum"),
            rejected_orders=(
                "orders_count_wo_new",
                lambda s: s[work_df.loc[s.index, "status_from_engine"] == "rejected by engine"].sum(),
            ),
            success_orders=(
                "orders_count_wo_new",
                lambda s: s[
                    (work_df.loc[s.index, "status_from_engine"] != "rejected by engine")
                    & (work_df.loc[s.index, "order_status"] == "success")
                ].sum(),
            ),
        )
    )
    grouped = grouped[grouped["total_orders"] > 0]
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "engine",
                "status_group",
                "orders",
                "total_orders",
                "pct",
                "conversion_pct",
            ]
        )
    grouped["non_rejected_orders"] = grouped["total_orders"] - grouped["rejected_orders"]
    if metric_mode == "success":
        grouped = grouped[grouped["non_rejected_orders"] > 0]
        if grouped.empty:
            return pd.DataFrame(
                columns=["engine", "status_group", "orders", "total_orders", "pct", "conversion_pct"]
            )

        grouped["not_success_orders"] = grouped["non_rejected_orders"] - grouped["success_orders"]
        grouped["success_pct"] = np.where(
            grouped["non_rejected_orders"] > 0,
            grouped["success_orders"] / grouped["non_rejected_orders"] * 100,
            0.0,
        )
        grouped["not_success_pct"] = 100 - grouped["success_pct"]
        grouped["conversion_pct"] = grouped["success_pct"]

        result = pd.concat(
            [
                grouped[["engine", "not_success_orders", "non_rejected_orders", "not_success_pct", "conversion_pct"]]
                .rename(
                    columns={
                        "not_success_orders": "orders",
                        "non_rejected_orders": "total_orders",
                        "not_success_pct": "pct",
                    }
                )
                .assign(status_group="not-success"),
                grouped[["engine", "success_orders", "non_rejected_orders", "success_pct", "conversion_pct"]]
                .rename(
                    columns={
                        "success_orders": "orders",
                        "non_rejected_orders": "total_orders",
                        "success_pct": "pct",
                    }
                )
                .assign(status_group="success"),
            ],
            ignore_index=True,
        )
        return result[["engine", "status_group", "orders", "total_orders", "pct", "conversion_pct"]]

    grouped["rejected_pct"] = np.where(
        grouped["total_orders"] > 0,
        grouped["rejected_orders"] / grouped["total_orders"] * 100,
        0.0,
    )
    grouped["conversion_pct"] = 100 - grouped["rejected_pct"]

    result = pd.concat(
        [
            grouped[["engine", "rejected_orders", "total_orders", "rejected_pct", "conversion_pct"]]
            .rename(columns={"rejected_orders": "orders", "rejected_pct": "pct"})
            .assign(status_group="rejected"),
            grouped[["engine", "non_rejected_orders", "total_orders", "conversion_pct"]]
            .rename(columns={"non_rejected_orders": "orders", "conversion_pct": "pct"})
            .assign(status_group="non-rejected"),
        ],
        ignore_index=True,
    )
    return result[["engine", "status_group", "orders", "total_orders", "pct", "conversion_pct"]]


def make_gateway_conv_timeseries(df_, metric_mode="take"):
    if df_.empty:
        return pd.DataFrame(
            columns=["period_start", "status_group", "orders", "total_orders", "pct", "conversion_pct"]
        )

    grouped = (
        df_.groupby("period_start", as_index=False)
        .agg(
            total_orders=("orders_count_wo_new", "sum"),
            rejected_orders=(
                "orders_count_wo_new",
                lambda s: s[df_.loc[s.index, "status_from_engine"] == "rejected by engine"].sum(),
            ),
            success_orders=(
                "orders_count_wo_new",
                lambda s: s[
                    (df_.loc[s.index, "status_from_engine"] != "rejected by engine")
                    & (df_.loc[s.index, "order_status"] == "success")
                ].sum(),
            ),
        )
    ).sort_values("period_start")

    grouped = grouped[grouped["total_orders"] > 0]
    if grouped.empty:
        return pd.DataFrame(
            columns=["period_start", "status_group", "orders", "total_orders", "pct", "conversion_pct"]
        )

    grouped["non_rejected_orders"] = grouped["total_orders"] - grouped["rejected_orders"]

    if metric_mode == "success":
        grouped = grouped[grouped["non_rejected_orders"] > 0]
        if grouped.empty:
            return pd.DataFrame(
                columns=["period_start", "status_group", "orders", "total_orders", "pct", "conversion_pct"]
            )

        grouped["not_success_orders"] = grouped["non_rejected_orders"] - grouped["success_orders"]
        grouped["success_pct"] = np.where(
            grouped["non_rejected_orders"] > 0,
            grouped["success_orders"] / grouped["non_rejected_orders"] * 100,
            0.0,
        )
        grouped["not_success_pct"] = 100 - grouped["success_pct"]
        grouped["conversion_pct"] = grouped["success_pct"]

        result = pd.concat(
            [
                grouped[["period_start", "not_success_orders", "non_rejected_orders", "not_success_pct", "conversion_pct"]]
                .rename(
                    columns={
                        "not_success_orders": "orders",
                        "non_rejected_orders": "total_orders",
                        "not_success_pct": "pct",
                    }
                )
                .assign(status_group="not-success"),
                grouped[["period_start", "success_orders", "non_rejected_orders", "success_pct", "conversion_pct"]]
                .rename(
                    columns={
                        "success_orders": "orders",
                        "non_rejected_orders": "total_orders",
                        "success_pct": "pct",
                    }
                )
                .assign(status_group="success"),
            ],
            ignore_index=True,
        )
        return result[["period_start", "status_group", "orders", "total_orders", "pct", "conversion_pct"]]

    grouped["rejected_pct"] = np.where(
        grouped["total_orders"] > 0,
        grouped["rejected_orders"] / grouped["total_orders"] * 100,
        0.0,
    )
    grouped["conversion_pct"] = 100 - grouped["rejected_pct"]

    result = pd.concat(
        [
            grouped[["period_start", "rejected_orders", "total_orders", "rejected_pct", "conversion_pct"]]
            .rename(columns={"rejected_orders": "orders", "rejected_pct": "pct"})
            .assign(status_group="rejected"),
            grouped[["period_start", "non_rejected_orders", "total_orders", "conversion_pct"]]
            .rename(columns={"non_rejected_orders": "orders", "conversion_pct": "pct"})
            .assign(status_group="non-rejected"),
        ],
        ignore_index=True,
    )
    return result[["period_start", "status_group", "orders", "total_orders", "pct", "conversion_pct"]]


def make_trader_conv(df_, gateway, time_window=None):
    df_ = df_[df_["gateway"] == gateway]

    if time_window is not None:
        start_ts, end_ts = time_window
        mask = (df_["timestamp"] >= start_ts) & (df_["timestamp"] < end_ts)
        df_ = df_[mask]

    if df_.empty:
        return pd.DataFrame(columns=["trader", "status", "orders", "pct", "amount"])

    agg = (
        df_.groupby(["trader", "status"])
        .agg(
            orders=("status", "size"),
            amount=("amount", "sum"),
        )
        .reset_index()
    )
    total = agg.groupby("trader")["orders"].transform("sum")
    agg["pct"] = agg["orders"] / total * 100
    return agg


def make_gateway_conv_daily(df_):
    """
    Дневная конверсия по шлюзам: success / fail, нормировано до 100% (как на скрине).
    Возвращает DataFrame с колонками: date, gateway, fail, success (в процентах).
    """
    df_ = df_.copy()
    df_["date"] = df_["timestamp"].dt.floor("D")

    agg = (
        df_.groupby(["date", "gateway", "status"])
        .size()
        .rename("orders")
        .reset_index()
    )
    if agg.empty:
        return pd.DataFrame(columns=["date", "gateway", "fail", "success"])

    total = agg.groupby(["date", "gateway"])["orders"].transform("sum")
    agg["pct"] = agg["orders"] / total * 100

    daily = (
        agg.pivot_table(
            index=["date", "gateway"],
            columns="status",
            values="pct",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # гарантируем наличие обоих столбцов
    for col in ["fail", "success"]:
        if col not in daily.columns:
            daily[col] = 0.0

    return daily


def make_gateway_conv_timeseries_legacy(df_, gateway):
    """
    Почасовая/помежеутковая (по slot_15m) конверсия для одного шлюза:
    success / fail, нормировано до 100%.
    """
    df_ = df_[df_["gateway"] == gateway].copy()
    if df_.empty:
        return pd.DataFrame(columns=["slot_15m", "fail", "success"])

    agg = (
        df_.groupby(["slot_15m", "status"])
        .size()
        .rename("orders")
        .reset_index()
    )

    total = agg.groupby("slot_15m")["orders"].transform("sum")
    agg["pct"] = agg["orders"] / total * 100

    ts = (
        agg.pivot_table(
            index="slot_15m",
            columns="status",
            values="pct",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in ["fail", "success"]:
        if col not in ts.columns:
            ts[col] = 0.0

    ts = ts.sort_values("slot_15m")
    return ts


# ------------------------
# Dash-приложение
# ------------------------

engines = sorted(df[ENGINE_COL].dropna().astype(str).unique().tolist())
currencies = sorted(df[CURRENCY_COL].dropna().astype(str).unique().tolist())
app.layout = html.Div(
    style={"padding": "20px", "backgroundColor": "#0f172a", "color": "#e5e7eb", "minHeight": "100vh"},
    children=[
        html.H2(
            "Payout Dashboard",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),
        html.P(
            "Клик по бару 15‑минутного окна → провал в шлюзы. "
            "Клик по столбцу шлюза → провал в трейдеров.",
            style={"textAlign": "center", "color": "#9ca3af", "marginBottom": "30px"},
        ),
        html.P(
            "Примечание: в расчетах исключены записи с order_status = 'new'.",
            style={"textAlign": "center", "color": "#fbbf24", "marginTop": "-18px", "marginBottom": "18px"},
        ),

        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "20px", "alignItems": "center"},
            children=[
                html.Div(
                    children=[
                        html.Label("Шлюз:", style={"marginRight": "8px"}),
                        dcc.Dropdown(
                            id="gateway-filter",
                            options=[{"label": gw, "value": gw} for gw in engines],
                            value=[],
                            multi=True,
                            placeholder="Все шлюзы",
                            style={"color": "#111827"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.Label("Currency:", style={"marginRight": "8px"}),
                        dcc.Dropdown(
                            id="currency-filter",
                            options=[{"label": cur, "value": cur} for cur in currencies],
                            value=[],
                            multi=True,
                            placeholder="Все валюты",
                            style={"color": "#111827"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.Label("Интервал даты:", style={"marginRight": "8px"}),
                        dcc.DatePickerRange(
                            id="date-range-filter",
                            start_date=default_start_date,
                            end_date=default_end_date,
                            display_format="YYYY-MM-DD",
                            minimum_nights=0,
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "20px", "alignItems": "center"},
            children=[
                html.Div(
                    children=[
                        html.Label("Время от:", style={"marginRight": "8px"}),
                        dcc.Dropdown(
                            id="start-time-filter",
                            options=time_options,
                            value=default_start_time,
                            clearable=False,
                            style={"color": "#111827"},
                        ),
                    ],
                    style={"width": "220px"},
                ),
                html.Div(
                    children=[
                        html.Label("Время до:", style={"marginRight": "8px"}),
                        dcc.Dropdown(
                            id="end-time-filter",
                            options=time_options,
                            value=default_end_time,
                            clearable=False,
                            style={"color": "#111827"},
                        ),
                    ],
                    style={"width": "220px"},
                ),
            ],
        ),

        # скрытые стейты для drill-down
        dcc.Store(id="selected-gateway"),

        html.Div(
            children=[
                dcc.Graph(id="time-bar-chart", animate=False, config={"displayModeBar": False}),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "center", "marginTop": "-8px", "marginBottom": "12px"},
            children=[
                dcc.RadioItems(
                    id="conversion-mode",
                    options=[
                        {"label": "Во взятие", "value": "take"},
                        {"label": "В success", "value": "success"},
                    ],
                    value="take",
                    inline=True,
                    labelStyle={"marginRight": "14px"},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "24px"},
            children=[
                html.Div(
                    children=[
                        html.H4("Конверсия во взятие ордера шлюзом", style={"textAlign": "center"}),
                        dcc.Graph(id="gateway-conv-chart", animate=False, config={"displayModeBar": False}),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.H4("Динамика конверсии выбранного шлюза", style={"textAlign": "center"}),
                        dcc.Graph(id="trader-conv-chart", animate=True, config={"displayModeBar": False}),
                    ],
                    style={"flex": "1"},
                ),
            ],
        ),

        # конверсия по времени (верх - выбранный шлюз, справа - Aifory)
        html.Div(
            style={"marginTop": "32px"},
            children=[
                html.H4(
                    "Конверсия по времени",
                    style={"textAlign": "center"},
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "marginBottom": "8px",
                        "gap": "8px",
                    },
                    children=[
                        html.Label("Шлюз:", style={"marginRight": "4px"}),
                        dcc.Dropdown(
                            id="gateway-daily-select",
                            options=[{"label": gw, "value": gw} for gw in engines],
                            value=engines[0],
                            clearable=False,
                            style={"width": "200px", "color": "#111827"},
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "24px", "marginTop": "8px"},
                    children=[
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.H5(
                                    "Выбранный шлюз",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="gateway-timeseries-conv-chart",
                                    animate=True,
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.H5(
                                    "Aifory (GW_A)",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="gateway-timeseries-aifory-chart",
                                    animate=True,
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ------------------------
# Callbacks
# ------------------------

@app.callback(
    Output("time-bar-chart", "figure"),
    Input("gateway-filter", "value"),
    Input("currency-filter", "value"),
    Input("date-range-filter", "start_date"),
    Input("date-range-filter", "end_date"),
    Input("start-time-filter", "value"),
    Input("end-time-filter", "value"),
)
def update_time_chart(selected_gateways, selected_currencies, start_date, end_date, start_time, end_time):
    agg = apply_dashboard_filters(
        df,
        selected_gateways=selected_gateways,
        selected_currencies=selected_currencies,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
    )

    if agg.empty:
        return go.Figure(
            layout=go.Layout(
                template="plotly_dark",
                xaxis_title="15‑мин окно",
                yaxis_title="Кол-во ордеров",
                annotations=[
                    dict(
                        text="Нет данных для отображения",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16),
                    )
                ],
            )
        )

    # Агрегация до уровня слот+статус, чтобы разбивка в hover была корректной.
    agg = (
        agg.groupby(["period_start", "status_from_engine"], as_index=False)
        .agg(orders_count=("orders_count", "sum"))
        .sort_values("period_start")
    )
    agg["slot_total_orders"] = agg.groupby("period_start")["orders_count"].transform("sum")
    first_status_in_slot = agg.groupby("period_start")["status_from_engine"].transform("min")
    agg["total_line"] = np.where(
        agg["status_from_engine"] == first_status_in_slot,
        "<br>orders(total)=" + agg["slot_total_orders"].round(0).astype(int).astype(str),
        "",
    )

    fig = px.bar(
        agg,
        x="period_start",
        y="orders_count",
        color="status_from_engine",  # Разбивка по статусу ордера
        custom_data=["total_line"],
        labels={"period_start": "15‑мин окно", "orders_count": "Кол-во ордеров", "status_from_engine": "Статус ордера"},
        title="Конверсия по статусам ордеров",
    )
    fig.update_traces(
        hovertemplate=(
            "slot=%{x}<br>"
            "status=%{fullData.name}<br>"
            "orders(status)=%{y}%{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2937"),
        hovermode="x unified",
        transition_duration=300,
    )
    return fig


@app.callback(
    Output("gateway-conv-chart", "figure"),
    Input("time-bar-chart", "clickData"),
    Input("conversion-mode", "value"),
    Input("gateway-filter", "value"),
    Input("currency-filter", "value"),
    Input("date-range-filter", "start_date"),
    Input("date-range-filter", "end_date"),
    Input("start-time-filter", "value"),
    Input("end-time-filter", "value"),
)
def update_gateway_chart(
    slot_click_data,
    conversion_mode,
    selected_gateways,
    selected_currencies,
    start_date,
    end_date,
    start_time,
    end_time,
):
    df_slice = apply_dashboard_filters(
        df,
        selected_gateways=selected_gateways,
        selected_currencies=selected_currencies,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
    )
    if slot_click_data and slot_click_data.get("points"):
        x_val = slot_click_data["points"][0]["x"]
        start_ts = pd.to_datetime(x_val)
        end_ts = start_ts + timedelta(minutes=15)
        mask = (df_slice[TIME_COL] >= start_ts) & (df_slice[TIME_COL] < end_ts)
        df_slice = df_slice[mask]

    agg = make_gateway_conv(df_slice, metric_mode=conversion_mode)

    if agg.empty:
        return go.Figure(
            layout=go.Layout(
                template="plotly_dark",
                annotations=[
                    dict(
                        text="Нет данных по шлюзам для выбранного окна",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14),
                    )
                ],
            )
        )

    if conversion_mode == "success":
        status_order = ["not-success", "success"]
        color_map = {"not-success": "#ef4444", "success": "#22c55e"}
        title_text = "Структура по шлюзам: success + not-success = 100% (из non-rejected)"
        formula_text = "conversion(success/non-rejected)"
    else:
        status_order = ["rejected", "non-rejected"]
        color_map = {"rejected": "#ef4444", "non-rejected": "#22c55e"}
        title_text = "Структура по шлюзам: rejected + non-rejected = 100%"
        formula_text = "conversion(non-rejected/total)"

    agg["status_group"] = pd.Categorical(agg["status_group"], categories=status_order, ordered=True)
    agg = agg.sort_values(["engine", "status_group"])

    fig = px.bar(
        agg,
        x="engine",
        y="pct",
        color="status_group",
        custom_data=["orders", "total_orders", "conversion_pct"],
        color_discrete_map=color_map,
        category_orders={"status_group": status_order},
        labels={"engine": "Шлюз", "pct": "Доля ордеров, %"},
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis=dict(range=[0, 100], title="Конверсия, %"),
        xaxis_title="Шлюз",
        title=(f"{title_text}<br>"),
        transition_duration=300,
    )
    fig.update_traces(
        hovertemplate=(
            "gateway=%{x}<br>"
            "status=%{fullData.name}<br>"
            "share=%{y:.1f}%<br>"
            "orders=%{customdata[0]:.0f} / total=%{customdata[1]:.0f}<br>"
            f"{formula_text}=%{{customdata[2]:.1f}}%<extra></extra>"
        )
    )
    return fig


@app.callback(
    Output("selected-gateway", "data"),
    Input("gateway-conv-chart", "clickData"),
    prevent_initial_call=True,
)
def store_selected_gateway(click_data):
    if not click_data:
        return dash.no_update
    gw = click_data["points"][0]["x"]
    return {"gateway": gw}


@app.callback(
    Output("trader-conv-chart", "figure"),
    Input("selected-gateway", "data"),
    Input("conversion-mode", "value"),
    Input("gateway-filter", "value"),
    Input("currency-filter", "value"),
    Input("date-range-filter", "start_date"),
    Input("date-range-filter", "end_date"),
    Input("start-time-filter", "value"),
    Input("end-time-filter", "value"),
)
def update_selected_gateway_timeseries(
    gw_data,
    conversion_mode,
    selected_gateways,
    selected_currencies,
    start_date,
    end_date,
    start_time,
    end_time,
):
    gateway = None
    if gw_data and "gateway" in gw_data:
        gateway = gw_data["gateway"]
    elif selected_gateways:
        gateway = selected_gateways[0]

    if not gateway:
        return go.Figure(
            layout=go.Layout(
                template="plotly_dark",
                annotations=[
                    dict(
                        text="Кликните по бару шлюза на левом графике",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14),
                    )
                ],
            )
        )

    filtered_df = apply_dashboard_filters(
        df,
        selected_gateways=selected_gateways,
        selected_currencies=selected_currencies,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
    )
    filtered_df = filtered_df[filtered_df["engine"].astype(str) == str(gateway)]

    agg = make_gateway_conv_timeseries(filtered_df, metric_mode=conversion_mode)
    if agg.empty:
        return go.Figure(
            layout=go.Layout(
                template="plotly_dark",
                annotations=[
                    dict(
                        text=f"Нет данных по шлюзу {gateway}",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14),
                    )
                ],
            )
        )

    if conversion_mode == "success":
        status_order = ["not-success", "success"]
        color_map = {"not-success": "#ef4444", "success": "#22c55e"}
        formula_text = "conversion(success/non-rejected)"
        title_text = f"Динамика success-конверсии: {gateway}"
    else:
        status_order = ["rejected", "non-rejected"]
        color_map = {"rejected": "#ef4444", "non-rejected": "#22c55e"}
        formula_text = "conversion(non-rejected/total)"
        title_text = gateway

    agg["status_group"] = pd.Categorical(agg["status_group"], categories=status_order, ordered=True)
    agg = agg.sort_values(["period_start", "status_group"])

    fig = px.bar(
        agg,
        x="period_start",
        y="pct",
        color="status_group",
        custom_data=["orders", "total_orders", "conversion_pct"],
        color_discrete_map=color_map,
        category_orders={"status_group": status_order},
        labels={"period_start": "Время", "pct": "Доля, %"},
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        yaxis=dict(range=[0, 100], title="Конверсия, %"),
        xaxis_title="15-мин окно",
        margin=dict(l=40, r=20, t=60, b=40),
        title=title_text,
        transition_duration=300,
    )
    fig.update_traces(
        hovertemplate=(
            "time=%{x}<br>"
            "status=%{fullData.name}<br>"
            "share=%{y:.1f}%<br>"
            "orders=%{customdata[0]:.0f} / total=%{customdata[1]:.0f}<br>"
            f"{formula_text}=%{{customdata[2]:.1f}}%<extra></extra>"
        )
    )
    return fig


# @app.callback(
#     Output("gateway-daily-select", "value"),
#     Input("selected-gateway", "data"),
#     prevent_initial_call=True,
# )
# def sync_gateway_dropdown(gw_data):
#     if not gw_data:
#         return dash.no_update
#     return gw_data.get("gateway")


# @app.callback(
#     Output("trader-conv-chart", "figure"),
#     Input("selected-slot", "data"),
# )
# def update_trader_chart(slot_data):
#     # трейдеры есть только в шлюзе Aifory → фиксируем gw
#     gateway = "GW_A"
#     time_window = None
#     subtitle = f"Шлюз {gateway}, весь период"
#     if slot_data:
#         start_ts = pd.to_datetime(slot_data["start"])
#         end_ts = pd.to_datetime(slot_data["end"])
#         time_window = (start_ts, end_ts)
#         subtitle = f"Шлюз {gateway}, {start_ts:%Y-%m-%d %H:%M} — {end_ts:%H:%M}"

#     agg = make_trader_conv(df, gateway=gateway, time_window=time_window)
#     if agg.empty:
#         return go.Figure(
#             layout=go.Layout(
#                 template="plotly_dark",
#                 annotations=[
#                     dict(
#                         text="Нет трейдеров для выбранного шлюза/окна",
#                         x=0.5,
#                         y=0.5,
#                         xref="paper",
#                         yref="paper",
#                         showarrow=False,
#                         font=dict(size=14),
#                     )
#                 ],
#             )
#         )

#     # stacked bar по трейдерам (Aifory)
#     fig = px.bar(
#         agg,
#         x="trader",
#         y="pct",
#         color="status",
#         color_discrete_map={"success": "#22c55e", "fail": "#ef4444"},
#         labels={"trader": "Трейдер", "pct": "Доля ордеров, %"},
#     )
#     fig.update_layout(
#         template="plotly_dark",
#         barmode="stack",
#         margin=dict(l=40, r=20, t=60, b=80),
#         yaxis=dict(range=[0, 100], title="Конверсия, %"),
#         xaxis_title="Трейдер",
#         title=f"Трейдеры шлюза Aifory<br><sup>{subtitle}</sup>",
#         xaxis_tickangle=-45,
#         transition_duration=300,
#     )
#     fig.update_traces(
#         hovertemplate=(
#             "trader=%{x}<br>"
#             "status=%{color}<br>"
#             "pct=%{y:.1f}%<extra></extra>"
#         )
#     )
#     return fig


# @app.callback(
#     Output("gateway-timeseries-conv-chart", "figure"),
#     Input("gateway-daily-select", "value"),
#     Input("selected-gateway", "data"),
# )
# def update_gateway_timeseries_chart(dropdown_gateway, gw_data):
#     # приоритет: клик по графику шлюзов, потом значение дропдауна
#     if gw_data and "gateway" in gw_data:
#         gateway = gw_data["gateway"]
#     else:
#         gateway = dropdown_gateway

#     if not gateway:
#         return go.Figure(
#             layout=go.Layout(
#                 template="plotly_dark",
#                 annotations=[
#                     dict(
#                         text="Выберите шлюз",
#                         x=0.5,
#                         y=0.5,
#                         xref="paper",
#                         yref="paper",
#                         showarrow=False,
#                         font=dict(size=14),
#                     )
#                 ],
#             )
#         )

#     ts = make_gateway_conv_timeseries(df, gateway)
#     if ts.empty:
#         return go.Figure(
#             layout=go.Layout(
#                 template="plotly_dark",
#                 annotations=[
#                     dict(
#                         text="Нет данных по времени для выбранного шлюза",
#                         x=0.5,
#                         y=0.5,
#                         xref="paper",
#                         yref="paper",
#                         showarrow=False,
#                         font=dict(size=14),
#                     )
#                 ],
#             )
#         )

#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=ts["slot_15m"],
#             y=ts["fail"],
#             mode="lines",
#             stackgroup="one",
#             line=dict(width=0),
#             name="Неуспешные ордера",
#             marker_color="#f97373",
#             hovertemplate="time=%{x}<br>fail=%{y:.1f}%<extra></extra>",
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=ts["slot_15m"],
#             y=ts["success"],
#             mode="lines",
#             stackgroup="one",
#             line=dict(width=0),
#             name="Успешные ордера",
#             marker_color="#22c55e",
#             hovertemplate="time=%{x}<br>success=%{y:.1f}%<extra></extra>",
#         )
#     )

#     fig.update_layout(
#         template="plotly_dark",
#         yaxis=dict(range=[0, 100], title="Конверсия, %"),
#         xaxis_title="Время",
#         margin=dict(l=40, r=20, t=60, b=40),
#         title=f"Конверсия по времени – шлюз {gateway}",
#         transition_duration=300,
#     )
#     return fig


# @app.callback(
#     Output("gateway-timeseries-aifory-chart", "figure"),
#     Input("gateway-daily-select", "value"),
# )
# def update_gateway_timeseries_aifory(_):
#     gateway = "GW_A"
#     ts = make_gateway_conv_timeseries(df, gateway)
#     if ts.empty:
#         return go.Figure(
#             layout=go.Layout(
#                 template="plotly_dark",
#                 annotations=[
#                     dict(
#                         text="Нет данных по времени для Aifory",
#                         x=0.5,
#                         y=0.5,
#                         xref="paper",
#                         yref="paper",
#                         showarrow=False,
#                         font=dict(size=14),
#                     )
#                 ],
#             )
#         )

#     # ts в широком формате (slot_15m, fail, success) — переводим в длинный для px.bar
#     ts_long = ts.melt(
#         id_vars=["slot_15m"],
#         value_vars=["fail", "success"],
#         var_name="status",
#         value_name="pct",
#     )
#     fig = px.bar(
#         ts_long,
#         x="slot_15m",
#         y="pct",
#         color="status",
#         color_discrete_map={"success": "#22c55e", "fail": "#ef4444"},
#         labels={"slot_15m": "Время", "pct": "Доля ордеров, %"},
#     )
#     fig.update_layout(
#         template="plotly_dark",
#         barmode="stack",
#         yaxis=dict(range=[0, 100], title="Конверсия, %"),
#         xaxis_title="Время",
#         margin=dict(l=40, r=20, t=60, b=40),
#         title="Конверсия по времени – Aifory (GW_A)",
#         transition_duration=300,
#     )
#     fig.update_traces(
#         hovertemplate=(
#             "time=%{x}<br>"
#             "status=%{color}<br>"
#             "pct=%{y:.1f}%<extra></extra>"
#         )
#     )
#     return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)

