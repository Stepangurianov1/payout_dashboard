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
app.title = "Funnel Dashboard (Demo)"

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
    return pd.read_sql_table("payout_engine_info", engine_dwh, schema="cascade")


df = load_payout_engine_info_cached()
# np.random.seed(42)

# now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
# start = now - timedelta(hours=6)  # последние 6 часов

# time_index = pd.date_range(start=start, end=now, freq="2min")  # "сырые" события раз в 2 минуты

# gateways = ["GW_A", "GW_B", "GW_C", "GW_D"]
# traders_per_gateway = {
#     "GW_A": [f"A_T{i}" for i in range(1, 6)],
#     "GW_B": [f"B_T{i}" for i in range(1, 5)],
#     "GW_C": [f"C_T{i}" for i in range(1, 4)],
#     "GW_D": [f"D_T{i}" for i in range(1, 7)],
# }

# rows = []
# for ts in time_index:
#     # кол-во ордеров в момент времени
#     n_orders = np.random.poisson(4)
#     for _ in range(n_orders):
#         gw = np.random.choice(gateways, p=[0.4, 0.3, 0.2, 0.1])
#         trader = np.random.choice(traders_per_gateway[gw])
#         amount = np.random.lognormal(mean=4.0, sigma=0.6)  # искусственные суммы
#         base_success = {"GW_A": 0.9, "GW_B": 0.8, "GW_C": 0.7, "GW_D": 0.6}[gw]
#         status = "success" if np.random.rand() < base_success else "fail"
#         rows.append(
#             {
#                 "timestamp": ts,
#                 "gateway": gw,
#                 "trader": trader,
#                 "status": status,
#                 "amount": amount,
#             }
#         )

# df = pd.DataFrame(rows)
# df["slot_15m"] = df["timestamp"].dt.floor("15min")


# ------------------------
# Вспомогательные функции агрегации
# ------------------------

def make_time_agg(df_):
    agg = (
        df_.groupby("slot_15m")
        .size()
        .rename("orders_count")
        .reset_index()
        .sort_values("slot_15m")
    )
    return agg


def make_gateway_conv(df_):
    agg = (
        df_.groupby(["engine", "status_from_engine"], as_index=False)
        .agg(orders=("orders_count_wo_new", "sum"))
    )
    total_orders = agg.groupby("engine")["orders"].transform("sum")
    agg["pct"] = np.where(total_orders > 0, agg["orders"] / total_orders * 100, 0)
    return agg


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


def make_gateway_conv_timeseries(df_, gateway):
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

engines = df["engine"].unique()
app.layout = html.Div(
    style={"padding": "20px", "backgroundColor": "#0f172a", "color": "#e5e7eb", "minHeight": "100vh"},
    children=[
        html.H2(
            "Funnel Dashboard (demo)",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),
        html.P(
            "Клик по бару 15‑минутного окна → провал в шлюзы. "
            "Клик по столбцу шлюза → провал в трейдеров.",
            style={"textAlign": "center", "color": "#9ca3af", "marginBottom": "30px"},
        ),

        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "20px", "alignItems": "center"},
            children=[
                html.Div(
                    children=[
                        html.Label("Минимальное количество ордеров в 15‑мин окне:", style={"marginRight": "8px"}),
                        dcc.Slider(
                            id="min-orders-slider",
                            min=0,
                            max=50,
                            step=5,
                            value=5,
                            marks={i: str(i) for i in range(0, 55, 10)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.Label("Статус по умолчанию для нормировки:", style={"marginRight": "8px"}),
                        dcc.Dropdown(
                            id="status-filter",
                            options=[
                                {"label": "Все", "value": "all"},
                                {"label": "Success", "value": "success"},
                                {"label": "Fail", "value": "fail"},
                            ],
                            value="all",
                            clearable=False,
                            style={"width": "160px", "color": "#111827"},
                        ),
                    ]
                ),
            ],
        ),

        # скрытые стейты для drill-down
        dcc.Store(id="selected-slot"),
        dcc.Store(id="selected-gateway"),

        html.Div(
            children=[
                dcc.Graph(id="time-bar-chart", animate=True, config={"displayModeBar": False}),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "24px"},
            children=[
                html.Div(
                    children=[
                        html.H4("Конверсия по шлюзам (окно / drill-down)", style={"textAlign": "center"}),
                        dcc.Graph(id="gateway-conv-chart", animate=True, config={"displayModeBar": False}),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.H4("Конверсия по трейдерам шлюза Aifory", style={"textAlign": "center"}),
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
    Input("min-orders-slider", "value"),
)
def update_time_chart(min_orders):
    agg = df
    if min_orders:
        agg = agg[agg["orders_count"] >= min_orders]

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

    # Добавление разбивки по статусу с помощью цвета
    fig = px.bar(
        agg,
        x="period_start",
        y="orders_count",
        color="status_from_engine",  # Разбивка по статусу ордера
        labels={"period_start": "15‑мин окно", "orders_count": "Кол-во ордеров", "status_from_engine": "Статус ордера"},
        title="Конверсия по статусам ордеров",
    )
    fig.update_traces(
        hovertemplate="slot=%{x}<br>orders=%{y}<br>status=%{fullData.name}<extra></extra>",
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
    Output("selected-slot", "data"),
    Input("time-bar-chart", "clickData"),
    prevent_initial_call=True,
)
def store_selected_slot(click_data):
    if not click_data:
        return dash.no_update
    x_val = click_data["points"][0]["x"]
    slot_dt = pd.to_datetime(x_val)
    start_ts = slot_dt
    end_ts = slot_dt + timedelta(minutes=15)
    return {"start": start_ts.isoformat(), "end": end_ts.isoformat()}


@app.callback(
    Output("gateway-conv-chart", "figure"),
    Input("selected-slot", "data"),
    Input("status-filter", "value"),
)
def update_gateway_chart(slot_data, status_filter):
    subtitle = "за весь период"
    if slot_data:
        start_ts = pd.to_datetime(slot_data["start"])
        end_ts = pd.to_datetime(slot_data["end"])
        subtitle = f"{start_ts:%Y-%m-%d %H:%M} — {end_ts:%H:%M}"

    agg = make_gateway_conv(df)

    if status_filter != "all":
        agg = agg[agg["status_from_engine"] == status_filter]

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

    # обратно к stacked bar по шлюзам
    fig = px.bar(
        agg,
        x="engine",
        y="pct",
        color="status_from_engine",
        color_discrete_map={"success": "#22c55e", "fail": "#ef4444"},
        labels={"engine": "Шлюз", "pct": "Доля ордеров, %"},
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis=dict(range=[0, 100], title="Конверсия, %"),
        xaxis_title="Шлюз",
        title=f"Шлюзы, нормировано до 100%<br><sup>{subtitle}</sup>",
        transition_duration=300,
    )
    fig.update_traces(
        hovertemplate=(
            "gateway=%{x}<br>"
            "status=%{fullData.name}<br>"
            "pct=%{y:.1f}%<extra></extra>"
        )
    )
    return fig


# @app.callback(
#     Output("selected-gateway", "data"),
#     Input("gateway-conv-chart", "clickData"),
#     prevent_initial_call=True,
# )
# def store_selected_gateway(click_data):
#     if not click_data:
#         return dash.no_update
#     gw = click_data["points"][0]["x"]
#     return {"gateway": gw}


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

