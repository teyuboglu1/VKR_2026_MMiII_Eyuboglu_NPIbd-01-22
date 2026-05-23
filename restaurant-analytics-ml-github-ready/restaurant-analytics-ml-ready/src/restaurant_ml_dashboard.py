from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

RANDOM_STATE = 42
TEST_DAYS = 60

FORECAST_TARGETS = {
    "Revenue": "Выручка",
    "Profit": "Прибыль",
    "Quantity": "Количество проданных блюд",
    "Unique_Customers": "Уникальные клиенты",
}

DIMENSION_OPTIONS = {
    "Full_Date": "Дата",
    "Restaurant_Name": "Ресторан",
    "Season": "Сезон",
    "Month_Name": "Месяц",
    "Day_Of_Week_Name": "День недели",
    "Restaurant_Type": "Тип ресторана",
    "District": "Район",
    "Category_Name": "Категория блюда",
    "Subcategory_Name": "Подкатегория блюда",
    "Cuisine_Type": "Кухня",
    "Menu_Item_Name": "Блюдо",
    "Is_Seasonal": "Сезонное блюдо",
    "Gender": "Пол клиента",
    "Age_Group": "Возрастная группа",
    "Loyalty_Status": "Статус лояльности",
    "Preferred_Channel": "Предпочитаемый канал",
    "Segment": "Сегмент клиента",
    "Average_Check_Group": "Группа среднего чека",
}

METRIC_OPTIONS = {
    "Revenue": "Выручка",
    "Profit": "Прибыль",
    "Quantity": "Количество блюд",
    "Discount": "Скидки",
    "Unique_Customers": "Уникальные клиенты",
    "Sales_Lines": "Строки продаж",
    "Avg_Rating": "Средний рейтинг",
    "Avg_Check": "Средний чек",
}

CHART_OPTIONS = {
    "line": "Линейный график",
    "bar": "Столбчатая диаграмма",
    "area": "Диаграмма с областями",
    "pie": "Круговая диаграмма",
    "treemap": "Treemap",
}

MODEL_OPTIONS = {
    "Best_RMSE": "Лучшая модель по RMSE",
    "Baseline_Median": "Baseline Median",
    "Ridge": "Ridge",
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "HistGradientBoosting": "HistGradientBoosting",
}

SEASON_ORDER = ["Зима", "Весна", "Лето", "Осень"]

WEEKDAY_NAMES = {
    1: "Понедельник",
    2: "Вторник",
    3: "Среда",
    4: "Четверг",
    5: "Пятница",
    6: "Суббота",
    7: "Воскресенье",
}

MONTH_NAMES = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь",
}


class DashboardData:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.engine = create_engine(f"duckdb:///{self.db_path.as_posix()}")
        self.sales_df = self._load_sales_data()
        self.date_dim = self._load_date_dim()
        self.restaurant_dim = self._load_restaurant_dim()
        self.daily_ml_base = self._prepare_daily_ml_base()
        self.trained_cache: dict[str, dict[str, Any]] = {}

    def sql(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(text(query), self.engine)

    def _load_sales_data(self) -> pd.DataFrame:
        data = self.sql(
            """
            SELECT
                f.Sales_Fact_ID,

                d.Date_ID,
                d.Full_Date,
                d.Day_Number,
                d.Month_Number,
                d.Month_Name,
                d.Quarter_Number,
                d.Year_Number,
                d.Day_Of_Week,
                d.Is_Weekend,
                d.Is_Holiday,
                d.Season,

                r.Restaurant_ID,
                r.Restaurant_Name,
                r.City,
                r.District,
                r.Restaurant_Type,
                r.Seating_Capacity,
                r.Delivery_Available,
                r.Drive_Thru_Available,

                m.Menu_Item_ID,
                m.Menu_Item_Name,
                m.Category_Name,
                m.Subcategory_Name,
                m.Cuisine_Type,
                m.Is_Seasonal,

                c.Customer_ID,
                c.Gender,
                c.Age_Group,
                c.Loyalty_Status,
                c.Preferred_Channel,
                c.Segment,
                c.Average_Check_Group,

                f.Quantity,
                CAST(f.Unit_Price AS DOUBLE) AS Unit_Price,
                CAST(f.Discount_Amount AS DOUBLE) AS Discount,
                CAST(f.Cost_Amount AS DOUBLE) AS Cost,
                CAST(f.Revenue_Amount AS DOUBLE) AS Revenue,
                CAST(f.Profit_Amount AS DOUBLE) AS Profit,
                f.Preparation_Time,
                f.Service_Time,
                CAST(f.Rating_Value AS DOUBLE) AS Rating

            FROM Fact_Sales f
            JOIN Dim_Date d
                ON f.Date_ID = d.Date_ID
            JOIN Dim_Restaurant r
                ON f.Restaurant_ID = r.Restaurant_ID
            JOIN Dim_Menu_Item m
                ON f.Menu_Item_ID = m.Menu_Item_ID
            JOIN Dim_Customer c
                ON f.Customer_ID = c.Customer_ID
            ORDER BY
                d.Full_Date,
                r.Restaurant_ID,
                f.Sales_Fact_ID
            """
        )

        if data.empty:
            raise ValueError("В Fact_Sales нет данных. Запустите генератор базы с параметром --demo.")

        data["Full_Date"] = pd.to_datetime(data["Full_Date"])
        data["Day_Of_Week_Name"] = data["Day_Of_Week"].map(WEEKDAY_NAMES)
        data["Is_Seasonal"] = data["Is_Seasonal"].map({True: "Да", False: "Нет"}).fillna(data["Is_Seasonal"])

        return data

    def _load_date_dim(self) -> pd.DataFrame:
        data = self.sql(
            """
            SELECT
                Date_ID,
                Full_Date,
                Day_Number,
                Month_Number,
                Month_Name,
                Quarter_Number,
                Year_Number,
                Day_Of_Week,
                Is_Weekend,
                Is_Holiday,
                Season
            FROM Dim_Date
            ORDER BY Full_Date
            """
        )
        data["Full_Date"] = pd.to_datetime(data["Full_Date"])
        return data

    def _load_restaurant_dim(self) -> pd.DataFrame:
        return self.sql(
            """
            SELECT
                Restaurant_ID,
                Restaurant_Name,
                City,
                District,
                Restaurant_Type,
                Seating_Capacity,
                Delivery_Available,
                Drive_Thru_Available
            FROM Dim_Restaurant
            ORDER BY Restaurant_ID
            """
        )

    def _prepare_daily_ml_base(self) -> pd.DataFrame:
        daily_sales = (
            self.sales_df
            .groupby(
                [
                    "Date_ID",
                    "Full_Date",
                    "Restaurant_ID",
                    "Restaurant_Name",
                    "City",
                    "District",
                    "Restaurant_Type",
                    "Seating_Capacity",
                    "Delivery_Available",
                    "Drive_Thru_Available",
                    "Day_Number",
                    "Month_Number",
                    "Month_Name",
                    "Quarter_Number",
                    "Year_Number",
                    "Day_Of_Week",
                    "Is_Weekend",
                    "Is_Holiday",
                    "Season",
                ],
                as_index=False,
            )
            .agg(
                Sales_Lines=("Sales_Fact_ID", "count"),
                Unique_Customers=("Customer_ID", "nunique"),
                Quantity=("Quantity", "sum"),
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Discount=("Discount", "sum"),
                Avg_Rating=("Rating", "mean"),
            )
        )

        min_date = daily_sales["Full_Date"].min()
        max_date = daily_sales["Full_Date"].max()

        date_calendar = self.date_dim[
            (self.date_dim["Full_Date"] >= min_date)
            & (self.date_dim["Full_Date"] <= max_date)
        ].copy()

        date_calendar["key"] = 1

        restaurant_features = self.restaurant_dim.copy()
        restaurant_features["key"] = 1

        full_grid = (
            date_calendar
            .merge(restaurant_features, on="key", how="inner")
            .drop(columns="key")
        )

        result = full_grid.merge(
            daily_sales[
                [
                    "Date_ID",
                    "Restaurant_ID",
                    "Sales_Lines",
                    "Unique_Customers",
                    "Quantity",
                    "Revenue",
                    "Profit",
                    "Discount",
                    "Avg_Rating",
                ]
            ],
            on=["Date_ID", "Restaurant_ID"],
            how="left",
        )

        fill_zero_columns = [
            "Sales_Lines",
            "Unique_Customers",
            "Quantity",
            "Revenue",
            "Profit",
            "Discount",
        ]

        result[fill_zero_columns] = result[fill_zero_columns].fillna(0)
        result["Avg_Rating"] = result["Avg_Rating"].fillna(result["Avg_Rating"].median())
        result = result.sort_values(["Restaurant_ID", "Full_Date"]).reset_index(drop=True)

        return result


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Зима"
    if month in [3, 4, 5]:
        return "Весна"
    if month in [6, 7, 8]:
        return "Лето"
    return "Осень"


def add_calendar_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["Full_Date"] = pd.to_datetime(result["Full_Date"])
    result["Day_Of_Year"] = result["Full_Date"].dt.dayofyear
    result["Week_Of_Year"] = result["Full_Date"].dt.isocalendar().week.astype(int)
    result["Is_Month_Start"] = result["Full_Date"].dt.is_month_start.astype(int)
    result["Is_Month_End"] = result["Full_Date"].dt.is_month_end.astype(int)
    result["Is_Weekend"] = result["Is_Weekend"].astype(int)
    result["Is_Holiday"] = result["Is_Holiday"].astype(int)
    result["Delivery_Available"] = result["Delivery_Available"].astype(int)
    result["Drive_Thru_Available"] = result["Drive_Thru_Available"].astype(int)

    return result


def add_lag_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result = df.sort_values(["Restaurant_ID", "Full_Date"]).copy()
    group = result.groupby("Restaurant_ID")[target_column]

    for lag in [1, 7, 14, 28]:
        result[f"{target_column}_Lag_{lag}"] = group.shift(lag)

    for window in [7, 14, 28]:
        result[f"{target_column}_Rolling_{window}_Mean"] = (
            result
            .groupby("Restaurant_ID")[target_column]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

        result[f"{target_column}_Rolling_{window}_Std"] = (
            result
            .groupby("Restaurant_ID")[target_column]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
        )

    std_columns = [
        f"{target_column}_Rolling_7_Std",
        f"{target_column}_Rolling_14_Std",
        f"{target_column}_Rolling_28_Std",
    ]

    result[std_columns] = result[std_columns].fillna(0)

    return result


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    denominator = np.maximum(np.abs(y_true), 1)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def prepare_ml_dataset(data_source: DashboardData, target_column: str):
    data = data_source.daily_ml_base.copy()
    data = add_calendar_ml_features(data)
    data = add_lag_features(data, target_column)

    lag_columns = [
        f"{target_column}_Lag_1",
        f"{target_column}_Lag_7",
        f"{target_column}_Lag_14",
        f"{target_column}_Lag_28",
        f"{target_column}_Rolling_7_Mean",
        f"{target_column}_Rolling_14_Mean",
        f"{target_column}_Rolling_28_Mean",
        f"{target_column}_Rolling_7_Std",
        f"{target_column}_Rolling_14_Std",
        f"{target_column}_Rolling_28_Std",
    ]

    data = data.dropna(subset=lag_columns).reset_index(drop=True)

    categorical_features = [
        "Restaurant_Name",
        "City",
        "District",
        "Restaurant_Type",
        "Season",
    ]

    numeric_features = [
        "Day_Number",
        "Month_Number",
        "Quarter_Number",
        "Year_Number",
        "Day_Of_Week",
        "Is_Weekend",
        "Is_Holiday",
        "Day_Of_Year",
        "Week_Of_Year",
        "Is_Month_Start",
        "Is_Month_End",
        "Seating_Capacity",
        "Delivery_Available",
        "Drive_Thru_Available",
    ] + lag_columns

    feature_columns = categorical_features + numeric_features

    return data, feature_columns, categorical_features, numeric_features


def make_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_model(model_name: str):
    if model_name == "Baseline_Median":
        return DummyRegressor(strategy="median")

    if model_name == "Ridge":
        return Ridge(alpha=10.0)

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Неизвестная модель: {model_name}")


def train_models_for_target(data_source: DashboardData, target_column: str) -> dict[str, Any]:
    if target_column in data_source.trained_cache:
        return data_source.trained_cache[target_column]

    data, feature_columns, categorical_features, numeric_features = prepare_ml_dataset(data_source, target_column)

    max_date = data["Full_Date"].max()
    test_start_date = max_date - pd.Timedelta(days=TEST_DAYS - 1)

    train_data = data[data["Full_Date"] < test_start_date].copy()
    test_data = data[data["Full_Date"] >= test_start_date].copy()

    X_train = train_data[feature_columns].copy()
    y_train = train_data[target_column].copy()
    X_test = test_data[feature_columns].copy()
    y_test = test_data[target_column].copy()

    model_names = [
        "Baseline_Median",
        "Ridge",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
    ]

    trained_models = {}
    final_models = {}
    metrics_rows = []
    test_predictions = test_data[["Full_Date", "Restaurant_ID", "Restaurant_Name", target_column]].copy()

    for model_name in model_names:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
                ("model", make_model(model_name)),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = np.clip(pipeline.predict(X_test), 0, None)

        trained_models[model_name] = pipeline

        metrics = calculate_metrics(y_test, y_pred)
        metrics["Model"] = model_name
        metrics_rows.append(metrics)

        test_predictions[f"Pred_{model_name}"] = y_pred

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["Model", "MAE", "RMSE", "MAPE", "R2"]]
    metrics_df = metrics_df.sort_values("RMSE").reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["Model"]

    for model_name in model_names:
        final_pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
                ("model", make_model(model_name)),
            ]
        )

        final_pipeline.fit(data[feature_columns].copy(), data[target_column].copy())
        final_models[model_name] = final_pipeline

    result = {
        "data": data,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "trained_models": trained_models,
        "final_models": final_models,
        "metrics_df": metrics_df,
        "test_predictions": test_predictions,
        "best_model_name": best_model_name,
    }

    data_source.trained_cache[target_column] = result
    return result


def make_future_calendar(future_dates) -> pd.DataFrame:
    result = pd.DataFrame({"Full_Date": pd.to_datetime(future_dates)})
    result["Date_ID"] = result["Full_Date"].dt.strftime("%Y%m%d").astype(int)
    result["Day_Number"] = result["Full_Date"].dt.day
    result["Month_Number"] = result["Full_Date"].dt.month
    result["Month_Name"] = result["Month_Number"].map(MONTH_NAMES)
    result["Quarter_Number"] = ((result["Month_Number"] - 1) // 3) + 1
    result["Year_Number"] = result["Full_Date"].dt.year
    result["Day_Of_Week"] = result["Full_Date"].dt.dayofweek + 1
    result["Is_Weekend"] = result["Day_Of_Week"].isin([6, 7]).astype(int)
    result["Is_Holiday"] = 0
    result["Season"] = result["Month_Number"].apply(get_season)
    result["Day_Of_Year"] = result["Full_Date"].dt.dayofyear
    result["Week_Of_Year"] = result["Full_Date"].dt.isocalendar().week.astype(int)
    result["Is_Month_Start"] = result["Full_Date"].dt.is_month_start.astype(int)
    result["Is_Month_End"] = result["Full_Date"].dt.is_month_end.astype(int)
    return result


def make_recursive_forecast(
    data_source: DashboardData,
    model,
    target_column: str,
    feature_columns: list[str],
    forecast_days: int,
) -> pd.DataFrame:
    history = data_source.daily_ml_base.copy()
    history = add_calendar_ml_features(history)
    history = history.sort_values(["Restaurant_ID", "Full_Date"]).reset_index(drop=True)

    last_date = history["Full_Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

    forecast_parts = []

    future_restaurants = data_source.restaurant_dim.copy()
    future_restaurants["Delivery_Available"] = future_restaurants["Delivery_Available"].astype(int)
    future_restaurants["Drive_Thru_Available"] = future_restaurants["Drive_Thru_Available"].astype(int)

    for current_date in future_dates:
        current_calendar = make_future_calendar([current_date])
        current_calendar["key"] = 1

        current_restaurants = future_restaurants.copy()
        current_restaurants["key"] = 1

        current_rows = current_calendar.merge(current_restaurants, on="key", how="inner").drop(columns="key")
        current_rows[target_column] = np.nan

        temp = pd.concat([history, current_rows], ignore_index=True, sort=False)
        temp = add_lag_features(temp, target_column)

        current_features = temp[temp["Full_Date"] == current_date].copy()
        X_future = current_features[feature_columns].copy()

        predicted_values = np.clip(model.predict(X_future), 0, None)
        current_features["Forecast"] = predicted_values

        predicted_map = current_features[["Restaurant_ID", "Forecast"]].copy()
        current_rows = current_rows.merge(predicted_map, on="Restaurant_ID", how="left")
        current_rows[target_column] = current_rows["Forecast"]
        current_rows = current_rows.drop(columns=["Forecast"])

        forecast_parts.append(
            current_features[
                [
                    "Full_Date",
                    "Restaurant_ID",
                    "Restaurant_Name",
                    "City",
                    "District",
                    "Restaurant_Type",
                    "Season",
                    "Day_Of_Week",
                    "Is_Weekend",
                    "Forecast",
                ]
            ].copy()
        )

        history = pd.concat([history, current_rows], ignore_index=True, sort=False)

    forecast = pd.concat(forecast_parts, ignore_index=True)
    forecast = forecast.sort_values(["Full_Date", "Restaurant_ID"]).reset_index(drop=True)
    return forecast


def filter_sales_data(
    data_source: DashboardData,
    selected_restaurants,
    selected_seasons,
    start_date,
    end_date,
) -> pd.DataFrame:
    data = data_source.sales_df.copy()

    if start_date:
        data = data[data["Full_Date"] >= pd.to_datetime(start_date)]

    if end_date:
        data = data[data["Full_Date"] <= pd.to_datetime(end_date)]

    if selected_restaurants:
        data = data[data["Restaurant_Name"].isin(selected_restaurants)]

    if selected_seasons:
        data = data[data["Season"].isin(selected_seasons)]

    return data


def aggregate_dashboard_data(data: pd.DataFrame, metric: str, dimension_1: str, dimension_2: str):
    dimensions = [dimension_1]

    if dimension_2 and dimension_2 != "None" and dimension_2 != dimension_1:
        dimensions.append(dimension_2)

    if metric == "Revenue":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Revenue", "sum"))
    elif metric == "Profit":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Profit", "sum"))
    elif metric == "Quantity":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Quantity", "sum"))
    elif metric == "Discount":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Discount", "sum"))
    elif metric == "Unique_Customers":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Customer_ID", "nunique"))
    elif metric == "Sales_Lines":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Sales_Fact_ID", "count"))
    elif metric == "Avg_Rating":
        result = data.groupby(dimensions, as_index=False).agg(Value=("Rating", "mean"))
    elif metric == "Avg_Check":
        grouped = data.groupby(dimensions, as_index=False).agg(
            Revenue=("Revenue", "sum"),
            Unique_Customers=("Customer_ID", "nunique"),
        )
        grouped["Value"] = grouped["Revenue"] / grouped["Unique_Customers"].replace(0, np.nan)
        result = grouped[dimensions + ["Value"]].copy()
    else:
        result = data.groupby(dimensions, as_index=False).agg(Value=("Revenue", "sum"))

    result["Value"] = result["Value"].fillna(0)

    if dimension_1 == "Full_Date":
        result = result.sort_values("Full_Date")
    else:
        result = result.sort_values("Value", ascending=False)

    return result, dimensions


def make_dashboard_figure(aggregated: pd.DataFrame, dimensions: list[str], metric: str, chart_type: str):
    metric_label = METRIC_OPTIONS.get(metric, metric)
    dimension_1 = dimensions[0]
    dimension_1_label = DIMENSION_OPTIONS.get(dimension_1, dimension_1)
    dimension_2 = dimensions[1] if len(dimensions) > 1 else None
    dimension_2_label = DIMENSION_OPTIONS.get(dimension_2, dimension_2) if dimension_2 else None

    title = f"{metric_label} по измерению «{dimension_1_label}»"
    if dimension_2:
        title += f" и «{dimension_2_label}»"

    labels = {
        dimension_1: dimension_1_label,
        "Value": metric_label,
    }
    if dimension_2:
        labels[dimension_2] = dimension_2_label

    if chart_type == "line":
        fig = px.line(aggregated, x=dimension_1, y="Value", color=dimension_2, markers=True, title=title, labels=labels)
    elif chart_type == "bar":
        fig = px.bar(aggregated, x=dimension_1, y="Value", color=dimension_2, barmode="group", title=title, labels=labels)
    elif chart_type == "area":
        fig = px.area(aggregated, x=dimension_1, y="Value", color=dimension_2, title=title, labels=labels)
    elif chart_type == "pie":
        if dimension_2:
            temp = aggregated.copy()
            temp["Combined_Dimension"] = temp[dimension_1].astype(str) + " | " + temp[dimension_2].astype(str)
            fig = px.pie(temp, names="Combined_Dimension", values="Value", title=title, hole=0.4)
        else:
            fig = px.pie(aggregated, names=dimension_1, values="Value", title=title, hole=0.4)
    elif chart_type == "treemap":
        fig = px.treemap(aggregated, path=dimensions, values="Value", title=title)
    else:
        fig = px.bar(aggregated, x=dimension_1, y="Value", color=dimension_2, title=title, labels=labels)

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        height=650,
        hovermode="x unified",
    )

    return fig


def make_options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": str(value), "value": str(value)} for value in values]


def create_app(data_source: DashboardData) -> Dash:
    app = Dash(__name__)
    app.title = "Restaurant ML Dashboard"

    sales_df = data_source.sales_df

    restaurant_options = make_options(sorted(sales_df["Restaurant_Name"].dropna().unique()))
    season_options = make_options([season for season in SEASON_ORDER if season in set(sales_df["Season"].dropna())])

    app.layout = html.Div(
        style={
            "fontFamily": "Arial",
            "backgroundColor": "#f6f8fb",
            "padding": "20px",
        },
        children=[
            html.H1(
                "Анализ эффективности ресторанного бизнеса и прогноз продаж",
                style={"textAlign": "center", "marginBottom": "10px"},
            ),
            html.Div(
                "Интерактивный дашборд на основе многомерного хранилища данных и моделей машинного обучения",
                style={"textAlign": "center", "marginBottom": "25px", "fontSize": "18px", "color": "#555"},
            ),
            dcc.Tabs(
                value="tab-analytics",
                children=[
                    dcc.Tab(
                        label="Многомерный анализ",
                        value="tab-analytics",
                        children=[
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(4, minmax(220px, 1fr))",
                                    "gap": "15px",
                                    "backgroundColor": "white",
                                    "padding": "18px",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                    "marginTop": "20px",
                                },
                                children=[
                                    html.Div([
                                        html.Label("Период"),
                                        dcc.DatePickerRange(
                                            id="date-range",
                                            min_date_allowed=sales_df["Full_Date"].min().date(),
                                            max_date_allowed=sales_df["Full_Date"].max().date(),
                                            start_date=sales_df["Full_Date"].min().date(),
                                            end_date=sales_df["Full_Date"].max().date(),
                                            display_format="YYYY-MM-DD",
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Рестораны"),
                                        dcc.Dropdown(
                                            id="restaurant-filter",
                                            options=restaurant_options,
                                            value=[x["value"] for x in restaurant_options],
                                            multi=True,
                                            placeholder="Выберите рестораны",
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Сезоны"),
                                        dcc.Dropdown(
                                            id="season-filter",
                                            options=season_options,
                                            value=[x["value"] for x in season_options],
                                            multi=True,
                                            placeholder="Выберите сезоны",
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Метрика"),
                                        dcc.Dropdown(
                                            id="metric-selector",
                                            options=[{"label": label, "value": value} for value, label in METRIC_OPTIONS.items()],
                                            value="Revenue",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Основное измерение"),
                                        dcc.Dropdown(
                                            id="dimension-1-selector",
                                            options=[{"label": label, "value": value} for value, label in DIMENSION_OPTIONS.items()],
                                            value="Full_Date",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Дополнительное измерение"),
                                        dcc.Dropdown(
                                            id="dimension-2-selector",
                                            options=[{"label": "Без дополнительного измерения", "value": "None"}]
                                            + [{"label": label, "value": value} for value, label in DIMENSION_OPTIONS.items()],
                                            value="Restaurant_Name",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Тип визуализации"),
                                        dcc.Dropdown(
                                            id="chart-type-selector",
                                            options=[{"label": label, "value": value} for value, label in CHART_OPTIONS.items()],
                                            value="line",
                                            clearable=False,
                                        ),
                                    ]),
                                ],
                            ),
                            html.Div(
                                style={
                                    "backgroundColor": "white",
                                    "padding": "18px",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                    "marginTop": "20px",
                                },
                                children=[dcc.Graph(id="main-analytics-chart")],
                            ),
                            html.Div(
                                style={
                                    "backgroundColor": "white",
                                    "padding": "18px",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                    "marginTop": "20px",
                                },
                                children=[
                                    html.H3("Агрегированные данные"),
                                    dash_table.DataTable(
                                        id="analytics-table",
                                        page_size=12,
                                        sort_action="native",
                                        filter_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "8px",
                                            "fontFamily": "Arial",
                                            "fontSize": "14px",
                                        },
                                        style_header={"fontWeight": "bold", "backgroundColor": "#eef2f7"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Прогноз продаж",
                        value="tab-forecast",
                        children=[
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(4, minmax(220px, 1fr))",
                                    "gap": "15px",
                                    "backgroundColor": "white",
                                    "padding": "18px",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                    "marginTop": "20px",
                                },
                                children=[
                                    html.Div([
                                        html.Label("Целевая переменная"),
                                        dcc.Dropdown(
                                            id="forecast-target",
                                            options=[{"label": label, "value": value} for value, label in FORECAST_TARGETS.items()],
                                            value="Revenue",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Модель"),
                                        dcc.Dropdown(
                                            id="forecast-model",
                                            options=[{"label": label, "value": value} for value, label in MODEL_OPTIONS.items()],
                                            value="Best_RMSE",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Горизонт прогноза, дней"),
                                        dcc.Slider(
                                            id="forecast-horizon",
                                            min=7,
                                            max=90,
                                            step=1,
                                            value=30,
                                            marks={7: "7", 30: "30", 60: "60", 90: "90"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Рестораны"),
                                        dcc.Dropdown(
                                            id="forecast-restaurant-filter",
                                            options=restaurant_options,
                                            value=[x["value"] for x in restaurant_options],
                                            multi=True,
                                            placeholder="Выберите рестораны",
                                        ),
                                    ]),
                                ],
                            ),
                            html.Div(
                                style={
                                    "backgroundColor": "white",
                                    "padding": "18px",
                                    "borderRadius": "14px",
                                    "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                    "marginTop": "20px",
                                },
                                children=[dcc.Loading(type="circle", children=dcc.Graph(id="forecast-chart"))],
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "20px",
                                    "marginTop": "20px",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "18px",
                                            "borderRadius": "14px",
                                            "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                        },
                                        children=[
                                            html.H3("Качество моделей"),
                                            dash_table.DataTable(
                                                id="model-metrics-table",
                                                page_size=10,
                                                sort_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "8px",
                                                    "fontFamily": "Arial",
                                                    "fontSize": "14px",
                                                },
                                                style_header={"fontWeight": "bold", "backgroundColor": "#eef2f7"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "18px",
                                            "borderRadius": "14px",
                                            "boxShadow": "0 4px 14px rgba(0,0,0,0.08)",
                                        },
                                        children=[
                                            html.H3("Сводка прогноза"),
                                            dash_table.DataTable(
                                                id="forecast-summary-table",
                                                page_size=10,
                                                sort_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "8px",
                                                    "fontFamily": "Arial",
                                                    "fontSize": "14px",
                                                },
                                                style_header={"fontWeight": "bold", "backgroundColor": "#eef2f7"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("main-analytics-chart", "figure"),
        Output("analytics-table", "data"),
        Output("analytics-table", "columns"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("restaurant-filter", "value"),
        Input("season-filter", "value"),
        Input("metric-selector", "value"),
        Input("dimension-1-selector", "value"),
        Input("dimension-2-selector", "value"),
        Input("chart-type-selector", "value"),
    )
    def update_analytics_chart(
        start_date,
        end_date,
        selected_restaurants,
        selected_seasons,
        metric,
        dimension_1,
        dimension_2,
        chart_type,
    ):
        filtered = filter_sales_data(
            data_source=data_source,
            selected_restaurants=selected_restaurants,
            selected_seasons=selected_seasons,
            start_date=start_date,
            end_date=end_date,
        )

        if filtered.empty:
            fig = go.Figure()
            fig.update_layout(template="plotly_white", title="Нет данных для выбранных фильтров", title_x=0.5)
            return fig, [], []

        aggregated, dimensions = aggregate_dashboard_data(
            data=filtered,
            metric=metric,
            dimension_1=dimension_1,
            dimension_2=dimension_2,
        )

        fig = make_dashboard_figure(
            aggregated=aggregated,
            dimensions=dimensions,
            metric=metric,
            chart_type=chart_type,
        )

        table_data = aggregated.copy()

        if "Full_Date" in table_data.columns:
            table_data["Full_Date"] = pd.to_datetime(table_data["Full_Date"]).dt.strftime("%Y-%m-%d")

        table_data["Value"] = table_data["Value"].round(2)

        table_columns = [
            {
                "name": DIMENSION_OPTIONS.get(column, column) if column != "Value" else METRIC_OPTIONS.get(metric, metric),
                "id": column,
            }
            for column in table_data.columns
        ]

        return fig, table_data.to_dict("records"), table_columns

    @app.callback(
        Output("forecast-chart", "figure"),
        Output("model-metrics-table", "data"),
        Output("model-metrics-table", "columns"),
        Output("forecast-summary-table", "data"),
        Output("forecast-summary-table", "columns"),
        Input("forecast-target", "value"),
        Input("forecast-model", "value"),
        Input("forecast-horizon", "value"),
        Input("forecast-restaurant-filter", "value"),
    )
    def update_forecast(target_column, selected_model, forecast_horizon, selected_restaurants):
        training_result = train_models_for_target(data_source, target_column)

        metrics_df = training_result["metrics_df"].copy()
        feature_columns = training_result["feature_columns"]
        final_models = training_result["final_models"]

        if selected_model == "Best_RMSE":
            model_name = training_result["best_model_name"]
        else:
            model_name = selected_model

        model = final_models[model_name]

        forecast_df = make_recursive_forecast(
            data_source=data_source,
            model=model,
            target_column=target_column,
            feature_columns=feature_columns,
            forecast_days=forecast_horizon,
        )

        if selected_restaurants:
            forecast_df = forecast_df[forecast_df["Restaurant_Name"].isin(selected_restaurants)].copy()

        target_label = FORECAST_TARGETS.get(target_column, target_column)

        if forecast_df.empty:
            fig = go.Figure()
            fig.update_layout(template="plotly_white", title="Нет данных для выбранных ресторанов", title_x=0.5)
            forecast_summary = pd.DataFrame()
        else:
            fig = px.line(
                forecast_df,
                x="Full_Date",
                y="Forecast",
                color="Restaurant_Name",
                markers=True,
                title=f"Прогноз: {target_label}. Модель: {model_name}. Горизонт: {forecast_horizon} дней",
                labels={
                    "Full_Date": "Дата",
                    "Forecast": f"Прогноз: {target_label}",
                    "Restaurant_Name": "Ресторан",
                },
            )

            fig.update_layout(
                template="plotly_white",
                title_x=0.5,
                height=650,
                hovermode="x unified",
            )

            forecast_summary = (
                forecast_df
                .groupby("Restaurant_Name", as_index=False)
                .agg(
                    Forecast_Total=("Forecast", "sum"),
                    Forecast_Avg_Daily=("Forecast", "mean"),
                    Forecast_Min_Daily=("Forecast", "min"),
                    Forecast_Max_Daily=("Forecast", "max"),
                )
                .sort_values("Forecast_Total", ascending=False)
            )

        metrics_view = metrics_df.copy()
        metrics_view[["MAE", "RMSE", "MAPE", "R2"]] = metrics_view[["MAE", "RMSE", "MAPE", "R2"]].round(4)

        metrics_columns = [{"name": column, "id": column} for column in metrics_view.columns]

        if not forecast_summary.empty:
            forecast_summary[
                ["Forecast_Total", "Forecast_Avg_Daily", "Forecast_Min_Daily", "Forecast_Max_Daily"]
            ] = forecast_summary[
                ["Forecast_Total", "Forecast_Avg_Daily", "Forecast_Min_Daily", "Forecast_Max_Daily"]
            ].round(2)

        forecast_summary_columns = [{"name": column, "id": column} for column in forecast_summary.columns]

        return (
            fig,
            metrics_view.to_dict("records"),
            metrics_columns,
            forecast_summary.to_dict("records"),
            forecast_summary_columns,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Интерактивный Dash-дэшборд для анализа ресторанного бизнеса и прогнозирования продаж."
    )
    parser.add_argument("--db", default="restaurant_dw.duckdb", help="Путь к DuckDB-файлу.")
    parser.add_argument("--host", default="127.0.0.1", help="Host для запуска Dash-сервера.")
    parser.add_argument("--port", type=int, default=8050, help="Port для запуска Dash-сервера.")
    parser.add_argument("--debug", action="store_true", help="Включить debug-режим Dash.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).resolve()

    if not db_path.exists():
        print(f"Файл базы данных не найден: {db_path}", file=sys.stderr)
        print("Передайте путь через --db или поместите restaurant_dw.duckdb рядом со скриптом.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Загрузка данных из базы: {db_path}")
    data_source = DashboardData(db_path=db_path)

    print(f"Строк продаж: {len(data_source.sales_df):,}".replace(",", " "))
    print(f"Ресторанов: {data_source.restaurant_dim['Restaurant_ID'].nunique()}")
    print(
        "Период данных: "
        f"{data_source.sales_df['Full_Date'].min().date()} — {data_source.sales_df['Full_Date'].max().date()}"
    )

    app = create_app(data_source)

    print(f"Дэшборд запущен: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
