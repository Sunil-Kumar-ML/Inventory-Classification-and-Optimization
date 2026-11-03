import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel(r"C:\Users\sunil\Downloads\Flipkart_project.xlsx")
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.info())

#now we will do forecasting of each of the 9 categories of products 
# Monthly demand columns
months = [
    "Jan_Demand","Feb_Demand","Mar_Demand","Apr_Demand",
    "May_Demand","Jun_Demand","Jul_Demand","Aug_Demand",
    "Sep_Demand","Oct_Demand","Nov_Demand","Dec_Demand"
]
# Group by ABC-XYZ class
grouped = df.groupby("Metrics")[months].sum()
print(grouped)

#Transpose the table
forecast_data = grouped.T
forecast_data.index = pd.date_range(
    start="2024-01-01",
    periods=12,
    freq="MS"
)
print(forecast_data)

#forecast the demand of next 6 months for class AX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
ax = forecast_data["AX"]
model = ExponentialSmoothing(
    ax,
    trend="add",
    seasonal=None
)
fit = model.fit()
forecast = fit.forecast(6)
print(forecast)

#plotting the forecasted demand
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(ax, label="Actual")

plt.plot(forecast.index,
         forecast.values,
         label="Forecast",
         linestyle="--")

plt.legend()

plt.title("AX Demand Forecast")

plt.show()

#forecasting for evry class
forecast_results = pd.DataFrame()

for cls in forecast_data.columns:

    ts = forecast_data[cls]

    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal=None
    )

    fit = model.fit()

    forecast = fit.forecast(6)

    forecast_results[cls] = forecast

forecast_results.index = pd.date_range(
    start="2025-01-01",
    periods=6,
    freq="MS"
)

print(forecast_results)

#plotting the forecast
import matplotlib.pyplot as plt

for cls in forecast_data.columns:

    plt.figure(figsize=(10,5))

    plt.plot(
        forecast_data.index,
        forecast_data[cls],
        marker="o",
        label="Actual"
    )

    plt.plot(
        forecast_results.index,
        forecast_results[cls],
        marker="o",
        linestyle="--",
        label="Forecast"
    )

    plt.title(f"{cls} Demand Forecast")

    plt.xlabel("Month")

    plt.ylabel("Demand")

    plt.legend()

    plt.grid(True)

    plt.show()

#we have forecasted only by exponential smoothing method now we will try different methods
# ============================================================
# DEMAND FORECASTING USING MULTIPLE MODELS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)

# ------------------------------------------------------------
# Function to calculate evaluation metrics
# ------------------------------------------------------------

def evaluate(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return mae, rmse, mape


# ------------------------------------------------------------
# DataFrames to store results
# ------------------------------------------------------------

forecast_results = pd.DataFrame()

comparison_results = []


# ------------------------------------------------------------
# Forecast every ABC-XYZ Class
# ------------------------------------------------------------

for cls in forecast_data.columns:

    print("="*60)
    print(f"Forecasting Class : {cls}")
    print("="*60)

    ts = forecast_data[cls]

    # --------------------------------------------------------
    # Train Test Split
    # --------------------------------------------------------

    train = ts.iloc[:10]

    test = ts.iloc[10:]


    # ========================================================
    # 1. Moving Average
    # ========================================================

    moving_average = train.mean()

    forecast_ma = np.repeat(
        moving_average,
        len(test)
    )

    mae_ma, rmse_ma, mape_ma = evaluate(
        test,
        forecast_ma
    )


    # ========================================================
    # 2. Simple Exponential Smoothing
    # ========================================================

    ses_model = SimpleExpSmoothing(train).fit()

    forecast_ses = ses_model.forecast(len(test))

    mae_ses, rmse_ses, mape_ses = evaluate(
        test,
        forecast_ses
    )


    # ========================================================
    # 3. Holt Linear Trend
    # ========================================================

    holt_model = Holt(train).fit()

    forecast_holt = holt_model.forecast(len(test))

    mae_holt, rmse_holt, mape_holt = evaluate(
        test,
        forecast_holt
    )


    # ========================================================
    # 4. Holt-Winters
    # ========================================================

    hw_model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None
    ).fit()

    forecast_hw = hw_model.forecast(len(test))

    mae_hw, rmse_hw, mape_hw = evaluate(
        test,
        forecast_hw
    )


    # ========================================================
    # Comparison Table
    # ========================================================

    comparison = pd.DataFrame({

        "Model":[
            "Moving Average",
            "SES",
            "Holt",
            "Holt-Winters"
        ],

        "MAE":[
            mae_ma,
            mae_ses,
            mae_holt,
            mae_hw
        ],

        "RMSE":[
            rmse_ma,
            rmse_ses,
            rmse_holt,
            rmse_hw
        ],

        "MAPE":[
            mape_ma,
            mape_ses,
            mape_holt,
            mape_hw
        ]

    })

    print(comparison)


    # ========================================================
    # Best Model
    # ========================================================

    best_model = comparison.sort_values(
        "RMSE"
    ).iloc[0]["Model"]

    print(f"\nBest Model : {best_model}")


    # ========================================================
    # Retrain on complete data
    # ========================================================

    if best_model == "Moving Average":

        final_forecast = np.repeat(
            ts.mean(),
            6
        )

    elif best_model == "SES":

        model = SimpleExpSmoothing(ts).fit()

        final_forecast = model.forecast(6)

    elif best_model == "Holt":

        model = Holt(ts).fit()

        final_forecast = model.forecast(6)

    else:

        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal=None
        ).fit()

        final_forecast = model.forecast(6)


    # ========================================================
    # Store Forecast
    # ========================================================

    forecast_results[cls] = final_forecast


    comparison_results.append({

        "Class":cls,

        "Best Model":best_model,

        "MAE":comparison["MAE"].min(),

        "RMSE":comparison["RMSE"].min(),

        "MAPE":comparison["MAPE"].min()

    })


# ============================================================
# Forecast Months
# ============================================================

forecast_results.index = pd.date_range(

    start="2025-01-01",

    periods=6,

    freq="MS"

)


# ============================================================
# Comparison Results
# ============================================================

comparison_results = pd.DataFrame(comparison_results)


print("\nForecast Results")
print(forecast_results)

print("\nModel Comparison")
print(comparison_results)


# ============================================================
# Save Files
# ============================================================

# forecast_results.to_excel(
#     "ABC_XYZ_Forecast.xlsx"
# )

# comparison_results.to_excel(
#     "Forecast_Model_Comparison.xlsx",
#     index=False
# )

# print("\nFiles Saved Successfully.")

#plotting all graphs
import matplotlib.pyplot as plt

for cls in forecast_data.columns:

    plt.figure(figsize=(10,5))

    plt.plot(
        forecast_data.index,
        forecast_data[cls],
        marker="o",
        label="Actual"
    )

    plt.plot(
        forecast_results.index,
        forecast_results[cls],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Forecast"
    )

    plt.title(f"{cls} Demand Forecast")

    plt.xlabel("Month")

    plt.ylabel("Demand")

    plt.grid(True)

    plt.legend()

    plt.show()
