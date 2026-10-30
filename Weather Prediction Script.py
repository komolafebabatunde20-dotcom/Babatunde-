import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_and_compare_weather():
    # 1. Simulate / Load Historical Data for Nigeria (2017 vs Current Year)
    np.random.seed(42)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Baseline 2017 Data (Typical historical records for average temperature and rainfall)
    temp_2017 = [27.5, 29.0, 30.5, 31.0, 29.5, 27.0, 25.5, 25.0, 26.0, 27.5, 28.5, 27.8]
    rain_2017 = [15.0, 20.0, 45.0, 120.0, 180.0, 250.0, 300.0, 280.0, 310.0, 190.0, 40.0, 10.0]

    # Current Climate Trend Modeling (~0.8°C warming shift & altered rainfall patterns)
    temp_current_predicted = [t + np.random.normal(0.8, 0.2) for t in temp_2017]
    rain_current_predicted = [r * np.random.normal(0.95, 0.08) for r in rain_2017]

    df_weather = pd.DataFrame({
        "Month": months,
        "Temp_2017": temp_2017,
        "Temp_Current": temp_current_predicted,
        "Rain_2017": rain_2017,
        "Rain_Current": rain_current_predicted
    })

    print("--- Weather Comparison Data Overview (2017 vs. Current) ---")
    print(df_weather.head())

    # 2. Predictive Analysis using Linear Regression
    def predict_trend(data_2017, data_current):
        X = np.array([2017, 2026]).reshape(-1, 1)  # Comparing 2017 to current year
        predictions = []
        for y17, y_curr in zip(data_2017, data_current):
            y = np.array([y17, y_curr]).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            # Predict for upcoming year
            next_year_pred = model.predict(np.array([[2027]]))[0][0]
            predictions.append(round(next_year_pred, 2))
        return predictions

    df_weather["Temp_Next_Year_Pred"] = predict_trend(df_weather["Temp_2017"], df_weather["Temp_Current"])

    print("\n--- Temperature Predictions for Next Year based on Trends ---")
    print(df_weather[["Month", "Temp_2017", "Temp_Current", "Temp_Next_Year_Pred"]])

    # 3. Plotting the Comparison
    plt.figure(figsize=(14, 6))

    # Temperature Comparison Subplot
    plt.subplot(1, 2, 1)
    plt.plot(df_weather["Month"], df_weather["Temp_2017"], marker='o', linestyle='--', color='blue', label='2017 Temperature (°C)')
    plt.plot(df_weather["Month"], df_weather["Temp_Current"], marker='o', linestyle='-', color='red', label='Current Temperature (°C)')
    plt.title("Nigeria Monthly Temperature: 2017 vs Current")
    plt.xlabel("Months")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)

    # Rainfall Comparison Subplot
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(months)) - 0.2, df_weather["Rain_2017"], width=0.4, label='2017 Rainfall (mm)', color='skyblue')
    plt.bar(np.arange(len(months)) + 0.2, df_weather["Rain_Current"], width=0.4, label='Current Rainfall (mm)', color='orange')
    plt.xticks(np.arange(len(months)), months)
    plt.title("Nigeria Monthly Rainfall: 2017 vs Current")
    plt.xlabel("Months")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_compare_weather()