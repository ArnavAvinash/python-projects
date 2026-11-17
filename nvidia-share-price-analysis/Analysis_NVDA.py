# %%
%pip install yfinance pandas


# %%
import yfinance as yf
import pandas as pd

nvda = yf.download(
    "NVDA",
    period="1y",
    interval="1d",
    auto_adjust=False
)

nvda.head()

# %%
nvda.to_csv("NVDA_historical_data.csv")


# %%
qqq = yf.download("QQQ", period="1y", interval="1d",
                  auto_adjust=False)

soxx = yf.download("SOXX", period="1y", interval="1d",
                   auto_adjust=False)

qqq.to_csv("QQQ_historical_data.csv")
soxx.to_csv("SOXX_historical_data.csv")

# %%
qqq = yf.download("QQQ", period="1y", interval="1d",
                  auto_adjust=False)

soxx = yf.download("SOXX", period="1y", interval="1d",
                   auto_adjust=False)



# %%
print("Number of rows:", len(nvda))
print("Number of columns:", len(nvda.columns))

display(nvda.tail())

# %%
print(nvda.columns.tolist())
print(nvda.dtypes)

# %%
print(nvda.isnull().sum())
print(nvda.index.min())
print(nvda.index.max())

# %%
qqq = yf.download(
    "QQQ",
    period="1y",
    interval="1d",
    auto_adjust=False
)

soxx = yf.download(
    "SOXX",
    period="1y",
    interval="1d",
    auto_adjust=False
)

print("QQQ:", qqq.shape)
print("SOXX:", soxx.shape)

# %%
prices = pd.concat(
    [
        nvda["Adj Close"].rename("NVDA"),
        qqq["Adj Close"].rename("QQQ"),
        soxx["Adj Close"].rename("SOXX")
    ],
    axis=1
)

print(prices.head())
print(prices.shape)
print(prices.isnull().sum())

# %%
print(nvda.columns)

# %%
print(nvda.head(2))

# %%
prices = pd.concat(
    [
        nvda[("Adj Close", "NVDA")].rename("NVDA"),
        qqq[("Adj Close", "QQQ")].rename("QQQ"),
        soxx[("Adj Close", "SOXX")].rename("SOXX")
    ],
    axis=1
)

print(prices.head())
print(prices.shape)
print(prices.isnull().sum())

# %%
returns = prices.pct_change() * 100

returns = returns.dropna()

print(returns.head())
print(returns.shape)
print(returns.isnull().sum())

# %%
stats = returns.describe().round(4)

display(stats)

# %%
correlation = returns.corr()

display(correlation.round(4))

# %%
%pip install statsmodels

# %%
import statsmodels.api as sm

# NVIDIA against QQQ
X_qqq = sm.add_constant(returns["QQQ"])
model_qqq = sm.OLS(returns["NVDA"], X_qqq).fit()

# NVIDIA against SOXX
X_soxx = sm.add_constant(returns["SOXX"])
model_soxx = sm.OLS(returns["NVDA"], X_soxx).fit()

print("NVIDIA vs QQQ")
print(model_qqq.summary())

print("\nNVIDIA vs SOXX")
print(model_soxx.summary())

# %%
print("SOXX beta:", model_soxx.params["SOXX"])
print("SOXX R-squared:", model_soxx.rsquared)
print("SOXX beta p-value:", model_soxx.pvalues["SOXX"])
print("SOXX intercept:", model_soxx.params["const"])
print("SOXX Durbin-Watson:",
      sm.stats.stattools.durbin_watson(model_soxx.resid))

# %%
nvda_residuals = model_qqq.resid

print(nvda_residuals.head())

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    nvda_residuals.index,
    nvda_residuals,
    label="NVIDIA benchmark-adjusted returns"
)

plt.axhline(0, linestyle="--")

plt.title("NVIDIA: Daily Benchmark-Adjusted Returns")
plt.xlabel("Date")
plt.ylabel("Residual Return (%)")
plt.legend()
plt.grid(True)

plt.show()

# %%
%pip install matplotlib

# %%
import matplotlib.pyplot as plt

# %%
plt.figure(figsize=(12, 5))

plt.plot(
    nvda_residuals.index,
    nvda_residuals,
    label="NVIDIA benchmark-adjusted returns"
)

plt.axhline(0, linestyle="--")

plt.title("NVIDIA: Daily Benchmark-Adjusted Returns")
plt.xlabel("Date")
plt.ylabel("Residual Return (%)")
plt.legend()
plt.grid(True)

plt.show()

# %%
# Calculate the returns predicted by the QQQ regression
predicted_returns = model_qqq.predict(
    sm.add_constant(returns["QQQ"])
)

# Calculate cumulative relative performance
cumulative_relative = (
    ((1 + returns["NVDA"] / 100) /
     (1 + predicted_returns / 100))
    .cumprod() - 1
) * 100

print(cumulative_relative.tail())

# %%
plt.figure(figsize=(12, 5))

plt.plot(
    cumulative_relative.index,
    cumulative_relative,
    label="Cumulative relative performance"
)

plt.axhline(0, linestyle="--")

plt.title("NVIDIA: Cumulative Benchmark-Adjusted Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative relative performance (%)")
plt.legend()
plt.grid(True)

plt.show()

# %%
predicted_returns = model_qqq.predict(
    sm.add_constant(returns["QQQ"])
)

recent = pd.DataFrame({
    "NVDA_actual": returns["NVDA"],
    "QQQ_return": returns["QQQ"],
    "NVDA_predicted": predicted_returns,
    "Residual": nvda_residuals
})

display(recent.tail(10).round(3))

# %%
from statsmodels.tsa.stattools import adfuller

# Convert cumulative relative performance into log form
relative_log = np.log1p(cumulative_relative / 100)

# Augmented Dickey-Fuller stationarity test
adf_result = adfuller(relative_log.dropna())

print("ADF statistic:", adf_result[0])
print("P-value:", adf_result[1])
print("Critical values:", adf_result[4])

# %%
import numpy as np

# %%
relative_log = np.log1p(cumulative_relative / 100)

adf_result = adfuller(relative_log.dropna())

print("ADF statistic:", adf_result[0])
print("P-value:", adf_result[1])
print("Critical values:", adf_result[4])

# %%
adf_residuals = adfuller(nvda_residuals.dropna())

print("ADF statistic:", adf_residuals[0])
print("P-value:", adf_residuals[1])
print("Critical values:", adf_residuals[4])

# %%
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(nvda_residuals, lags=20)
plt.title("Autocorrelation of NVIDIA Benchmark-Adjusted Returns")
plt.show()

# %%

print(prices.loc["2026-10-26":"2026-11-13"])

# %%
import pandas as pd
import statsmodels.api as sm

# Ensure dates are in datetime format
returns.index = pd.to_datetime(returns.index)

# Use only observations before the first public announcement
pre_event_returns = returns.loc[
    returns.index < "2026-10-29"
]

# Estimate the benchmark relationship using pre-event data
event_model = sm.OLS(
    pre_event_returns["NVDA"],
    sm.add_constant(pre_event_returns["QQQ"])
).fit()

print(event_model.summary())

# %%
# Predict NVDA returns using the pre-event benchmark model
predicted_event_returns = event_model.predict(
    sm.add_constant(returns["QQQ"])
)

# Calculate abnormal returns
event_abnormal_returns = (
    returns["NVDA"] - predicted_event_returns
)

print(event_abnormal_returns.loc["2026-10-26":])

# %%
# Plot daily abnormal returns
plt.figure(figsize=(13, 6))

plt.plot(
    event_abnormal_returns.loc["2026-10-26":"2026-11-13"],
    marker="o"
)

# Zero abnormal-return reference
plt.axhline(0, linestyle="--", linewidth=1)

# Mark important announcement dates
events = {
    "First revelation": "2026-10-29",
    "Acknowledgement": "2026-11-09",
    "Second report": "2026-11-11",
    "Resignation": "2026-11-12"
}

for label, date in events.items():
    plt.axvline(
        pd.Timestamp(date),
        linestyle=":",
        alpha=0.7,
        label=label
    )

plt.title("NVDA Daily Abnormal Returns Around Public Disclosures")
plt.ylabel("Abnormal return (%)")
plt.xlabel("Date")
plt.legend()
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Select the event period
event_period = event_abnormal_returns.loc[
    "2026-10-28":"2026-11-13"
]

# Calculate cumulative abnormal returns
cumulative_abnormal_returns = event_period.cumsum()

# Plot cumulative abnormal returns
plt.figure(figsize=(13, 6))

plt.plot(
    cumulative_abnormal_returns,
    marker="o"
)

plt.axhline(0, linestyle="--", linewidth=1)

# Mark the announcement dates
for label, date in events.items():
    plt.axvline(
        pd.Timestamp(date),
        linestyle=":",
        alpha=0.7,
        label=label
    )

plt.title("NVDA Cumulative Abnormal Returns During Public Disclosures")
plt.ylabel("Cumulative abnormal return (%)")
plt.xlabel("Date")
plt.legend()
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(cumulative_abnormal_returns.round(3))

# %%
# Start the event study on the first public announcement
event_period = event_abnormal_returns.loc[
    "2026-10-29":"2026-11-13"
]

# Recalculate cumulative abnormal returns
car_from_first_announcement = event_period.cumsum()

# Plot the revised CAR
plt.figure(figsize=(13, 6))

plt.plot(
    car_from_first_announcement,
    marker="o"
)

plt.axhline(0, linestyle="--", linewidth=1)

for label, date in events.items():
    plt.axvline(
        pd.Timestamp(date),
        linestyle=":",
        alpha=0.7,
        label=label
    )

plt.title("NVDA Cumulative Abnormal Returns Since First Public Revelation")
plt.ylabel("Cumulative abnormal return (%)")
plt.xlabel("Date")
plt.legend()
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(car_from_first_announcement.round(3))

# %%
from scipy import stats
import numpy as np

# Pre-event residual volatility
sigma = np.sqrt(event_model.mse_resid)

# Event-period cumulative abnormal return
car_total = car_from_first_announcement.sum()

# Number of trading observations
n = len(event_period)

# Standard error of cumulative abnormal return
car_se = sigma * np.sqrt(n)

# Calculate t-statistic
car_t = car_total / car_se

# Two-sided p-value
car_p = 2 * stats.t.sf(abs(car_t), df=event_model.df_resid)

print("Pre-event residual standard deviation:", round(sigma, 4))
print("Total CAR:", round(car_total, 4), "%")
print("CAR standard error:", round(car_se, 4))
print("t-statistic:", round(car_t, 4))
print("p-value:", round(car_p, 6))

# %%
car_total = car_from_first_announcement.sum()

# %%
# Correct total cumulative abnormal return
car_total = event_period.sum()

# Alternatively:
car_total = car_from_first_announcement.iloc[-1]

# Recalculate test statistics
sigma = np.sqrt(event_model.mse_resid)
n = len(event_period)

car_se = sigma * np.sqrt(n)
car_t = car_total / car_se
car_p = 2 * stats.t.sf(abs(car_t), df=event_model.df_resid)

print("Correct CAR:", round(car_total, 4), "%")
print("Standard error:", round(car_se, 4))
print("t-statistic:", round(car_t, 4))
print("p-value:", round(car_p, 6))

# %%
event_windows = {
    "First revelation": ("2026-10-28", "2026-10-30"),
    "Huang acknowledgement": ("2026-11-06", "2026-11-10"),
    "Second relationship report": ("2026-11-10", "2026-11-12"),
    "Resignation": ("2026-11-11", "2026-11-13")
}

event_summary = []

sigma = np.sqrt(event_model.mse_resid)

for event, (start, end) in event_windows.items():

    window = event_abnormal_returns.loc[start:end]

    car = window.sum()
    n = len(window)
    standard_error = sigma * np.sqrt(n)

    t_stat = car / standard_error
    p_value = 2 * stats.t.sf(
        abs(t_stat),
        df=event_model.df_resid
    )

    event_summary.append({
        "Event": event,
        "CAR (%)": car,
        "Standard Error": standard_error,
        "t-statistic": t_stat,
        "p-value": p_value
    })

event_table = pd.DataFrame(event_summary)

print(event_table.round(5).to_string(index=False))

# %%
# Display daily abnormal returns around the announcements

daily_event_table = event_results.loc[
    "2026-10-28":"2026-11-13"
].copy()

daily_event_table["Event"] = ""

daily_event_table.loc["2026-10-29", "Event"] = "First revelation"
daily_event_table.loc["2026-11-09", "Event"] = "Acknowledgement"
daily_event_table.loc["2026-11-11", "Event"] = "Second relationship"
daily_event_table.loc["2026-11-12", "Event"] = "Resignation"

print(daily_event_table.round(4).to_string())

# %%
# Reconstruct the event results table

event_results = pd.DataFrame({
    "NVDA_actual": returns["NVDA"],
    "QQQ_actual": returns["QQQ"],
    "NVDA_predicted": predicted_event_returns,
    "Abnormal_return": event_abnormal_returns
})

# %%
daily_event_table = event_results.loc[
    "2026-10-28":"2026-11-13"
].copy()

daily_event_table["Event"] = ""

daily_event_table.loc["2026-10-29", "Event"] = "First revelation"
daily_event_table.loc["2026-11-09", "Event"] = "Acknowledgement"
daily_event_table.loc["2026-11-11", "Event"] = "Second relationship"
daily_event_table.loc["2026-11-12", "Event"] = "Resignation"

print(daily_event_table.round(4).to_string())

# %%
phases = {
    "Initial disclosure": ("2026-10-29", "2026-10-30"),
    "Between disclosures": ("2026-11-02", "2026-11-06"),
    "Acknowledgement aftermath": ("2026-11-09", "2026-11-10"),
    "Final developments": ("2026-11-11", "2026-11-13")
}

phase_results = []

for phase, (start, end) in phases.items():

    daily = event_abnormal_returns.loc[start:end]

    phase_results.append({
        "Phase": phase,
        "Trading days": len(daily),
        "CAR (%)": daily.sum(),
        "Average daily abnormal return (%)": daily.mean()
    })

phase_table = pd.DataFrame(phase_results)

print(phase_table.round(4).to_string(index=False))

# %%
# Calculate cumulative abnormal performance through the end
# of the available one-year sample

full_car = event_abnormal_returns.loc[
    "2026-10-29":
].cumsum()

# Plot cumulative abnormal performance
plt.figure(figsize=(13, 6))

plt.plot(full_car, marker="o", markersize=3)

plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(
    pd.Timestamp("2026-11-12"),
    linestyle=":",
    label="Resignation"
)

plt.title("NVDA Cumulative Abnormal Performance: Event and Aftermath")
plt.ylabel("Cumulative abnormal return (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

# Compare the event CAR with historical 12-day periods

pre_event_abnormal = event_abnormal_returns.loc[
    event_abnormal_returns.index < "2026-10-29"
]

historical_12day_car = pre_event_abnormal.rolling(12).sum().dropna()

event_car = event_abnormal_returns.loc["2026-10-29":"2026-11-13"].sum()

print("Event-period CAR:", round(event_car, 2), "%")
print("Worst historical 12-day CAR:",
      round(historical_12day_car.min(), 2), "%")

print("Historical percentile:",
      round((historical_12day_car < event_car).mean() * 100, 2), "%")


# %%

# Estimate the pre-event NVDA relationship with SOXX

soxx_model = sm.OLS(
    pre_event_returns["NVDA"],
    sm.add_constant(pre_event_returns["SOXX"])
).fit()

soxx_predicted = soxx_model.predict(
    sm.add_constant(returns["SOXX"])
)

soxx_abnormal = returns["NVDA"] - soxx_predicted

soxx_event_car = soxx_abnormal.loc[
    "2026-10-29":"2026-11-13"
].sum()

print("QQQ-based event CAR:", round(event_car, 2), "%")
print("SOXX-based event CAR:", round(soxx_event_car, 2), "%")


# %%

# Calculate cumulative abnormal performance from the first event date

cumulative_abnormal = event_abnormal_returns.loc[
    "2026-10-29":
].cumsum()

# The event-period CAR establishes the initial shortfall
event_car = cumulative_abnormal.loc["2026-10-29":"2026-11-13"].iloc[-1]

# Only observations AFTER the event period count as recovery
recovery_path = cumulative_abnormal.loc[
    cumulative_abnormal.index > "2026-11-13"
]

# Recovery milestones
milestones = {
    "25% recovered": event_car * 0.75,
    "50% recovered": event_car * 0.50,
    "75% recovered": event_car * 0.25,
    "100% recovered": 0
}

for milestone, threshold in milestones.items():
    dates = recovery_path[recovery_path >= threshold]

    if len(dates) > 0:
        print(milestone, ":", dates.index[0].date())
    else:
        print(milestone, ": Not yet reached")


# %%

# Identify historical negative 12-day abnormal-performance episodes

historical_12day_car = pre_event_abnormal.rolling(12).sum()

# Identify particularly severe negative periods
severe_periods = historical_12day_car[
    historical_12day_car <= -5
]

print("Number of severe historical observations:",
      len(severe_periods))

print(severe_periods.nsmallest(10))


# %%

# Group consecutive severe observations into separate episodes

severe = historical_12day_car.dropna()
severe = severe[severe <= -5]

# Identify breaks between consecutive trading-day observations
groups = severe.index.to_series().diff().dt.days.gt(5).cumsum()

# Select the worst observation within each episode
episodes = severe.groupby(groups).idxmin()

# Create a summary of distinct episodes
episode_summary = pd.DataFrame({
    "Trough_date": episodes.values,
    "12_day_CAR": [
        severe.loc[date] for date in episodes.values
    ]
})

print(episode_summary.to_string(index=False))
print("\nNumber of distinct episodes:", len(episode_summary))


# %%

# Calculate historical recovery times

recovery_results = []

for _, episode in episode_summary.iterrows():

    trough_date = pd.Timestamp(episode["Trough_date"])
    initial_loss = episode["12_day_CAR"]

    # Only use trading days after the episode's trough date
    future_returns = pre_event_abnormal[
        pre_event_abnormal.index > trough_date
    ]

    # Accumulate subsequent abnormal returns
    cumulative_recovery = future_returns.cumsum()

    # Recovery thresholds
    half_target = abs(initial_loss) * 0.50
    full_target = abs(initial_loss)

    half_dates = cumulative_recovery[
        cumulative_recovery >= half_target
    ]

    full_dates = cumulative_recovery[
        cumulative_recovery >= full_target
    ]

    # Record recovery times in trading days
    half_days = (
        future_returns.index.get_loc(half_dates.index[0]) + 1
        if len(half_dates) > 0 else None
    )

    full_days = (
        future_returns.index.get_loc(full_dates.index[0]) + 1
        if len(full_dates) > 0 else None
    )

    recovery_results.append({
        "Episode": trough_date.date(),
        "Initial_CAR": round(initial_loss, 2),
        "50pct_Recovery_Days": half_days,
        "Full_Recovery_Days": full_days
    })

recovery_table = pd.DataFrame(recovery_results)

print(recovery_table.to_string(index=False))


# %%

# Examine whether shock severity predicts recovery duration

import scipy.stats as stats

severity = recovery_table["Initial_CAR"].abs()
half_recovery = recovery_table["50pct_Recovery_Days"]

slope, intercept, r_value, p_value, std_err = stats.linregress(
    severity, half_recovery
)

print("Relationship between loss severity and recovery time")
print("Correlation:", round(r_value, 3))
print("R-squared:", round(r_value**2, 3))
print("P-value:", round(p_value, 4))
print("Estimated additional recovery days per 1% loss:",
      round(slope, 2))

# Exploratory extrapolation to the current 19% shock
forecast_19 = intercept + slope * 19

print("Exploratory 50% recovery estimate for 19% shock:",
      round(forecast_19, 1), "trading days")


# %%

# Estimate recovery initiation lag for each historical episode

lag_results = []

for _, episode in episode_summary.iterrows():

    date = pd.Timestamp(episode["Trough_date"])

    future = pre_event_abnormal.loc[
        pre_event_abnormal.index > date
    ]

    initiation_lag = None

    for i in range(len(future) - 2):
        if (future.iloc[i:i+3] > 0).all():
            initiation_lag = i + 1
            break

    lag_results.append({
        "Episode": date.date(),
        "Recovery_Initiation_Lag": initiation_lag
    })

lag_table = pd.DataFrame(lag_results)

print(lag_table.to_string(index=False))
print("\nMedian initiation lag:",
      lag_table["Recovery_Initiation_Lag"].median())


# %%

# Summarise historical recovery behaviour

print("Initiation lag statistics:")
print(lag_table["Recovery_Initiation_Lag"].describe())

print("\n50% recovery statistics:")
print(recovery_table["50pct_Recovery_Days"].describe())

print("\nFull recovery statistics:")
print(recovery_table["Full_Recovery_Days"].describe())


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Actual cumulative abnormal performance
actual = event_abnormal_returns.loc["2026-10-29":"2026-11-13"].dropna()
actual_car = actual.cumsum()

# 2. Forecast horizons (trading days)
horizons = {
    "Neutral (23 days)": 23,
    "Prudent (45 days)": 45,
    "Pessimistic (59 days)": 59
}

# Approximate future trading dates (weekends excluded)
future_dates = pd.bdate_range(
    start=actual_car.index[-1] + pd.offsets.BDay(1),
    periods=59
)

# 3. Plot actual performance and hypothetical recovery paths
plt.figure(figsize=(13, 7))

plt.plot(
    actual_car.index, actual_car.values,
    color="black", linewidth=2.5, label="Actual cumulative abnormal performance"
)

plt.axhline(0, color="grey", linestyle="--", linewidth=1.2,
            label="Full recovery (0%)")

for label, days in horizons.items():
    forecast = np.linspace(actual_car.iloc[-1], 0, days + 1)[1:]

    dates = future_dates[:days]

    plt.plot(
        dates, forecast,
        linestyle="--", linewidth=2,
        label=label
    )

    plt.scatter(dates[-1], forecast[-1], s=45)

plt.title("NVDA: Actual and Projected Benchmark-Adjusted Recovery",
          fontsize=14)
plt.xlabel("Date")
plt.ylabel("Cumulative abnormal performance (%)")
plt.legend()
plt.grid(alpha=0.25)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Starting cumulative abnormal performance
starting_car = actual_car.iloc[-1]

# Simulation settings
n_simulations = 5000
forecast_days = 59

# Historical daily abnormal returns
historical_returns = pre_event_abnormal.dropna().values

# Bootstrap: randomly sample historical daily abnormal returns
np.random.seed(42)

simulated_returns = np.random.choice(
    historical_returns,
    size=(n_simulations, forecast_days),
    replace=True
)

# Calculate cumulative future performance
simulated_car = starting_car + np.cumsum(simulated_returns, axis=1)

# %%
# Full recovery occurs when cumulative abnormal performance reaches 0
recovery_probabilities = {}

for day in [23, 45, 59]:
    recovered = np.any(simulated_car[:, :day] >= 0, axis=1)

    probability = recovered.mean() * 100

    recovery_probabilities[day] = probability

    print(f"Probability of full recovery within {day} days: "
          f"{probability:.2f}%")

# %%
# Future trading dates
future_dates = pd.bdate_range(
    start=actual_car.index[-1] + pd.offsets.BDay(1),
    periods=forecast_days
)

# Calculate percentile bands
median_path = np.percentile(simulated_car, 50, axis=0)
lower_band = np.percentile(simulated_car, 5, axis=0)
upper_band = np.percentile(simulated_car, 95, axis=0)

# Plot
plt.figure(figsize=(13, 7))

# Actual performance
plt.plot(
    actual_car.index,
    actual_car.values,
    color="black",
    linewidth=2.5,
    label="Actual cumulative abnormal performance"
)

# Simulated median
plt.plot(
    future_dates,
    median_path,
    linestyle="--",
    linewidth=2,
    label="Median simulated path"
)

# 90% simulation interval
plt.fill_between(
    future_dates,
    lower_band,
    upper_band,
    alpha=0.2,
    label="5th–95th percentile simulation range"
)

# Full recovery threshold
plt.axhline(
    0,
    color="grey",
    linestyle=":",
    label="Full recovery (0%)"
)

# Separate observed and forecast periods
plt.axvline(
    actual_car.index[-1],
    color="grey",
    linestyle="--",
    alpha=0.7,
    label="Forecast begins"
)

plt.title("NVDA: Probabilistic Benchmark-Adjusted Recovery Simulation")
plt.xlabel("Date")
plt.ylabel("Cumulative abnormal performance (%)")
plt.legend()
plt.grid(alpha=0.25)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# %%
print(recovery_probabilities)

# %%
import pandas as pd
import numpy as np

# Historical cumulative abnormal performance
historical_car = pre_event_abnormal.cumsum()

# Identify previous cumulative performance peaks
historical_peak = historical_car.cummax()

# Calculate drawdown from previous peak
historical_drawdown = historical_car - historical_peak

# Identify severe drawdowns of at least 5 percentage points
severe_days = historical_drawdown[historical_drawdown <= -5]

print("Number of severe drawdown observations:", len(severe_days))
print("Worst historical drawdown:", historical_drawdown.min())
print("Date of worst drawdown:", historical_drawdown.idxmin())

# %%
# Group consecutive severe drawdown observations into episodes
groups = severe_days.index.to_series().diff().dt.days.gt(5).cumsum()

episodes = []

for _, group in severe_days.groupby(groups):

    trough_date = group.idxmin()
    trough_value = historical_car.loc[trough_date]

    # Previous peak immediately before the trough
    peak_value = historical_peak.loc[trough_date]

    # Find the first subsequent date cumulative performance
    # returns to the previous peak
    future = historical_car.loc[trough_date:]

    recovered = future[future >= peak_value]

    if len(recovered) > 1:
        recovery_date = recovered.index[1] if recovered.index[0] == trough_date else recovered.index[0]

        recovery_days = len(
            historical_car.loc[trough_date:recovery_date]
        ) - 1
    else:
        recovery_date = None
        recovery_days = np.nan

    episodes.append({
        "Trough date": trough_date,
        "Drawdown (%)": trough_value - peak_value,
        "Recovery date": recovery_date,
        "Recovery days": recovery_days
    })

recovery_table = pd.DataFrame(episodes)

print(recovery_table)

# %%
# Identify periods below the previous cumulative performance peak
underwater = historical_drawdown < 0

episodes = []
start = None

for i in range(len(underwater)):

    if underwater.iloc[i] and start is None:
        start = i

    # End of an underwater episode
    if start is not None and (
        not underwater.iloc[i] or i == len(underwater) - 1
    ):

        end = i - 1 if not underwater.iloc[i] else i

        period = historical_drawdown.iloc[start:end+1]

        # Identify deepest trough
        trough_date = period.idxmin()
        drawdown = period.min()

        # Keep only drawdowns of at least 5 percentage points
        if drawdown <= -5:

            if not underwater.iloc[end]:
                recovery_date = historical_drawdown.index[end + 1]
                recovery_days = end + 1 - period.index.get_loc(trough_date) - start
            else:
                recovery_date = None
                recovery_days = np.nan

            # Calculate trading days directly
            if recovery_date is not None:
                recovery_days = (
                    historical_drawdown.index.get_loc(recovery_date)
                    - historical_drawdown.index.get_loc(trough_date)
                )

            episodes.append({
                "Trough date": trough_date,
                "Drawdown (%)": drawdown,
                "Recovery date": recovery_date,
                "Recovery days": recovery_days
            })

        start = None

corrected_recovery_table = pd.DataFrame(episodes)

print(corrected_recovery_table)

# %%
episodes = []
start = None

for i in range(len(underwater)):

    if underwater.iloc[i] and start is None:
        start = i

    if start is not None and (
        not underwater.iloc[i] or i == len(underwater) - 1
    ):

        # Last day below the previous peak
        end = i - 1 if not underwater.iloc[i] else i

        period = historical_drawdown.iloc[start:end+1]

        trough_date = period.idxmin()
        drawdown = period.min()

        if drawdown <= -5:

            if not underwater.iloc[i]:
                recovery_date = historical_drawdown.index[i]

                recovery_days = (
                    historical_drawdown.index.get_loc(recovery_date)
                    - historical_drawdown.index.get_loc(trough_date)
                )
            else:
                recovery_date = None
                recovery_days = np.nan

            episodes.append({
                "Trough date": trough_date,
                "Drawdown (%)": drawdown,
                "Recovery date": recovery_date,
                "Recovery days": recovery_days
            })

        start = None

corrected_recovery_table = pd.DataFrame(episodes)

print(corrected_recovery_table)

# %%
# Current event CAR
starting_car = actual_car.iloc[-1]

# Target: 0.1 basis points = 0.001 percentage points
target_car = 0.001

# Historical neutral recovery estimate
neutral_days = 48

# Calculate required improvement
required_recovery = target_car - starting_car

print("Current CAR:", round(starting_car, 4), "%")
print("Target CAR:", target_car, "%")
print("Required recovery:", round(required_recovery, 4), "percentage points")

# Approximate future date (weekends excluded)
last_observation = actual_car.index[-1]

neutral_date = pd.bdate_range(
    start=last_observation + pd.offsets.BDay(1),
    periods=neutral_days
)[-1]

print("Neutral recovery day:", neutral_days)
print("Estimated neutral recovery date:", neutral_date.date())

# %%
sept_trough = pd.Timestamp("2026-09-14")
sample_end = historical_car.index[-1]

censored_days = (
    historical_car.index.get_loc(sample_end)
    - historical_car.index.get_loc(sept_trough)
)

print("Unrecovered historical duration:", censored_days, "trading days")

# %%
from lifelines import KaplanMeierFitter

# Completed recovery durations and censored observation
durations = [48, 40, 48, 32]
observed = [True, True, True, False]

kmf = KaplanMeierFitter()

kmf.fit(
    durations=durations,
    event_observed=observed
)

print("Estimated median recovery duration:",
      kmf.median_survival_time_)

print(kmf.survival_function_)

# %%
%pip install lifelines

# %%
from lifelines import KaplanMeierFitter

# Completed recovery durations and censored observation
durations = [48, 40, 48, 32]
observed = [True, True, True, False]

kmf = KaplanMeierFitter()

kmf.fit(
    durations=durations,
    event_observed=observed
)

print("Estimated median recovery duration:",
      kmf.median_survival_time_)

print(kmf.survival_function_)

# %%
import pandas_market_calendars as mcal
import pandas as pd

nasdaq = mcal.get_calendar("NASDAQ")

last_date = pd.Timestamp("2026-11-13")

# Get future trading sessions
schedule = nasdaq.schedule(
    start_date=last_date + pd.Timedelta(days=1),
    end_date=last_date + pd.Timedelta(days=120)
)

# Neutral estimate: 48 trading sessions
neutral_date = schedule.index[47]

print("Neutral recovery date:", neutral_date.date())

# %%
%pip install pandas_market-calendars

# %%
import pandas_market_calendars as mcal
import pandas as pd

nasdaq = mcal.get_calendar("NASDAQ")

last_date = pd.Timestamp("2026-11-13")

# Get future trading sessions
schedule = nasdaq.schedule(
    start_date=last_date + pd.Timedelta(days=1),
    end_date=last_date + pd.Timedelta(days=120)
)

# Neutral estimate: 48 trading sessions
neutral_date = schedule.index[47]

print("Neutral recovery date:", neutral_date.date())


# %%
# Exact recovery target
target_car = 0.001

# Check the latest observed cumulative abnormal performance
starting_car = actual_car.iloc[-1]

print(f"Latest cumulative abnormal performance: {starting_car:.4f}%")
print(f"Required recovery target: {target_car:.4f}%")
print(f"Remaining gap: {target_car - starting_car:.4f} percentage points")

# %%
# First day each simulated path reaches +0.1 bp
first_recovery_days = []

for path in simulated_car:

    hits = np.where(path >= target_car)[0]

    if len(hits) > 0:
        first_recovery_days.append(hits[0] + 1)
    else:
        first_recovery_days.append(np.nan)

first_recovery_days = np.array(first_recovery_days)

# Probability of recovery within the simulation horizon
recovery_probability = np.isfinite(first_recovery_days).mean() * 100

# Recovery timing conditional on simulated recovery
completed = first_recovery_days[np.isfinite(first_recovery_days)]

print(f"Probability of recovery within 59 days: {recovery_probability:.2f}%")

if len(completed) > 0:
    print("Median recovery day, conditional on recovery:",
          np.median(completed))
    print("75th percentile recovery day, conditional on recovery:",
          np.percentile(completed, 75))
else:
    print("No simulated paths achieved the recovery target.")

# %%
import pandas_market_calendars as mcal
import pandas as pd

nasdaq = mcal.get_calendar("NASDAQ")

last_date = actual_car.index[-1]

schedule = nasdaq.schedule(
    start_date=last_date + pd.Timedelta(days=1),
    end_date=last_date + pd.Timedelta(days=180)
)

def recovery_date(trading_days):
    if trading_days <= 0:
        return last_date
    if trading_days > len(schedule):
        return None
    return schedule.index[trading_days - 1].date()

print("Neutral historical date:", recovery_date(48))

if len(completed) > 0:
    neutral_sim_day = int(np.ceil(np.median(completed)))
    pessimistic_sim_day = int(np.ceil(np.percentile(completed, 75)))

    print("Neutral simulated date:", recovery_date(neutral_sim_day))
    print("Pessimistic simulated date:", recovery_date(pessimistic_sim_day))

# %%
print("Recovery probability:", round(recovery_probability, 2), "%")
print("Number of recovering paths:", len(completed))
print("Total simulated paths:", len(simulated_car))
print("Conditional median:", np.median(completed))
print("Conditional 75th percentile:", np.percentile(completed, 75))

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Recovery dates
historical_date = datetime(2027, 1, 26)
simulation_median = datetime(2027, 1, 13)
simulation_75th = datetime(2027, 1, 28)

start_date = datetime(2026, 11, 13)

# Simulation probabilities
recovery_probability = 12.52
no_recovery_probability = 100 - recovery_probability

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 7),
    gridspec_kw={'height_ratios': [2, 1]}
)

# TOP PANEL: Recovery timeline
ax1.axvline(start_date, linestyle='--', color='grey',
            label='Last observed trading date')

ax1.scatter(historical_date, 3, s=110,
            label='Historical analogue')

ax1.scatter(simulation_median, 2, s=110,
            label='Simulation median')

ax1.scatter(simulation_75th, 1, s=110,
            label='Simulation 75th percentile')

ax1.set_yticks([1, 2, 3])
ax1.set_yticklabels([
    'Simulation 75th percentile',
    'Simulation median',
    'Historical analogue'
])

ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

ax1.set_title('NVDA: Indicative Benchmark-Adjusted Recovery Dates')
ax1.set_xlabel('Potential recovery date')
ax1.grid(axis='x', alpha=0.3)
ax1.legend(loc='lower right', fontsize=8)

# BOTTOM PANEL: Simulation probability
ax2.barh(
    ['Simulated paths'],
    [recovery_probability],
    label='Recovered'
)

ax2.barh(
    ['Simulated paths'],
    [no_recovery_probability],
    left=[recovery_probability],
    label='Not recovered within 59 days'
)

ax2.text(
    recovery_probability / 2, 0,
    f'{recovery_probability:.2f}%',
    ha='center', va='center'
)

ax2.text(
    recovery_probability + no_recovery_probability / 2, 0,
    f'{no_recovery_probability:.2f}%',
    ha='center', va='center'
)

ax2.set_xlim(0, 100)
ax2.set_xlabel('Percentage of simulated paths')
ax2.set_title('Bootstrap Simulation: 5,000 Paths')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
           ncol=2, fontsize=8)

plt.tight_layout()
plt.show()

# %%
# TOP PANEL: Recovery timeline

dates = [
    historical_date,
    simulation_median,
    simulation_75th
]

positions = [3, 2, 1]

labels = [
    "Historical analogue",
    "Simulation median",
    "Simulation 75th percentile"
]

colours = ["tab:blue", "tab:orange", "tab:green"]

# Last observed trading date
ax1.axvline(
    start_date,
    linestyle="--",
    color="grey",
    label="Last observed trading date"
)

# Plot and annotate each recovery date
for date, position, label, colour in zip(
    dates, positions, labels, colours
):
    ax1.scatter(date, position, s=110, color=colour, zorder=3)

    ax1.annotate(
        date.strftime("%d %b %Y"),
        (date, position),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

# Labels
ax1.set_yticks(positions)
ax1.set_yticklabels(labels)

# Add horizontal reference lines
ax1.grid(axis="x", alpha=0.3)
ax1.grid(axis="y", linestyle=":", alpha=0.2)

# Give the chart some breathing room
ax1.set_xlim(
    start_date - pd.Timedelta(days=3),
    max(dates) + pd.Timedelta(days=12)
)

ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

ax1.set_title("NVDA: Indicative Benchmark-Adjusted Recovery Dates")
ax1.set_xlabel("Potential recovery date")

ax1.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    fontsize=8
)

# %%

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from datetime import datetime

# ==========================================
# 1. INPUT DATA
# ==========================================

total_paths = 5000
recovering_paths = 626
recovery_probability = 12.52
no_recovery_probability = 87.48

# Conditional simulation recovery dates
simulation_median = datetime(2027, 1, 13)
simulation_75th = datetime(2027, 1, 28)

# Independent historical analogue
historical_date = datetime(2027, 1, 26)

# ==========================================
# 2. CREATE FIGURE
# ==========================================

fig, ax = plt.subplots(figsize=(15, 8))

ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")

fig.suptitle(
    "NVDA: Benchmark-Adjusted Recovery Probability and Indicative Dates",
    fontsize=16,
    fontweight="bold",
    y=0.96
)

# ==========================================
# 3. HELPER FUNCTION FOR BOXES
# ==========================================

def draw_box(x, y, width, height, title, subtitle,
             facecolor, edgecolor="black"):

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.15",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        x + width / 2,
        y + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=10
    )


# ==========================================
# 4. STARTING POINT
# ==========================================

draw_box(
    0.4, 4.2, 2.5, 1.5,
    "5,000 simulated paths",
    "100% of bootstrap outcomes",
    "#DCE6F1"
)

# ==========================================
# 5. BRANCH: RECOVERY / NO RECOVERY
# ==========================================

# Recovery branch
draw_box(
    4.2, 6.3, 3.0, 1.6,
    "Recovery achieved",
    "626 paths | 12.52%",
    "#D9EAD3"
)

# No recovery branch
draw_box(
    4.2, 2.0, 3.0, 1.6,
    "No recovery within horizon",
    "4,374 paths | 87.48%",
    "#F4CCCC"
)

# Arrows from starting point
ax.add_patch(FancyArrowPatch(
    (2.9, 5.1), (4.2, 7.0),
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.8
))

ax.add_patch(FancyArrowPatch(
    (2.9, 4.7), (4.2, 2.8),
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.8
))

# ==========================================
# 6. CONDITIONAL RECOVERY DATE BRANCH
# ==========================================

draw_box(
    9.0, 6.8, 2.6, 1.4,
    "13 January 2027",
    "Conditional median: 40 days",
    "#D9EAD3"
)

draw_box(
    9.0, 4.7, 2.6, 1.4,
    "28 January 2027",
    "Conditional 75th percentile: 50 days",
    "#D9EAD3"
)

# Arrow from recovery branch to date estimates
ax.add_patch(FancyArrowPatch(
    (7.2, 7.1), (9.0, 7.5),
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.8
))

ax.add_patch(FancyArrowPatch(
    (7.2, 6.9), (9.0, 5.4),
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.8
))

# ==========================================
# 7. HISTORICAL ANALOGUE
# ==========================================

draw_box(
    9.0, 1.7, 2.6, 1.4,
    "26 January 2027",
    "Historical analogue: 48 days",
    "#DCE6F1"
)

# Dashed connector: independent historical evidence
ax.add_patch(FancyArrowPatch(
    (7.2, 2.5), (9.0, 2.4),
    arrowstyle="->",
    mutation_scale=16,
    linewidth=1.5,
    linestyle="dashed",
    color="grey"
))

# ==========================================
# 8. EXPLANATORY NOTES
# ==========================================

ax.text(
    12.0, 7.5,
    "SIMULATION DATES",
    fontsize=11,
    fontweight="bold"
)

ax.text(
    12.0, 6.8,
    "Dates describe only the\n626 paths that recovered.\n\nThey are conditional\nrecovery-time estimates.",
    fontsize=10,
    va="top"
)

ax.text(
    12.0, 2.7,
    "HISTORICAL REFERENCE",
    fontsize=11,
    fontweight="bold"
)

ax.text(
    12.0, 2.0,
    "Independent historical\ncomparison, not a\nbootstrap probability.",
    fontsize=10,
    va="top"
)

# ==========================================
# 9. FOOTNOTE
# ==========================================

ax.text(
    7.5, 0.55,
    "Bootstrap horizon: 59 trading days | "
    "87.48% did not recover within this horizon "
    "(not evidence of permanent non-recovery).",
    ha="center",
    fontsize=9,
    style="italic"
)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()


# %%
import pandas as pd

for name, obj in globals().copy().items():
    if isinstance(obj, pd.DataFrame):
        if any("NVDA" in str(col).upper() for col in obj.columns):
            print(f"\nDataFrame name: {name}")
            print("Columns:", obj.columns.tolist())
            print("Last 5 observations:")
            print(obj.tail())

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

# ==========================================
# 1. EXTRACT ACTUAL NVIDIA CLOSING PRICES
# ==========================================

prices = nvda[("Close", "NVDA")].dropna()

event_start = pd.Timestamp("2026-10-29")
last_date = prices.index[-1]

# Pre-event closing price: 28 October
target_price = prices.loc[prices.index < event_start].iloc[-1]

# Last observed closing price: 13 November
starting_price = prices.iloc[-1]

print("Pre-event target price:", round(target_price, 2))
print("Latest observed price:", round(starting_price, 2))
print("Required recovery:", round(
    (target_price / starting_price - 1) * 100, 2
), "%")

# ==========================================
# 2. HISTORICAL PRE-EVENT DAILY RETURNS
# ==========================================

historical_prices = prices.loc[prices.index < event_start]

historical_returns = historical_prices.pct_change().dropna().values

# ==========================================
# 3. SIMULATE 5,000 FUTURE PRICE PATHS
# ==========================================

n_simulations = 5000
forecast_days = 252

np.random.seed(42)

simulated_returns = np.random.choice(
    historical_returns,
    size=(n_simulations, forecast_days),
    replace=True
)

simulated_prices = starting_price * np.cumprod(
    1 + simulated_returns, axis=1
)

# ==========================================
# 4. IDENTIFY FIRST BREAK-EVEN DAY
# ==========================================

first_recovery_days = np.full(n_simulations, np.nan)

for i, path in enumerate(simulated_prices):

    hits = np.where(path >= target_price)[0]

    if len(hits) > 0:
        first_recovery_days[i] = hits[0] + 1

completed = first_recovery_days[
    np.isfinite(first_recovery_days)
]

# ==========================================
# 5. RECOVERY PROBABILITIES
# ==========================================

prob_59 = np.mean(first_recovery_days <= 59) * 100
prob_120 = np.mean(first_recovery_days <= 120) * 100
prob_252 = np.mean(first_recovery_days <= 252) * 100

print("\nRECOVERY PROBABILITIES")
print("Within 59 trading days:", round(prob_59, 2), "%")
print("Within 120 trading days:", round(prob_120, 2), "%")
print("Within 252 trading days:", round(prob_252, 2), "%")

print("\nSIMULATION RESULTS")
print("Recovering paths:", len(completed), "/", n_simulations)

# ==========================================
# 6. CONDITIONAL RECOVERY TIMING
# ==========================================

if len(completed) > 0:

    median_day = int(np.ceil(np.median(completed)))
    percentile_75 = int(np.ceil(np.percentile(completed, 75)))

    print("\nCONDITIONAL RECOVERY ESTIMATES")
    print("Median recovery:", median_day, "trading days")
    print("75th percentile:", percentile_75, "trading days")

    # NASDAQ trading calendar
    nasdaq = mcal.get_calendar("NASDAQ")

    schedule = nasdaq.schedule(
        start_date=last_date + pd.Timedelta(days=1),
        end_date=last_date + pd.Timedelta(days=400)
    )

    def recovery_date(days):
        if days <= len(schedule):
            return schedule.index[days - 1].date()
        return None

    print("Conditional median date:", recovery_date(median_day))
    print("Conditional 75th percentile date:", recovery_date(percentile_75))

else:
    print("No simulated paths recovered within the forecast horizon.")

# ==========================================
# 7. FINAL VISUALISATION
# ==========================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 9),
    gridspec_kw={"height_ratios": [2, 1]}
)

# Plot median and percentile price paths
median_path = np.median(simulated_prices, axis=0)
lower_path = np.percentile(simulated_prices, 10, axis=0)
upper_path = np.percentile(simulated_prices, 90, axis=0)

future_dates = schedule.index[:forecast_days]

ax1.plot(
    future_dates, median_path,
    label="Median simulated price path"
)

ax1.fill_between(
    future_dates, lower_path, upper_path,
    alpha=0.2,
    label="10th–90th percentile range"
)

ax1.axhline(
    target_price,
    linestyle="--",
    color="red",
    label="Pre-event break-even price"
)

ax1.axhline(
    starting_price,
    linestyle=":",
    color="grey",
    label="Starting price"
)

ax1.set_title("NVDA: Simulated Share-Price Recovery")
ax1.set_ylabel("Share price ($)")
ax1.set_xlabel("Forecast date")
ax1.legend()
ax1.grid(alpha=0.3)

# Cumulative probability of first recovery
days = np.arange(1, forecast_days + 1)

cumulative_probability = np.array([
    np.mean(first_recovery_days <= day) * 100
    for day in days
])

ax2.plot(days, cumulative_probability)

ax2.set_title("Cumulative Probability of Share-Price Break-Even")
ax2.set_xlabel("Trading days after 13 November 2026")
ax2.set_ylabel("Probability (%)")
ax2.set_ylim(0, 100)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Probability-based break-even dates

print("PROBABILITY-BASED BREAK-EVEN DATES")

for probability in [25, 50, 75]:

    hits = np.where(
        cumulative_probability >= probability
    )[0]

    if len(hits) > 0:

        day = hits[0] + 1
        date = schedule.index[day - 1].date()

        print(
            f"{probability}% probability: "
            f"{day} trading days | {date}"
        )

    else:

        print(
            f"{probability}% probability: "
            "Not reached within 252 trading days"
        )


