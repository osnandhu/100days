# Inflation Agent - Deep Dive

## What Does This Agent Do?

The inflation agent answers: **"What will prices look like in the next 6 months?"**

It fetches real economic data from the World Bank, fits a forecasting model,
and produces predictions with confidence intervals.

---

## Core Concepts

### What is CPI (Consumer Price Index)?

CPI measures how much prices change over time. If CPI inflation is 3.5%,
that means a basket of everyday goods that cost $100 last year now costs $103.50.

Singapore's MAS (Monetary Authority) tracks CPI monthly. The World Bank
aggregates this into annual figures that we use here.

### What is Time Series Forecasting?

Given data points ordered by time (e.g., inflation rates from 2015-2024),
predict what comes next. The key assumption: **patterns in the past will
continue into the future** (at least for the near term).

### What is Prophet?

Facebook (Meta) built Prophet for business forecasting. It decomposes a
time series into three components:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + error(t)
```

- **Trend**: Is the overall direction up, down, or flat?
- **Seasonality**: Does the pattern repeat annually? (Inflation often spikes
  during certain months due to energy costs, harvest cycles, etc.)
- **Error**: Random noise that no model can predict.

Prophet fits these components automatically — you don't need to specify
ARIMA parameters like (p, d, q) which require statistical expertise.

---

## How the Code Works (Step by Step)

### Step 1: Fetch Data (WorldBankCPITool)

```python
url = "https://api.worldbank.org/v2/country/SGP/indicator/FP.CPI.TOTL.ZG"
params = {"date": "2015:2024", "format": "json"}
response = requests.get(url, params=params)
```

- `SGP` = Singapore's ISO code
- `FP.CPI.TOTL.ZG` = "Consumer prices, annual % change"
- Returns JSON: `[metadata, [{date: "2023", value: 4.82}, ...]]`

The tool cleans null values and sorts by year.

### Step 2: Prepare for Prophet (ProphetForecastTool)

Prophet needs a DataFrame with columns `ds` (datestamp) and `y` (value).
Our World Bank data is annual, but Prophet works best with more frequent
data. So we expand each year into 12 monthly data points:

```python
for year in data:
    for month in range(1, 13):
        dates.append(Timestamp(year, month, 1))
        values.append(year_cpi_value + small_noise)
```

The noise prevents Prophet from seeing perfectly flat 12-month segments.

### Step 3: Fit and Forecast

```python
model = Prophet(yearly_seasonality=True, interval_width=0.95)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=6, freq="MS")
forecast = model.predict(future)
```

- `interval_width=0.95` → 95% confidence interval (meaning: "we're 95%
  sure the actual value falls between lower_bound and upper_bound")
- `freq="MS"` → month-start frequency

### Step 4: Return Predictions

The output includes for each month:
- `predicted_inflation`: best estimate
- `lower_bound`: pessimistic scenario
- `upper_bound`: optimistic scenario

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| World Bank API (no auth) | Free, reliable, no API key management |
| Annual → monthly interpolation | Prophet needs regular data; trade-off is reduced precision |
| 95% confidence interval | Standard in economics; 99% would be too wide to be useful |
| `allow_delegation=False` on agent | This agent should do its own work, not hand off to others |

---

## Common Pitfalls

1. **World Bank API returns `null` for recent years** — Data lags 1-2 years.
   Don't set `end_year` to the current year without null handling.

2. **Prophet warnings about "less than 2 years of data"** — Need at least
   2 full seasonal cycles. Our 10-year range (2015-2024) is sufficient.

3. **Interpolation overstates precision** — Our monthly predictions are
   really annual trends spread across months. Don't treat individual
   monthly differences as meaningful.

---

## Practice Exercises (Weekend)

### Exercise 1: Change the Country
Modify `WorldBankCPITool` to accept a country code parameter. Try fetching
data for Malaysia (MYS), Indonesia (IDN), or the USA (USA). Compare the
inflation trajectories.

### Exercise 2: Try Different Forecast Periods
Change `forecast_months` from 6 to 12 or 24. Notice how confidence
intervals widen as you forecast further ahead — this is uncertainty growing.

### Exercise 3: Compare with ARIMA
Install `statsforecast` and implement ARIMA as an alternative tool.
Compare its predictions with Prophet's. Which has tighter confidence
intervals? Which is easier to use?

### Exercise 4: Add a Plotting Tool
Create a new CrewAI tool that uses matplotlib to save a forecast chart
as a PNG. The agent could then reference the chart in its output.
