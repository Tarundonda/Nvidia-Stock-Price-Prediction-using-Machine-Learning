# Load data and run prediction
df = pd.read_csv("Nvidia DATA.csv")
forecast_col = 'Close'
forecast_out = 5
test_size = 0.2

X_train, X_test, Y_train, Y_test , X_lately = prepare_data(df, forecast_col, forecast_out, test_size)
learner = LinearRegression()
learner.fit(X_train, Y_train)
score = learner.score(X_test, Y_test)
forecast = learner.predict(X_lately)

print("Accuracy:", score)
print("Forecast:", forecast)
