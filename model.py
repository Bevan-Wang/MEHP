import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from pmdarima import auto_arima
import pmdarima as pm
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import json
from sklearn.preprocessing import MinMaxScaler


def mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return 2.0 * np.mean(np.abs(prediction - actual) / (np.abs(prediction) + np.abs(actual))) * 100

def symmetric_mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return np.mean(np.abs((prediction - actual) / actual)) * 100

def get_arima(data, train_len, test_len):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    train = data.head(train_len).values.tolist()
    test = data.tail(test_len).values.tolist()

    # Initialize model
    model = auto_arima(train, max_p=5, max_q=5, seasonal=False, trace=True,
                       error_action='ignore', suppress_warnings=True, maxiter=10)

    # Determine model parameters
    model.fit(train)
    order = model.get_params()['order']
    print('ARIMA order:', order, '\n')

    # Genereate predictions
    prediction = []
    for i in range(len(test)):
        model = pm.ARIMA(order=order)
        model.fit(train)
        print('working on', i + 1, 'of', test_len, '-- ' + str(int(100 * (i + 1) / test_len)) + '% complete')
        prediction.append(model.predict()[0])
        train.append(test[i])

    # Generate error data
    mse = mean_squared_error(test, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(pd.Series(test), pd.Series(test))
    smape = symmetric_mean_absolute_percentage_error(pd.Series(test), pd.Series(prediction))
    mae = mean_absolute_error(test, prediction)
    r2 = r2_score(test,prediction)
    return prediction, mse, rmse, mape, smape, mae, r2


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])
        return out


def get_lstm(data, train_len, test_len, lstm_len=3):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    dataset = np.reshape(data.values, (len(data), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    x_train = []
    y_train = []
    x_test = []

    for i in range(lstm_len, train_len):
        x_train.append(dataset_scaled[i - lstm_len:i, 0])
        y_train.append(dataset_scaled[i, 0])
    for i in range(train_len, len(dataset_scaled)):
        x_test.append(dataset_scaled[i - lstm_len:i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()

    # Create the PyTorch model
    model = LSTMModel(input_dim=1, hidden_dim=lstm_len)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    total_loss = 0
    # train
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_train)

        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print training progress
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}')

    # Calculate and print average loss
    average_loss = total_loss / 500
    print(f'Average Loss: {average_loss:.4f}')
    # Prediction
    model.eval()
    predict = model(x_test)
    predict = predict.data.numpy()
    prediction = scaler.inverse_transform(predict).tolist()

    output = []
    for i in range(len(prediction)):
        output.extend(prediction[i])
    prediction = output

    # Error calculation
    mse = mean_squared_error(data.tail(len(prediction)).values, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data.tail(len(prediction)).reset_index(drop=True), pd.Series(prediction))
    smape = symmetric_mean_absolute_percentage_error(data.tail(len(prediction)).reset_index(drop=True), pd.Series(prediction))
    mae = mean_absolute_error(data.tail(len(prediction)).values, prediction)
    r2 = r2_score(data.tail(len(prediction)).values, prediction)
    print("MSE:" + str(mse))
    print("RMSE:" + str(rmse))
    print("MAPE:" + str(mape))
    print("SMAPE:" + str(smape))
    print("MAE:" + str(mae))
    print("r2:"+ str(r2))

    return prediction, mse, rmse, mape, smape, mae, r2


def SMA(data, window):
    sma = np.convolve(data[target], np.ones(window), 'same') / window
    return sma


def EMA(data, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[0] = data.iloc[0]  # 设置初始值为序列的第一个值

    for i in range(1, len(data)):
        ema[i] = alpha * data.iloc[i] + (1 - alpha) * ema[i - 1]

    return ema


def WMA(data, window):
    weights = np.arange(1, window + 1)
    wma = np.convolve(data[target], weights / weights.sum(), 'same')
    return wma

def getLA(data, target):
    talib_moving_averages = ['WMA']

    functions = {
        # 'SMA': SMA,
        # 'EMA': EMA,
        'WMA': WMA,
    }

    # for ma in talib_moving_averages:
    #     functions[ma] = abstract.Function(ma)

    # Determine kurtosis "K" values for MA period 4-99
    kurtosis_results = {'period': []}
    for i in range(4, 100):
        kurtosis_results['period'].append(i)
        for ma in talib_moving_averages:
            # Run moving average, remove last 200 days (used later for test data set), trim MA result to last 60 days
            ma_output = functions[ma](data[:-200], i)[-50:]
            # Determine kurtosis "K" value
            k = kurtosis(ma_output, fisher=False)

            # add to dictionary
            if ma not in kurtosis_results.keys():
                kurtosis_results[ma] = []
            kurtosis_results[ma].append(k)

    kurtosis_results = pd.DataFrame(kurtosis_results)
    kurtosis_results.to_csv('kurtosis_results.csv')

    # Determine period with K closest to 3 +/-5%
    optimized_period = {}
    for ma in talib_moving_averages:
        difference = np.abs(kurtosis_results[ma] - 3)
        df = pd.DataFrame({'difference': difference, 'period': kurtosis_results['period']})
        df = df.sort_values(by=['difference'], ascending=True).reset_index(drop=True)
        if df.at[0, 'difference'] < 3 * 0.05:
            optimized_period[ma] = df.at[0, 'period']
        else:
            print(ma + ' is not viable, best K greater or less than 3 +/-5%')

    print('\nOptimized periods:', optimized_period)

    simulation = {}
    for ma in optimized_period:
        # Split data into low volatility and high volatility time series
        low_vol = pd.Series(functions[ma](data, optimized_period[ma]))
        high_vol = pd.Series(data[target] - low_vol)

        # Generate ARIMA and LSTM predictions
        print('\nWorking on ' + ma + ' predictions')
        try:
            low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape, low_vol_smape, low_vol_mae, low_vol_r2 = get_arima(low_vol, 800, 200)
        except:
            print('ARIMA error, skipping to next MA type')
            continue

        high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape, high_vol_smape, high_vol_mae, high_vol_r2 = get_lstm(high_vol, 800, 200)

        final_prediction = (pd.Series(low_vol_prediction) + pd.Series(high_vol_prediction))

        mse = mean_squared_error(data[target].tail(200).values, final_prediction.values)
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(data[target].tail(200).reset_index(drop=True), final_prediction)
        smape = symmetric_mean_absolute_percentage_error(data[target].tail(200).reset_index(drop=True), final_prediction)
        mae = mean_absolute_error(data[target].tail(200).values, final_prediction.values)
        r2 = r2_score(data[target].tail(200).values, final_prediction.values)

        # Generate prediction accuracy
        actual = data[target].tail(200).values
        df = pd.DataFrame({'real': actual, 'pre': final_prediction}).to_csv('results_AL.csv', index=False)

        result_1 = []
        result_2 = []
        for i in range(1, len(final_prediction)):
            # Compare prediction to previous close price
            if final_prediction[i] > actual[i - 1] and actual[i] > actual[i - 1]:
                result_1.append(1)
            elif final_prediction[i] < actual[i - 1] and actual[i] < actual[i - 1]:
                result_1.append(1)
            else:
                result_1.append(0)

            # Compare prediction to previous prediction
            if final_prediction[i] > final_prediction[i - 1] and actual[i] > actual[i - 1]:
                result_2.append(1)
            elif final_prediction[i] < final_prediction[i - 1] and actual[i] < actual[i - 1]:
                result_2.append(1)
            else:
                result_2.append(0)

        accuracy_1 = np.mean(result_1)
        accuracy_2 = np.mean(result_2)

        simulation[ma] = {'low_vol': {'prediction': low_vol_prediction, 'mse': low_vol_mse,
                                      'rmse': low_vol_rmse, 'mape': low_vol_mape, 'smape': low_vol_smape,
                                      'mae': low_vol_mae, 'r2': low_vol_r2},
                          'high_vol': {'prediction': high_vol_prediction, 'mse': high_vol_mse,
                                       'rmse': high_vol_rmse, 'mape': high_vol_mape, 'smape': high_vol_smape,
                                       'mae': high_vol_mae, 'r2': high_vol_r2},
                          'final': {'prediction': final_prediction.values.tolist(), 'mse': mse,
                                    'rmse': rmse, 'mape': mape, 'smape': smape, 'mae': mae, 'r2': r2},
                          'accuracy': {'prediction vs close': accuracy_1, 'prediction vs prediction': accuracy_2}}

        # save simulation data here as checkpoint
        with open('simulation_data.json', 'w') as fp:
            json.dump(simulation, fp)

    for ma in simulation.keys():
        print('\n' + ma)
        print('Prediction vs Close:\t\t' + str(round(100 * simulation[ma]['accuracy']['prediction vs close'], 2))
              + '% Accuracy')
        print(
            'Prediction vs Prediction:\t' + str(round(100 * simulation[ma]['accuracy']['prediction vs prediction'], 2))
            + '% Accuracy')
        print('MSE:\t', simulation[ma]['final']['mse'],
              '\nRMSE:\t', simulation[ma]['final']['rmse'],
              '\nMAPE:\t', simulation[ma]['final']['mape'],
              '\nSMAPE:\t', simulation[ma]['final']['smape'],
              '\nMAE:\t', simulation[ma]['final']['mae'],
              '\nR2:\t', simulation[ma]['final']['r2'])



if __name__ == '__main__':
    # Load historical data
    # CSV should have columns: ['Date', 'Confirmed']
    data = pd.read_csv('Fused_Date.csv', index_col=0, header=0).tail(1000).reset_index(drop=True)['Confirmed']
    time = pd.read_csv('Fused_Date.csv', index_col=0, header=0).tail(1000).reset_index(drop=True)['Date']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(data.to_numpy().reshape(-1, 1)).reshape(-1)
    data = pd.DataFrame({'Date': time, 'Confirmed': amplitude}).reset_index(drop=True)
    getLA(data, target)