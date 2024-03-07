import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import numpy as np

class AttentionFusion(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(AttentionFusion, self).__init__()

        self.linear_input2 = nn.Linear(input_size2, input_size1)

        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size1 * 2, 1)

    def forward(self, input1, input2):
        output2 = F.relu(self.linear_input2(input2))

        fusion_input = torch.cat((input1, output2), dim=1)

        attention_weights = F.softmax(self.linear(fusion_input), dim=1)

        fusion_output = torch.mul(input1, attention_weights) + torch.mul(output2, (1 - attention_weights))

        return fusion_output

def mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return np.mean(np.abs((prediction - actual) / actual)) * 100

def symmetric_mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return np.mean(np.abs((prediction - actual) / actual)) * 100

def cal_metric(data):
    mse = mean_squared_error(data['real'].values, data['pre'].values)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data['real'], data['pre'])
    smape = symmetric_mean_absolute_percentage_error(data['real'], data['pre'])
    mae = mean_absolute_error(data['real'].values, data['pre'].values)
    r2 = r2_score(data['real'].values, data['pre'].values)

    # Generate prediction accuracy
    actual = data['real'].values
    result_1 = []
    result_2 = []
    for i in range(1, len(data['pre'])):
        # Compare prediction to previous close price
        if data['pre'][i] >= actual[i - 1] and actual[i] >= actual[i - 1] and data['pre'][i] >= data['pre'][i - 1]:
            result_1.append(1)
        elif data['pre'][i] <= actual[i - 1] and actual[i] <= actual[i - 1]:
            result_1.append(1)
        else:
            result_1.append(0)

        # Compare prediction to previous prediction
        if data['pre'][i] >= data['pre'][i - 1] and actual[i] >= actual[i - 1]:
            result_2.append(1)
        elif data['pre'][i] <= data['pre'][i - 1] and actual[i] <= actual[i - 1]:
            result_2.append(1)
        else:
            result_2.append(0)

    accuracy_1 = np.mean(result_1)*100
    accuracy_2 = np.mean(result_2)*100

    print('Prediction vs Close:\t\t' + str('{:.2f}'.format(accuracy_1)) + '% Accuracy')
    print(
        'Prediction vs Prediction:\t' + str('{:.2f}'.format(accuracy_2)) + '% Accuracy')
    print('MSE:\t', mse,
          '\nRMSE:\t', rmse,
          '\nMAPE:\t', mape,
          '\nSMAPE:\t', smape,
          '\nMAE:\t', mae,
          '\nR2:\t', r2)

input_size1 = 200
input_size2 = 40
hidden_size = 50

model = AttentionFusion(input_size1, input_size2, hidden_size)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# data
confirmed = pd.read_csv('results_confirmed.csv').T.values
mobility = pd.read_csv('results_pop.csv').T.values
input1 = torch.tensor(confirmed).float()
input2 = torch.tensor(mobility).float()

# train
epochs = 100
for epoch in range(epochs):
    # forward
    output = model(input1, input2)

    # loss
    loss = criterion(output, torch.zeros_like(output))

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

print("Training finished!")

result = pd.DataFrame(output.detach().numpy(),index=['real', 'pre']).T
cal_metric(result)
df = result.to_csv('results_AT.csv', index=False)