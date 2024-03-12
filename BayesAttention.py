import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

class Attention0(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        # Calculate attention weights
        hidden = torch.tanh(self.W(inputs))
        scores = self.V(hidden)
        attention_weights = F.softmax(scores, dim=1)
        # Return attention weights
        return attention_weights

class Attention(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(Attention, self).__init__()

        self.linear_input2 = nn.Linear(input_size2, input_size1)

        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size1 * 2, 1)

    def forward(self, input1, input2):
        output2 = torch.tanh(self.linear_input2(input2))

        fusion_input = torch.cat((input1, output2), dim=1)

        attention_weights = F.softmax(self.linear(fusion_input), dim=1)

        # fusion_output = torch.mul(input1, attention_weights) + torch.mul(output2, (1 - attention_weights))

        return attention_weights

class BayesianNetworkWithAttention(nn.Module):
    def __init__(self):
        super(BayesianNetworkWithAttention, self).__init__()
        self.attention = Attention(input_size1, input_size2, hidden_size=50)
        self.linear_input2 = nn.Linear(input_size2, input_size1)

    def forward(self, input1, input2):
        output2 = torch.tanh(self.linear_input2(input2))
        # Calculate attention weights
        attention_weights = self.attention(input1, input2)
        # Fuse the outputs
        fusion_output = torch.mul(input1, attention_weights) + torch.mul(output2, (1 - attention_weights))
        return fusion_output, attention_weights

input_size1=200
input_size2=40
hidden_size=50

if __name__ == "__main__":
    # Load data
    confirmed = pd.read_csv('results_confirmed.csv').T.values
    mobility = pd.read_csv('results_pop.csv').T.values
    input1 = torch.tensor(confirmed).float()
    input2 = torch.tensor(mobility).float()

    # Initialize model
    model = BayesianNetworkWithAttention()

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        output, weights = model(input1, input2)

        # Calculate loss
        loss = criterion(output, torch.zeros_like(output))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    print("Training finished!")

    result = pd.DataFrame(output.detach().numpy(), index=['real', 'pre']).T
    # Print results
    print("Fused Output:")
    print(result)
    print("Attention Weights:")
    print(weights)

    # df = result.to_csv('Fused_Data.csv', index=False)