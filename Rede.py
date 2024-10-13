import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class Sara(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln0 = nn.Linear(2, 32)
        self.ln1 = nn.Linear(32, 64)
        self.ln2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.ln0(x))
        x = self.relu(self.ln1(x))
        x = self.ln2(x)
        return x

class Treinar:
    def __init__(self, lr:float=0.001):
        self.modelo = Sara()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.modelo.parameters(), lr=0.0001)
    
    def train(self, inputs, outputs, plot:bool=False,  epochs : int = 500):
        for epoch in range(epochs):
            predicted = self.modelo(inputs)
            loss = self.criterion(predicted, outputs)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        
            if int((epoch+1) % (epochs/5)) == 0:
                y_p = (predicted >= 0.6).float()
                act = predicted.round()
                acc = act.eq(outputs).sum() / float(outputs.shape[0])

                
                print(f"Epoch: {epoch+1}, Accuracy: {acc}, Loss:{loss.item():.4f}")
        if plot:
                plt.scatter(inputs[:,0], inputs[:,1], c=y_p, cmap='cool', alpha=0.5)
                plt.show()
