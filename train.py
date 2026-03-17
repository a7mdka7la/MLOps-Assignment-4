import torch
import torch.nn as nn
import os

# Define a simple linear model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def main():
    print("Initializing training...")
    model = SimpleLinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy data for y = 2x
    x_train = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    y_train = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)

    # Simple training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/simple_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")

if __name__ == '__main__':
    main()
