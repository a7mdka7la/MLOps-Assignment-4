import torch
import torch.nn as nn
import os


class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def predict(input_value):
    model_path = 'models/simple_model.pth'
    if not os.path.exists(model_path):
        print(
            f"Error: Model not found at {model_path}. "
            "Please run train.py first."
        )
        return

    model = SimpleLinearRegression()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        test_input = torch.tensor([[input_value]], dtype=torch.float32)
        prediction = model(test_input)
        print(f"Input: {input_value}, Prediction: {prediction.item():.4f}")


if __name__ == '__main__':
    print("Running inference...")
    predict(5.0)
