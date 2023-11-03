import torch

model_file = './M3_model_relu1024.pth'


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.FC1 = torch.nn.Linear(1014, 1024)
        self.FC2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.AvgPool1(x)
        # x = 0.117071 * x ** 2 + 0.5 * x + 0.375373
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = 0.117071 * x ** 2 + 0.5 * x + 0.375373
        x = self.FC2(x)
        return x

