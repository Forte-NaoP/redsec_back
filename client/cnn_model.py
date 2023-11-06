import torch

model_file = 'your_model_file.pth'

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # define layers

    def forward(self, x):
        # define forward pass
        return x
    

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNN().to(device)

    # write your code here

    # torch.save(model.state_dict(),model_file)
    pass