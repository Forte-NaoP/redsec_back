import torch

model_file = 'your_model_file.pth'
# data_size must be larger than maximum out_feature of your model fc layers
data_size = 1080


# must use this function as activation function of your model
def apporximate_relu(x):
    x = 0.117071 * x ** 2 + 0.5 * x + 0.375373
    return x


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