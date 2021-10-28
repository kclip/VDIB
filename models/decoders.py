import torch
from collections import OrderedDict


class conv(torch.nn.Module):
    def __init__(self, T: int, in_features: int, hid_features: int, out_features: int, bias: bool = True, sigmoid: bool = True, softmax: bool = False) -> None:
        super(conv, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1)

        self.fc = torch.nn.Sequential(OrderedDict([
            ('f6', torch.nn.Linear(8 * in_features // 16, 2048)),
            ('relu6', torch.nn.ReLU()),
            ('f7', torch.nn.Linear(2048, out_features))
        ]))
        self.sigmoid = sigmoid

    def forward(self, z):
        h1 = self.conv1(z)
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2))

        if self.sigmoid:
            return torch.sigmoid(self.fc(h3.view(z.shape[0], -1)))
        else:
            return self.fc(h3.view(z.shape[0], -1))


class mlp(torch.nn.Module):
    def __init__(self, in_features: int, hid_features: int, out_features: int, bias: bool = True, sigmoid: bool = True, softmax: bool = False, T: int = 0) -> None:
        super(mlp, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, hid_features, bias)
        self.fc2 = torch.nn.Linear(hid_features, out_features, bias)

        self.sigmoid = sigmoid
        self.softmax = softmax

    def forward(self, z) -> torch.Tensor:
        h = torch.nn.functional.relu(self.fc1(z))

        if self.sigmoid:
            return torch.sigmoid(self.fc2(h))
        elif self.softmax:
            return torch.log_softmax(self.fc2(h), dim=-1)
        else:
            return self.fc2(h)


class linear(torch.nn.Module):
    def __init__(self, in_features: int, hid_features: int, out_features: int, bias: bool = True, sigmoid: bool = True, softmax: bool = False, T: int = 0) -> None:
        super(linear, self).__init__()

        self.fc = torch.nn.Linear(in_features, out_features, bias)
        self.sigmoid = sigmoid
        self.softmax = softmax

    def forward(self, z) -> torch.Tensor:
        if self.sigmoid:
            return torch.sigmoid(self.fc(z))
        elif self.softmax:
            return torch.log_softmax(self.fc(z), dim=-1)
        else:
            return self.fc(z)

