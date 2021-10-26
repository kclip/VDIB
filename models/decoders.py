import torch


class conv(torch.nn.Module):
    def __init__(self, T: int, in_features: int, hid_features: int, out_features: int, bias: bool = True, sigmoid: bool = True, softmax: bool = False) -> None:
        super(conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(T, T//2, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(T//2, 20, 3, stride=2, padding=1)
        self.fc = torch.nn.Linear(20 * in_features // T // 4, out_features, bias)

        self.sigmoid = sigmoid
        self.softmax = softmax

    def forward(self, z) -> torch.Tensor:
        h1 = self.conv1(z)
        h2 = torch.relu(self.conv2(h1))

        if self.sigmoid:
            return torch.sigmoid(self.fc(h2.view(1, -1)))
        elif self.softmax:
            return torch.log_softmax(self.fc(h2.view(1, -1)), dim=-1)


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

