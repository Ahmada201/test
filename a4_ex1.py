import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(
            self,
            input_neurons: int,
            hidden_neurons: list,
            output_neurons: int,
            use_bias: bool,
            activation_function: torch.nn.Module = torch.nn.ReLU()
    ):

        super(SimpleNetwork, self).__init__()
        layers = []
        previous_neurons = input_neurons
        for current_neurons in hidden_neurons:
            layers.append(nn.Linear(previous_neurons, current_neurons, bias=use_bias))
            layers.append(activation_function)
            previous_neurons = current_neurons
        layers.append(nn.Linear(previous_neurons, output_neurons, bias=use_bias))


        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    torch.random.manual_seed(1234)
    simple_network = SimpleNetwork(input_neurons=40, hidden_neurons=[10, 20, 30], output_neurons=5, use_bias=True)
    input = torch.randn(1, 40)
    output = simple_network(input)
    print(output)