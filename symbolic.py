import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def idx_to_one_hot(labels, num_classes, device):
    y_onehot = torch.FloatTensor(len(labels), num_classes).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.unsqueeze(1), 1)
    return y_onehot


class LogicNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes*2, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)


def create_cifar10_logic(animate_ix, inaminate_ix):

    def logic_statement(target, within_group_ix, outside_group_ix, epsilon=2):
        # f"(predictions[:, {target}].unsqueeze(1) >= predictions).all(dim=1) & " + \
        return f"(target=={target}) & " + \
               "&".join([f"(predictions[:, {within_group_ix}] > (predictions[:, {i}].unsqueeze(1) + {epsilon})).all(dim=1)" for i in outside_group_ix])

    statement = []
    for a in animate_ix:
        statement.append(logic_statement(target=a, within_group_ix=animate_ix, outside_group_ix=inaminate_ix))

    for ia in inaminate_ix:
        statement.append(logic_statement(target=ia, within_group_ix=inaminate_ix, outside_group_ix=animate_ix))

    statement = " | ".join(statement)
    return lambda target, predictions: eval(statement)


def create_cifar10_group_precision(animate_ix, inaminate_ix):

    def logic_statement(target, within_group_ix, outside_group_ix, epsilon=2):
        # f"(predictions[:, {target}].unsqueeze(1) >= predictions).all(dim=1) & " + \
        return f"(target=={target}) & " + \
               "&".join(
                   [f"(predictions[:, {within_group_ix}] > (predictions[:, {i}].unsqueeze(1) + {epsilon})).all(dim=1)"
                    for i in outside_group_ix])

    statement = []
    for a in animate_ix:
        statement.append(logic_statement(target=a, within_group_ix=animate_ix, outside_group_ix=inaminate_ix))

    for ia in inaminate_ix:
        statement.append(logic_statement(target=ia, within_group_ix=inaminate_ix, outside_group_ix=animate_ix))

    statement = " | ".join(statement)
    return lambda target, predictions: eval(statement)


def get_cifar10_experiment_params(dataset):

    classes = dataset.classes

    inanimate = ['airplane', 'automobile', 'ship', 'truck']
    animate = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']

    animate_ix = [i for i,l in enumerate(classes) if l in animate]
    inanimate_ix = [i for i, l in enumerate(classes) if l in inanimate]

    examples = torch.ones(10, 10)
    examples *= -6

    for a in animate_ix:
        examples[a, animate_ix] = -1

    for ia in inanimate_ix:
        examples[ia, inanimate_ix] = -1

    examples[torch.arange(10), torch.arange(10)] = 1

    return examples, create_cifar10_logic(animate_ix, inanimate_ix), create_cifar10_group_precision(animate_ix, inanimate_ix)


def calc_logic_loss(predictions, targets, logic_net, logic_func, device):
    one_hot = idx_to_one_hot(targets, targets.max()+1, device)
    true_logic = logic_func(targets, predictions)
    predicted_logic = logic_net(torch.cat((predictions, one_hot), dim=1)).squeeze(1)
    return predicted_logic, true_logic


if __name__ == "__main__":
    animate = [0,1,2,3,4]
    inanimate = [5,6,7,8,9]
    test = create_cifar10_logic(animate, inanimate)
