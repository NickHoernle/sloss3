import torch
from torch import nn

import numpy as np

superclass_mapping = {
    'beaver': 'aquatic mammals',
    'dolphin': 'aquatic mammals',
    'otter': 'aquatic mammals',
    'seal': 'aquatic mammals',
    'whale': 'aquatic mammals',
    'aquarium_fish': 'fish',
    'flatfish': 'fish',
    'ray': 'fish',
    'shark': 'fish',
    'trout': 'fish',
    'orchid': 'flowers',
    'poppy': 'flowers',
    'rose': 'flowers',
    'sunflower': 'flowers',
    'tulip': 'flowers',
    'bottle': 'food containers',
    'bowl': 'food containers',
    'can': 'food containers',
    'cup': 'food containers',
    'plate': 'food containers',
    'apple': 'fruit and vegetables',
    'mushroom': 'fruit and vegetables',
    'orange': 'fruit and vegetables',
    'pear': 'fruit and vegetables',
    'sweet_pepper': 'fruit and vegetables',
    'clock': 'household electrical devices',
    'keyboard': 'household electrical devices',
    'lamp': 'household electrical devices',
    'telephone': 'household electrical devices',
    'television': 'household electrical devices',
    'bed': 'household furniture',
    'chair': 'household furniture',
    'couch': 'household furniture',
    'table': 'household furniture',
    'wardrobe': 'household furniture',
    'bee':'insects',
    'beetle':'insects',
    'butterfly':'insects',
    'caterpillar':'insects',
    'cockroach':'insects',
    'bear': 'large carnivores',
    'leopard': 'large carnivores',
    'lion': 'large carnivores',
    'tiger': 'large carnivores',
    'wolf': 'large carnivores',
    'bridge': 'large man-made outdoor things',
    'castle': 'large man-made outdoor things',
    'house': 'large man-made outdoor things',
    'road': 'large man-made outdoor things',
    'skyscraper': 'large man-made outdoor things',
    "cloud": "large natural outdoor scenes",
    "forest": "large natural outdoor scenes",
    "mountain": "large natural outdoor scenes",
    "plain": "large natural outdoor scenes",
    "sea": "large natural outdoor scenes",
    "camel": "large omnivores and herbivores",
    "cattle": "large omnivores and herbivores",
    "chimpanzee": "large omnivores and herbivores",
    "elephant": "large omnivores and herbivores",
    "kangaroo": "large omnivores and herbivores",
    "fox": "medium-sized mammals",
    "porcupine": "medium-sized mammals",
    "possum": "medium-sized mammals",
    "raccoon": "medium-sized mammals",
    "skunk": "medium-sized mammals",
    "crab": "non-insect invertebrates",
    "lobster": "non-insect invertebrates",
    "snail": "non-insect invertebrates",
    "spider": "non-insect invertebrates",
    "worm": "non-insect invertebrates",
    "baby": "people",
    "boy": "people",
    "girl": "people",
    "man": "people",
    "woman": "people",
    "crocodile" : "reptiles",
    "dinosaur" : "reptiles",
    "lizard" : "reptiles",
    "snake" : "reptiles",
    "turtle": "reptiles",
    "hamster": "small mammals",
    "mouse": "small mammals",
    "rabbit": "small mammals",
    "shrew": "small mammals",
    "squirrel": "small mammals",
    "maple_tree" :"trees",
    "oak_tree" :"trees",
    "palm_tree" :"trees",
    "pine_tree" :"trees",
    "willow_tree" :"trees",
    "bicycle": "vehicles 1",
    "bus": "vehicles 1",
    "motorcycle": "vehicles 1",
    "pickup_truck": "vehicles 1",
    "train": "vehicles 1",
    "lawn_mower": "vehicles 2",
    "rocket": "vehicles 2",
    "streetcar": "vehicles 2",
    "tank": "vehicles 2",
    "tractor": "vehicles 2"
}

super_class_label = {
    'aquatic mammals': 0,
    'fish': 1,
    'flowers': 2,
    'food containers': 3,
    'fruit and vegetables': 4,
    'household electrical devices': 5,
    'household furniture': 6,
    'insects': 7,
    'large carnivores': 8,
    'large man-made outdoor things': 9,
    'large natural outdoor scenes': 10,
    'large omnivores and herbivores': 11,
    'medium-sized mammals': 12,
    'non-insect invertebrates': 13,
    'people': 14,
    'reptiles': 15,
    'small mammals': 16,
    'trees': 17,
    'vehicles 1': 18,
    'vehicles 2': 19
}

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

    def logic_statement(target, within_group_ix, outside_group_ix, epsilon=5):
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


# def create_cifar10_group_precision(animate_ix, inaminate_ix):
#
#     def logic_statement(target, within_group_ix, outside_group_ix, epsilon=5):
#         # f"(predictions[:, {target}].unsqueeze(1) >= predictions).all(dim=1) & " + \
#         return f"(target=={target}) & " + \
#                "&".join(
#                    [f"(predictions[:, {within_group_ix}] > (predictions[:, {i}].unsqueeze(1) + {epsilon})).all(dim=1)"
#                     for i in outside_group_ix])
#
#     statement = []
#     for a in animate_ix:
#         statement.append(logic_statement(target=a, within_group_ix=animate_ix, outside_group_ix=inaminate_ix))
#
#     for ia in inaminate_ix:
#         statement.append(logic_statement(target=ia, within_group_ix=inaminate_ix, outside_group_ix=animate_ix))
#
#     statement = " | ".join(statement)
#     return lambda target, predictions: eval(statement)


def get_cifar10_experiment_params(dataset):

    classes = dataset.classes

    inanimate = ['airplane', 'automobile', 'ship', 'truck']
    animate = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']

    animate_ix = [i for i,l in enumerate(classes) if l in animate]
    inanimate_ix = [i for i, l in enumerate(classes) if l in inanimate]

    examples = torch.ones(10, 10)
    examples *= -10

    for a in animate_ix:
        examples[a, animate_ix] = 1

    for ia in inanimate_ix:
        examples[ia, inanimate_ix] = 1

    examples[torch.arange(10), torch.arange(10)] = 1

    return examples, create_cifar10_logic(animate_ix, inanimate_ix), create_cifar10_logic(animate_ix, inanimate_ix)


def build_logic(target, predictions, tgt, within_group_ix, outside_group_ix, epsilon=1):
    return torch.stack([target == tgt] + [((predictions[:, outside_group_ix] + epsilon) < predictions[:, i].unsqueeze(1)).all(dim=1) for i in within_group_ix], dim=1).all(dim=1)


def create_cifar100_logic(group_ixs):

    def logic(target, predictions):

        group_ix = np.array(group_ixs)
        statement = []
        for i, group in enumerate(group_ixs):
            for ix in group:
                statement.append(build_logic(target=target, predictions=predictions, tgt=ix, within_group_ix=group, outside_group_ix=group_ix[np.arange(len(group_ix)) != i].reshape(-1,)))

        return torch.stack(statement, dim=1).any(dim=1)

    return lambda target, predictions: logic(target, predictions)


def get_cifar100_experiment_params(dataset):
    classes = dataset.classes

    super_class_ix = []
    for sc_l in super_class_label.keys():
        super_class_ix.append([i for i, l in enumerate(classes) if superclass_mapping[l] == sc_l])

    examples = torch.ones(100, 100)
    examples *= -10

    for group in super_class_ix:
        for ix in group:
            examples[ix, group] = 1

    examples[torch.arange(100), torch.arange(100)] = 1

    return examples, create_cifar100_logic(super_class_ix), create_cifar100_logic(super_class_ix)


def calc_logic_loss(predictions, targets, logic_net, logic_func, num_classes=10, device="cpu"):
    one_hot = idx_to_one_hot(targets, num_classes, device)
    true_logic = logic_func(targets, predictions)
    predicted_logic = logic_net(torch.cat((predictions, one_hot), dim=1)).squeeze(1)
    return predicted_logic, true_logic


if __name__ == "__main__":
    animate = [0,1,2,3,4]
    inanimate = [5,6,7,8,9]
    test = create_cifar10_logic(animate, inanimate)
