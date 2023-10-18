from torchvision import transforms, datasets
from typing import *
import torch
import os
import clip
import json
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from .classes_info import *
from tqdm import tqdm
# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "~/ILSVRC2012"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    dir = IMAGENET_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=3, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(size=256, interpolation=3, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
    
class TextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def get_model_device(model):
    return next(model.parameters()).device

def get_main_text_weights(dataset, model):
    device = get_model_device(model)
    with torch.no_grad():
        if dataset == 'imagenet':
            zeroshot_weights = []
            for classname in tqdm(imagenet_classes):
                texts = [template.format(classname) for template in imagenet_templates] #format with class
                texts = model.tokenizer(texts).to(device) #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        elif dataset == 'cifar10':
            texts = torch.cat([model.tokenizer(f"a photo of {c}") for c in cifar10_classes]).to(device)
            zeroshot_weights = model.encode_text(texts)
            zeroshot_weights /= zeroshot_weights.norm(dim=-1, keepdim=True)
            zeroshot_weights = zeroshot_weights.t()
    print(f"Loaded the main text embedding for {dataset}.")
    return zeroshot_weights

def get_knowledge_text_weights(text, model, batch_size = 500):
    device = get_model_device(model)
    dataset = TextDataset(text)
    batch_size = batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_text_features = torch.tensor([]).half().to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            tokenized_batch = model.tokenizer(batch, truncate = True).to(device)
            text_features = model.encode_text(tokenized_batch)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            all_text_features = torch.cat((all_text_features, text_features), dim=0)
    print("Loaded the knowledge text embedding.")
    return all_text_features.t()

# def remove_repetitive_elements(knowledge_dict):
#     positive_statements = knowledge_dict['positive_statements']
#     negative_statements = knowledge_dict['negative_statements']
#     relations = knowledge_dict['relations']
#     unique_statements = []
#     unique_negative_statements = []
#     unique_relations = []
#     for i, statement in enumerate(positive_statements):
#         if statement not in unique_statements:
#             unique_statements.append(statement)
#             unique_negative_statements.append(negative_statements[i])
#             unique_relations.append(relations[i])
#     knowledge_dict['positive_statements'] = unique_statements
#     knowledge_dict['negative_statements'] = unique_negative_statements
#     knowledge_dict['relations'] = unique_relations
#     return knowledge_dict

def get_knowledge_sentences(dataset, knowledge_path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Construct the full path to the knowledge file
    full_path = os.path.join(dir_path, 'knowledge', f'{knowledge_path}.json')
    knowledge = json.load(open(full_path))
    knowledge_dict = OrderedDict()
    knowledge_index = 0
    all_negative_statements = []

    for cls in tqdm(knowledge.keys()):
        positive_statements = knowledge[cls]['positive_statements']
#         negative_statements = knowledge[cls]['negative_statements']

        for j in range(len(positive_statements)):
            pos = positive_statements[j]
#             neg = negative_statements[j]

            if pos not in knowledge_dict:
                knowledge_dict[pos] = {'knowledge_index': knowledge_index}
#                 all_negative_statements.append(neg)
                knowledge_index += 1

    return list(knowledge_dict.keys())#, all_negative_statements

def get_knowledge_rules(dataset, knowledge_path, train_main = False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Construct the full path to the knowledge file
    full_path = os.path.join(dir_path, 'knowledge', f'{knowledge_path}.json')
    knowledge = json.load(open(full_path))
    main_num = get_num_classes(dataset)
    gt_matrix = []
    indices = []
    bias = []
    values = []
    edge_index = []
    formula_num = 0
    knowledge_dict = OrderedDict()
    knowledge_index = 0

    for cls in range(main_num):
        gt_matrix.append([cls, cls])
        edge_index.extend([[cls, cls]])
        knowledge_index += 1
        if train_main:
            indices.extend([[cls, formula_num]])
            values.extend([-1])
            bias.append(1)
            formula_num += 1
            
    for cls in tqdm(knowledge.keys()):
        positive_statements = knowledge[cls]['positive_statements']
        relations = knowledge[cls]['relations']

        for j in range(len(positive_statements)):
            pos = positive_statements[j]
            rel = relations[j]

            if pos in knowledge_dict:
                cur_knowledge_index = knowledge_dict[pos]['knowledge_index']
            else:
                knowledge_dict[pos] = {'knowledge_index': knowledge_index}
                cur_knowledge_index = knowledge_index
                edge_index.extend([[cur_knowledge_index, cur_knowledge_index]])
                if train_main:
                    indices.extend([[cur_knowledge_index, formula_num]])
                    values.extend([-1])
                    bias.append(1)
                    formula_num += 1
                knowledge_index += 1
            gt_matrix.append([int(cls), cur_knowledge_index])
            edge_index.extend([[int(cls), cur_knowledge_index], [cur_knowledge_index, int(cls)]])
            if rel == 1:
                indices.extend([[int(cls), formula_num], [cur_knowledge_index, formula_num]])
                values.extend([1, -1])
                bias.append(0)
                formula_num += 1
            elif rel == 2:
                indices.extend([[int(cls), formula_num], [cur_knowledge_index, formula_num]])
                values.extend([-1, 1])
                bias.append(0)
                formula_num += 1
            elif rel == 3:
                indices.extend([[int(cls), formula_num], [cur_knowledge_index, formula_num]])
                values.extend([1, -1])
                bias.append(0)
                indices.extend([[int(cls), formula_num+1], [cur_knowledge_index, formula_num+1]])
                values.extend([-1, 1])
                bias.append(0)
                formula_num += 2
        
    formula = torch.sparse_coo_tensor(torch.tensor(indices).t(), 
                                      torch.tensor(values)).to_dense().float().to_sparse()
    gt_matrix = torch.sparse_coo_tensor(torch.tensor(gt_matrix).t(), 
                                      torch.ones(len(gt_matrix))).to_dense().float()
    bias = torch.tensor(bias).float()
    edge_index = torch.tensor(edge_index).t().long()
    multiclass_list = None
    print("Loaded the knowledge rules.")
    return gt_matrix, formula, bias, edge_index, multiclass_list