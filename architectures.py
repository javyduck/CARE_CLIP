import torch
import torch.nn as nn
import torch.nn.init as init
import clip
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from .archs.cifar_resnet import resnet as resnet_cifar
from .datasets import get_normalize_layer, get_main_text_weights, get_knowledge_text_weights, get_knowledge_sentences, get_knowledge_rules, get_num_classes
from torch.nn.functional import interpolate
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, ChebConv
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torchvision.transforms import Compose, Resize, Normalize
from torch_sparse import SparseTensor
from .guided_diffusion import imagenet_gdm
from .improved_diffusion import cifar_ddpm
import numpy as np
# from transformers import AutoModelForImageClassification
import random
import pickle
torch.set_float32_matmul_precision('high')
EPSILON = 1e-6

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", "ViT-L/14", 'RN50', 'RN101', 'RN50x4', 'RN50x16',\
 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

def get_architecture(arch: str, dataset: str, CLIP = False) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif CLIP:
        return robust_clip(arch, dataset)
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)

class robust_clip(torch.nn.Module):
    def __init__(self, arch, dataset, reasoning = False, knowledge_path = None, noise_sd = 0., denoising = False, denoising_ckpt = None, gcn_model = None, use_classifier = False, device = 'cuda'):
        super(robust_clip, self).__init__()
        self.device = device
        self.main_num = get_num_classes(dataset)
        self.clip, _ = clip.load(arch)
        self.clip.tokenizer = clip.tokenize 
        self.clip = self.clip.to(device)
        self.clip.eval()
        resize_transform = Resize(size=224, interpolation=3, antialias=True)
        normalize_transform = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.preprocess = Compose([resize_transform, normalize_transform])
        self.noise_sd = noise_sd
        self.arch = arch
        self.dataset = dataset
        self.use_classifier = use_classifier
#         if use_classifier:
#             if dataset == 'cifar10':
#                 self.classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
#                 self.classifier.eval().to(self.device)
#         else:
        #always use clip main
        self.main_text_weight = get_main_text_weights(dataset, self.clip)
            
        self.reasoning = reasoning
        if reasoning:
            positive_statements = get_knowledge_sentences(dataset, knowledge_path)
            negative_statements = ['a photo of '] * len(positive_statements)
            self.knowledge_text_weight = get_knowledge_text_weights(positive_statements + negative_statements, self.clip)
        else:
            self.knowledge_text_weight = None
        self.noise_sd = noise_sd
        if noise_sd == 0.:
            self.denoising = False
        else:
            self.denoising = denoising
            if denoising:
                if dataset == 'cifar10':
                    self.denoiser = cifar_ddpm(noise_sd, denoising_ckpt, device = device)
                elif dataset == 'imagenet':
                    self.denoiser = imagenet_gdm(noise_sd, denoising_ckpt, device = device)
                self.denoiser.eval()
        self.gcn_model = gcn_model
        if self.gcn_model != None:
            self.gcn_model.eval()
        
    def forward(self, images, only_main = True, return_clip_output = False, return_purified_image = False):
        ## the input should be tensor with range [0, 1]
        bn_size = images.shape[0]
        if self.denoising:
            images = self.denoiser(images)
            if return_purified_image:
                return images
            
        if self.use_classifier:
            if self.dataset == 'cifar10':
                new_images = interpolate((images-0.5)*2, (224, 224), mode='bicubic', antialias=True)
                logits_per_image_main = self.classifier(new_images).logits

        if not self.use_classifier or self.reasoning:
            images = self.preprocess(images)
            image_features = self.clip.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            logit_scale = self.clip.logit_scale.exp()
            if not self.use_classifier:
                logits_per_image_main = logit_scale * image_features @ self.main_text_weight
                
        confidence = logits_per_image_main.softmax(-1)
        if self.reasoning:
            logits_per_image_knowledge = logit_scale * image_features @ self.knowledge_text_weight
            confidence = logits_per_image_main.softmax(-1)
            knowledge_confidence = logits_per_image_knowledge.view(bn_size, 2, -1).softmax(1)[:, 0]
            confidence = torch.cat((confidence, knowledge_confidence), dim=1)
            
            if return_clip_output:
                return confidence
            
            if self.gcn_model != None:
                confidence = self.gcn_model(confidence)
                
        if only_main:
            confidence = confidence[:, :self.main_num]
        return confidence
        
def get_gcn(dataset: str, knowledge_path: str, eta: float, sample_num=10, embedding_dim=64, hidden_dim=64, train_main = False, attention = False, mode = 'sample', device = 'cuda') -> torch.nn.Module:
    """ Return a neural network (with random weights)
    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    gt_matrix, formula, bias, edge_index, multiclass_list = get_knowledge_rules(dataset, knowledge_path, train_main)
    gcn_model = MLN_GCN(dataset, formula, bias, gt_matrix, edge_index, sample_num=sample_num, embedding_dim=embedding_dim, hidden_dim=hidden_dim, eta = eta, train_main = train_main, attention = attention, multiclass_list = multiclass_list, mode = mode).to(device)
    return gcn_model

class MLN_GCN(torch.nn.Module):
    def __init__(self, dataset, formula, bias, gt_matrix, edge_index, sample_num=10, embedding_dim=64, hidden_dim=64, eta=0.5, train_main = False, attention = False, multiclass_list=None, mode = 'sample'):
        super(MLN_GCN, self).__init__()
        # number of predicates
        self.num = formula.shape[0]
        self.dataset = dataset
        self.main_num = get_num_classes(dataset)
        if multiclass_list == None:
            self.multiclass_list = [[0, self.main_num]]
        else:
            self.multiclass_list = multiclass_list
        # with only knowledge formulas
        self.formula = formula
        self.bias = bias
        self.gt_matrix = gt_matrix
        self.edge_index = edge_index
        if not formula.is_sparse:
            sparse_formula = formula.to_sparse()
        else:
            sparse_formula = formula
            
        self.predicate_indices = sparse_formula.indices()[0]
        self.formula_indices = sparse_formula.indices()[1]
        self.values = sparse_formula.values()
        self.mode = mode
        self.train_main = train_main
        self.attention = attention
        
        self.sample_num = sample_num
        self.eta = eta
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(self.embedding_dim, self.hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=False)
        self.conv3 = GCNConv(self.hidden_dim, 1, add_self_loops=False)
        self.w = nn.Parameter(torch.zeros(formula.shape[1]).unsqueeze(0))
        # embedding for the predicate
        self.embedding = nn.Parameter(torch.empty(self.num, self.embedding_dim).uniform_(-1, 1)) 
        init.kaiming_uniform_(self.embedding, nonlinearity='relu')
        if self.attention:
            self.edge_weight = nn.Parameter(torch.ones(edge_index.shape[1]))
        self.cache_batch_edge_index = None
        self.cache_batch_edge_index_size = -1
        
        self.softmax_mask = torch.zeros(self.gt_matrix.shape[1], dtype=torch.bool)
        for start, end in self.multiclass_list:
            self.softmax_mask[start:end] = True
        self.sigmoid_mask = ~self.softmax_mask
        
    def forward(self, x):
        edge_index = self.get_edge_index(x.shape[0], x.device)
        x = x.reshape(-1, self.num, 1) * self.embedding
        x = x.reshape(-1, self.embedding_dim)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index).view(-1, self.num)
        x = self.convert_to_confidence(x)
        return x
    
    def get_edge_index(self, batch_size, device):
        if self.cache_batch_edge_index_size == batch_size:
            pass
        else:
            self.cache_batch_edge_index_size = batch_size
            offset = torch.arange(0, batch_size * self.gt_matrix.shape[1], self.gt_matrix.shape[1]).view(1, -1).repeat(self.edge_index.shape[1], 1).t().reshape(1, -1)
            updated_edge_index = (self.edge_index.repeat(1, batch_size) + offset)
            updated_edge_index = SparseTensor(row=updated_edge_index[0], col=updated_edge_index[1]).t()
            self.cache_batch_edge_index = updated_edge_index.to(device)
        if self.attention:
            self.cache_batch_edge_index.set_value_(self.edge_weight.sigmoid().repeat(batch_size), layout='coo')
        return self.cache_batch_edge_index
    
    def get_w(self):
        return self.w.data
    
    def convert_to_confidence(self, logits):
        phi = torch.zeros_like(logits)
        for start, end in self.multiclass_list:
            phi[:, start:end] = torch.softmax(logits[:, start:end], dim=1)
        phi[:, self.sigmoid_mask] = torch.sigmoid(logits[:, self.sigmoid_mask])
        return phi
    
    def E_step(self, x, y):
        device = x.device
        x = x.float().clamp(min=EPSILON, max = 1 - EPSILON)
        tmp_bias = self.bias.to(device)
        tmp_predicate_indices = self.predicate_indices.to(device)
        tmp_formula_indices = self.formula_indices.to(device)
        tmp_values = self.values.to(device)
        sigmoid_mask = self.sigmoid_mask
        softmax_mask = self.softmax_mask
        
        phi = self.forward(x).clamp(min=EPSILON, max = 1 - EPSILON)
        y = self.gt_matrix[y.tolist()].to(device)
        coeff = []
        logQ = []
        
        for i in range(self.sample_num):
            with torch.no_grad():
                ### sample from Q ###
                sample = torch.zeros_like(phi)
                for start, end in self.multiclass_list:
                    index = Categorical(probs=phi[:, start:end]).sample().reshape(-1, 1) + start
                    sample = sample.scatter(1, index, 1.0)
                sample[:, sigmoid_mask] = Bernoulli(probs=phi[:, sigmoid_mask]).sample()
                
                knowledge_score = torch.zeros(phi.shape[0], len(tmp_bias)).to(device)
                vanilla_score = sample[:, tmp_predicate_indices] * tmp_values
                knowledge_score.index_add_(1, tmp_formula_indices, vanilla_score)
                knowledge_score += tmp_bias
                knowledge_score = self.neg_indicator(knowledge_score)
                
                phi_log = (phi.log() * sample).sum(-1) + (((1 - phi[:, sigmoid_mask]).log() * (1 - sample[:, sigmoid_mask]))).sum(-1)
                ### main score + knowledge score ###
                cur_coeff = phi_log - (knowledge_score * self.get_w()).sum(-1)
                if not self.train_main:
                    cur_coeff -= (x/(1-x)).log_().mul(sample).sum(-1)
                coeff.append(cur_coeff.unsqueeze(-1))
            phi_log =  (phi.log() * sample).sum(-1) + (((1 - phi[:, sigmoid_mask]).log() * (1 - sample[:, sigmoid_mask]))).sum(-1)
            logQ.append(phi_log.unsqueeze(-1))
        logQ = torch.cat(logQ, dim=1)
        coeff = torch.cat(coeff, dim=1)
        coeff -= coeff.mean(1, keepdim=True)
        phi_y_log = (phi.log() * y).sum(-1, keepdim=True) + (((1 - phi[:, sigmoid_mask]).log() * (1 - y[:, sigmoid_mask]))).sum(-1, keepdim=True)
    
# we only care about the output of the main
#         phi_y_log = (phi[:, :self.main_num].log() * y[:, :self.main_num]).sum(-1, keepdim=True)
    
        loss = (coeff * logQ).mean(1, keepdim=True) - self.eta * phi_y_log
#         loss = - self.eta * phi_y_log
        return phi, loss.mean()

    def M_step(self, x):
        device = x.device
        x = x.float().clamp_(min=EPSILON, max = 1 - EPSILON)
        tmp_bias = self.bias.to(device)
        tmp_predicate_indices = self.predicate_indices.to(device)
        tmp_formula_indices = self.formula_indices.to(device)
        tmp_values = self.values.to(device)
        sigmoid_mask = self.sigmoid_mask
        softmax_mask = self.softmax_mask
        
        with torch.no_grad():
            phi = self.forward(x).clamp(min=EPSILON, max = 1 - EPSILON)
            w = self.get_w()
            w_gradient = torch.zeros(len(tmp_bias)).to(device)
            for i in range(self.sample_num):
                sample = torch.zeros_like(phi)
                for start, end in self.multiclass_list:
                    index = Categorical(probs=phi[:, start:end]).sample().reshape(-1, 1) + start
                    sample = sample.scatter(1, index, 1.0)
                sample[:, sigmoid_mask] = Bernoulli(probs=phi[:, sigmoid_mask]).sample()

                knowledge_score = torch.zeros(phi.shape[0], len(tmp_bias)).to(device)
                vanilla_score = sample[:, tmp_predicate_indices] * tmp_values
                knowledge_score.index_add_(1, tmp_formula_indices, vanilla_score)
                knowledge_score += tmp_bias
                knowledge_score = knowledge_score[:, tmp_formula_indices] 
                n1 = self.neg_indicator(knowledge_score)
                knowledge_score = n1 * w[:, tmp_formula_indices] 

                markov_original_score = torch.zeros_like(sample, dtype=torch.float)
                markov_original_score.index_add_(1, tmp_predicate_indices, knowledge_score)
                if not self.train_main:
                    markov_original_score += (x/(1-x)).log_() * sample
                markov_original_score = markov_original_score[:, tmp_predicate_indices]

                flip_knowledge_score = knowledge_score + (1 - 2 * sample[:, tmp_predicate_indices]) * tmp_values 
                n2 = self.neg_indicator(flip_knowledge_score)
                flip_knowledge_score = n2 * w[:, tmp_formula_indices] 

                markov_flip_score = torch.zeros_like(sample, dtype=torch.float)
                markov_flip_score.index_add_(1, tmp_predicate_indices, flip_knowledge_score)
                if not self.train_main:
                    markov_flip_score += (x/(1-x)).log_() * (1-sample)
                markov_flip_score = markov_flip_score[:, tmp_predicate_indices]
                s = torch.softmax(torch.stack([markov_original_score, markov_flip_score], dim=-1), dim=-1)
                
                if self.mode == 'approx':
                    phi[:, sigmoid_mask] = phi[:, sigmoid_mask] * sample[:, sigmoid_mask] + (1-phi[:, sigmoid_mask]) * (1-sample[:, sigmoid_mask])
                    predicate_gradient = (phi[:, tmp_predicate_indices] - s[:,:,0]).mean(0)
                    w_gradient.index_add_(0, tmp_formula_indices, predicate_gradient)
                    break
                else:
                    predicate_gradient = (n1 - n1 * s[:,:,0] - n2 * s[:,:,1]).mean(0) / self.sample_num
                    w_gradient.index_add_(0, tmp_formula_indices, predicate_gradient)
            return w_gradient.unsqueeze(0)

    def neg_indicator(self, mat):
        return torch.where(mat> 0, torch.tensor(0, device=mat.device), torch.tensor(1, device=mat.device))
    
    def get_simulated_input(self, batch_size, percent):
        device = 'cuda'
        x = torch.zeros(batch_size, self.num, device=device)
        labels = torch.tensor(np.random.choice(np.arange(self.main_num), batch_size).tolist()).long()
        gt = self.gt_matrix[labels].to(device)
        sigmoid_mask = self.sigmoid_mask
        softmax_mask = self.softmax_mask

        ## pesudo multi
        for start, end in self.multiclass_list:
            target = gt[:, start:end].argmax(1)
            prob = torch.full((batch_size, end-start), (1 - percent) / (end - start - 1), device=device)
            prob[torch.arange(batch_size), target] = percent
            sampled_target = Categorical(probs=prob).sample()
            x[:, start:end] = self.sample_multi_confidence((batch_size, end-start), sampled_target)
        
        ## pesudo binary
        prob = torch.zeros_like(gt[:, sigmoid_mask])
        prob = torch.where(gt[:, sigmoid_mask] == 1, percent, (1 - percent))
        x[:, sigmoid_mask] = Bernoulli(probs=prob).sample()
        x[:, sigmoid_mask] = torch.where(Bernoulli(probs=prob).sample() == 0, 
                             torch.rand_like(x[:, sigmoid_mask]) * 0.5, 
                             torch.rand_like(x[:, sigmoid_mask]) * 0.5 + 0.5)
        return x, labels

    def sample_multi_confidence(self, shape, targets):
        device = targets.device
        n_classes = shape[1]
        batch_size = shape[0]

        # Draw random values from [0.5, 1) for target classes
        target_values = 0.5 + torch.rand(batch_size, device=device) * 0.5

        # Compute remaining confidence for each sample
        remaining_confidence = 1 - target_values

        # Generate random values for non-target classes
        non_target_values = torch.rand(batch_size, n_classes, device=device)

        # Set the target index value to 0, it will be replaced later by target_values
        non_target_values[torch.arange(batch_size), targets] = 0

        # Normalize so the sum of the random values equals remaining confidence
        sum_non_target_values = non_target_values.sum(dim=1, keepdim=True)
        non_target_values = non_target_values * (remaining_confidence.view(-1, 1) / sum_non_target_values)

        # Assign high confidence to target classes
        non_target_values[torch.arange(batch_size, device=device), targets] = target_values

        return non_target_values


def load_gcn_from_ckpt(checkpoint, device):
    gcn_model = get_gcn(checkpoint['dataset'],
                        checkpoint['knowledge_path'],
                        checkpoint['eta'], 
                        checkpoint['sample_num'], 
                        checkpoint['embedding_dim'], 
                        checkpoint['hidden_dim'], 
                        checkpoint['train_main'], 
                        checkpoint['attention'], 
                        checkpoint['mode'],
                        device = device)
    gcn_model.load_state_dict(checkpoint['gcn_state_dict'])
    return gcn_model