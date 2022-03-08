import os
import sys
import torch

from copy import deepcopy
from models.moco2_module import MocoV2
from torchvision import models


def resnet18(large=True):
    home_dir = os.environ['HOME']
    ckpt_dir = os.path.join(home_dir, 'checkpoints', 'seasonal-contrast')

    if large:
        url = 'https://zenodo.org/record/4728033/files/seco_resnet18_1m.ckpt'
        ckpt_path = os.path.join(ckpt_dir, 'seco_resnet18_1m.ckpt')
    else:
        url = 'https://zenodo.org/record/4728033/files/seco_resnet18_100k.ckpt'
        ckpt_path = os.path.join(ckpt_dir, 'seco_resnet18_100k.ckpt')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(ckpt_path):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, ckpt_path))
        torch.hub.download_url_to_file(url, ckpt_path)

    model = MocoV2.load_from_checkpoint(ckpt_path)

    resnet = models.resnet18(pretrained=False)

    resnet.conv1 = deepcopy(model.encoder_q[0])
    resnet.bn1 = deepcopy(model.encoder_q[1])
    resnet.relu = deepcopy(model.encoder_q[2])
    resnet.maxpool = deepcopy(model.encoder_q[3])
    resnet.layer1 = deepcopy(model.encoder_q[4])
    resnet.layer2 = deepcopy(model.encoder_q[5])
    resnet.layer3 = deepcopy(model.encoder_q[6])
    resnet.layer4 = deepcopy(model.encoder_q[7])

    return resnet


def resnet18_heads(large=True, index=1):
    home_dir = os.environ['HOME']
    ckpt_dir = os.path.join(home_dir, 'checkpoints', 'seasonal-contrast')

    if large:
        url = 'https://zenodo.org/record/4728033/files/seco_resnet18_1m.ckpt'
        ckpt_path = os.path.join(ckpt_dir, 'seco_resnet18_1m.ckpt')
    else:
        url = 'https://zenodo.org/record/4728033/files/seco_resnet18_100k.ckpt'
        ckpt_path = os.path.join(ckpt_dir, 'seco_resnet18_100k.ckpt')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(ckpt_path):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, ckpt_path))
        torch.hub.download_url_to_file(url, ckpt_path)

    model = MocoV2.load_from_checkpoint(ckpt_path)

    head = deepcopy(model.heads_q[index])

    return head


def resnet50(large=True):
    home_dir = os.environ['HOME']
    ckpt_dir = os.path.join(home_dir, 'checkpoints', 'seasonal-contrast')

    if large:
        url = 'https://zenodo.org/record/4728033/files/seco_resnet50_1m.ckpt'
        ckpt_path = os.path.join(ckpt_dir, 'seco_resnet50_1m.ckpt')
    else:
        url = 'https://zenodo.org/record/4728033/files/seco_resnet50_100k.ckpt'
        ckpt_path = os.path.join(ckpt_dir, 'seco_resnet50_100k.ckpt')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(ckpt_path):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, ckpt_path))
        torch.hub.download_url_to_file(url, ckpt_path)

    model = MocoV2.load_from_checkpoint(ckpt_path)

    resnet = models.resnet50(pretrained=False)

    resnet.conv1 = deepcopy(model.encoder_q[0])
    resnet.bn1 = deepcopy(model.encoder_q[1])
    resnet.relu = deepcopy(model.encoder_q[2])
    resnet.maxpool = deepcopy(model.encoder_q[3])
    resnet.layer1 = deepcopy(model.encoder_q[4])
    resnet.layer2 = deepcopy(model.encoder_q[5])
    resnet.layer3 = deepcopy(model.encoder_q[6])
    resnet.layer4 = deepcopy(model.encoder_q[7])

    return resnet