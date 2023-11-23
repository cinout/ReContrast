import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
from models.resnet import wide_resnet50_2
from models.de_resnet import (
    de_wide_resnet50_2,
)
from models.recontrast import ReContrast, ReContrast
import argparse
from utils import global_cosine_hm
import copy
from tqdm import tqdm

# import os
# from torch.utils.data import DataLoader
# from dataset import MVTecDataset
# import torch.backends.cudnn as cudnn
# from ptflops import get_model_complexity_info
# from torch.nn import functional as F
# from functools import partial
# import warnings
# import logging

# warnings.filterwarnings("ignore")


def train(_class_, args):
    print(_class_)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_iters = 2000
    batch_size = 16
    image_size = 256
    crop_size = 256

    data_transform, _ = get_data_transforms(image_size, crop_size)
    train_path = "datasets/loco/" + _class_ + "/train"
    # test_path = "datasets/mvtec_anomaly_detection/" + _class_

    train_data = ImageFolder(root=train_path, transform=data_transform)
    # test_data = MVTecDataset(
    #     root=test_path,
    #     transform=data_transform,
    #     gt_transform=gt_transform,
    #     phase="test",
    # )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False
    )
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_data, batch_size=1, shuffle=False, num_workers=1
    # )

    # TODO: change to 100
    encoder, bn = wide_resnet50_2(pretrained=True)
    # TODO:
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    # TODO:
    model = ReContrast(
        encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder
    )
    # for m in encoder.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eps = 1e-8

    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(bn.parameters()),
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    optimizer2 = torch.optim.AdamW(
        list(encoder.parameters()), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5
    )
    print("train image number:{}".format(len(train_data)))
    # print("test image number:{}".format(len(test_data)))

    # auroc_px_best, auroc_sp_best, aupro_px_best = 0, 0, 0
    it = 0
    for epoch in tqdm(range(int(np.ceil(total_iters / len(train_dataloader))))):
        print(f"---epoch: {epoch}---")
        # encoder batchnorm in eval for these classes.
        # model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])
        model.train(encoder_bn_train=True)

        # loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            en, de = model(img)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            # TODO:
            loss = (
                global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.0) / 2
                + global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.0) / 2
            )

            # loss = global_cosine(en[:3], de[:3], stop_grad=False) / 2 + \
            #        global_cosine(en[3:], de[3:], stop_grad=False) / 2

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer2.step()
            # loss_list.append(loss.item())

            # if (it + 1) % 250 == 0:
            #     auroc_px, auroc_sp, aupro_px = evaluation(
            #         model, test_dataloader, device
            #     )
            #     model.train(
            #         encoder_bn_train=_class_
            #         not in ["toothbrush", "leather", "grid", "tile", "wood", "screw"]
            #     )

            #     print(
            #         "Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}".format(
            #             auroc_px, auroc_sp, aupro_px
            #         )
            #     )
            #     if auroc_sp >= auroc_sp_best:
            #         auroc_px_best, auroc_sp_best, aupro_px_best = (
            #             auroc_px,
            #             auroc_sp,
            #             aupro_px,
            #         )
            it += 1
            if it == total_iters:
                break
        print("iter [{}/{}], loss:{:.4f}".format(it, total_iters, loss.item()))

    # visualize(model, test_dataloader, device, _class_=_class_, save_name=args.save_name)
    # return auroc_px, auroc_sp, aupro_px, auroc_px_best, auroc_sp_best, aupro_px_best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    item_list = [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for item in item_list:
        train(item, args)
