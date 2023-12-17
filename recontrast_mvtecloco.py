import math
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import random
from models.resnet import wide_resnet50_2
from models.de_resnet import (
    de_wide_resnet50_2,
)
from models.recontrast import LogicalMaskProducer, ReContrast
import argparse
from torchvision import transforms
import copy
from tqdm import tqdm
import tifffile
import os
from datetime import datetime
from functools import partial
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageOps
from dataset import transform_data, LogicalAnomalyDataset
from utils import FocalLoss

timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)

subdataset_mapper = {
    "breakfast_box": "bb",
    "juice_bottle": "jb",
    "pushpins": "pp",
    "screw_bag": "sb",
    "splicing_connectors": "sc",
}


@torch.no_grad()
def predict(
    image,
    model_stg2,
    ref_features,
    args,
    q_structure_start=None,
    q_structure_end=None,
    q_logic_start=None,
    q_logic_end=None,
):
    en, de, pred_mask = model_stg2(
        image, get_ref_features=False, ref_features=ref_features
    )

    map_logic = pred_mask[:, 1, :, :].unsqueeze(1)

    map_structure = torch.zeros((1, 1, args.image_size, args.image_size))
    map_structure = map_structure.to(device)
    for fs, ft in zip(en, de):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)  # [1, 1, res, res]
        a_map = F.interpolate(
            a_map,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=True,
        )
        map_structure += a_map
    map_structure = gaussian_filter(map_structure.to("cpu").detach().numpy(), sigma=4)
    map_structure = torch.tensor(map_structure, dtype=map_logic.dtype)
    map_structure = map_structure.to(device)

    if q_structure_start is not None:
        map_structure = (
            0.1
            * (map_structure - q_structure_start)
            / (q_structure_end - q_structure_start)
        )
    if q_logic_start is not None:
        map_logic = 0.1 * (map_logic - q_logic_start) / (q_logic_end - q_logic_start)
    map_combined = 0.5 * map_structure + 0.5 * map_logic

    return map_combined, map_structure, map_logic


def modify_grad(x, inds, factor=0.0):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm(a, b, alpha=1.0, factor=0.0):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += (
            torch.mean(1 - cos_loss(a_.view(a_.shape[0], -1), b_.view(b_.shape[0], -1)))
            * weight[item]
        )
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    return loss


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, path


def train(args, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir + f"_sd{seed}"

    os.makedirs(output_dir)

    train_output_dir = os.path.join(
        output_dir, "trainings", args.dataset, args.subdataset
    )  # for saving models
    test_output_dir = os.path.join(
        output_dir, "anomaly_maps", args.dataset, args.subdataset, "test"
    )  # for saving tiff files
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    train_path = "datasets/loco/" + args.subdataset + "/train"
    train_data = ImageFolder(root=train_path, transform=transform_data(args.image_size))
    print(f"train image number: {len(train_data)}")

    if args.stg1_ckpt is None:
        """
        --[STAGE 1]--:
        preparing dataset
        """
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size_stg1,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )
        train_dataloader_infinite = InfiniteDataloader(train_dataloader)

        """
        --[STAGE 1]--:
        preparing models
        """
        encoder, bn = wide_resnet50_2(pretrained=True)
        encoder = encoder.to(device)
        bn = bn.to(device)
        encoder_freeze = copy.deepcopy(encoder)

        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
        decoder = decoder.to(device)
        model_stg1 = ReContrast(
            encoder=encoder,
            encoder_freeze=encoder_freeze,
            bottleneck=bn,
            decoder=decoder,
        )
        model_stg1.train(mode=True, encoder_bn_train=True)
        model_stg1 = model_stg1.to(device)

        """
        --[STAGE 1]--:
        preparing optimizers
        """
        optimizer = torch.optim.AdamW(
            list(decoder.parameters()) + list(bn.parameters()),
            lr=2e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
        optimizer2 = torch.optim.AdamW(
            list(encoder.parameters()), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5
        )

        """
        --[STAGE 1]--:
        training
        """
        tqdm_obj = tqdm(range(args.iters_stg1))
        alpha_final = 1
        for iter, (img, label) in zip(tqdm_obj, train_dataloader_infinite):
            img = img.to(device)
            en, de = model_stg1(
                img
            )  # en: {en_freeze, en}, de: {recon_en, recon_en_freeze}

            alpha = min(
                -3 + (alpha_final - -3) * iter / (args.iters_stg1 * 0.1), alpha_final
            )

            loss = (
                global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.0) / 2
                + global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.0) / 2
            )

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()

            if iter % 20 == 0:
                print(
                    "iter [{}/{}], loss:{:.4f}".format(
                        iter, args.iters_stg1, loss.item()
                    )
                )
        torch.save(
            model_stg1.state_dict(), os.path.join(train_output_dir, f"model_stg1.pth")
        )

    """
    --[STAGE 2]--:
    preparing datasets # TODO: [LATER] test with geometric augmentions to images in the future
    """
    train_ref_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        # num_workers=1,
        pin_memory=True,
    )
    normal_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        # num_workers=1,
        pin_memory=True,
    )

    logicano_data = LogicalAnomalyDataset(
        num_logicano=args.num_logicano,
        subdataset=args.subdataset,
        image_size=args.image_size,
    )
    logicano_dataloader = torch.utils.data.DataLoader(
        logicano_data,
        batch_size=1,
        shuffle=True,
        # num_workers=1,
        pin_memory=True,
    )

    logicano_dataloader_infinite = InfiniteDataloader(logicano_dataloader)
    train_ref_dataloader_infinite = InfiniteDataloader(train_ref_dataloader)
    normal_dataloader_infinite = InfiniteDataloader(normal_dataloader)

    """
    --[STAGE 2]--:
    preparing model, including (1) freeze the stg1, (2) attention module, (3) DeConv module, (4) Clustering
    """

    (
        encoder,
        bottleneck,
    ) = wide_resnet50_2()
    encoder_freeze = copy.deepcopy(encoder)
    decoder = de_wide_resnet50_2(output_conv=2)

    pretrained_encoder = {}
    pretrained_encoder_freeze = {}
    pretrained_bottleneck = {}
    pretrained_decoder = {}

    if args.stg1_ckpt is None:
        # load from current model_stg1
        model_stg1_dict = model_stg1.state_dict()
    else:
        # load from local file
        model_stg1_dict = torch.load(args.stg1_ckpt, map_location=device)

    for k, v in model_stg1_dict.items():
        if k.startswith("encoder."):
            pretrained_encoder[k.replace("encoder.", "")] = v
        elif k.startswith("encoder_freeze."):
            pretrained_encoder_freeze[k.replace("encoder_freeze.", "")] = v
        elif k.startswith("bottleneck."):
            pretrained_bottleneck[k.replace("bottleneck.", "")] = v
        elif k.startswith("decoder."):
            pretrained_decoder[k.replace("decoder.", "")] = v
        else:
            raise Exception("Unknown key from model_stg1_dict")
    encoder.load_state_dict(
        pretrained_encoder, strict=False
    )  # because layer4 is not used
    encoder_freeze.load_state_dict(
        pretrained_encoder_freeze, strict=False
    )  # because layer4 is not used
    bottleneck.load_state_dict(pretrained_bottleneck)
    decoder.load_state_dict(pretrained_decoder)

    # prevent loss gradients
    for component in [encoder, encoder_freeze, bottleneck, decoder]:
        for param in component.parameters():
            param.requires_grad = False

    encoder = encoder.to(device)
    encoder_freeze = encoder_freeze.to(device)
    bottleneck = bottleneck.to(device)
    decoder = decoder.to(device)

    model_stg2 = LogicalMaskProducer(
        encoder=encoder,
        bottleneck=bottleneck,
        encoder_freeze=encoder_freeze,
        decoder=decoder,
    )
    model_stg2 = model_stg2.to(device)

    """
    --[STAGE 2]--:
    preparing optimizer
    """
    optimizer = torch.optim.AdamW(
        list(model_stg2.channel_reducer.parameters())
        + list(model_stg2.self_att_module.parameters())
        + list(model_stg2.deconv.parameters()),
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )

    """
    --[STAGE 2]--:
    training
    """
    tqdm_obj = tqdm(range(args.iters_stg2))
    model_stg2.train()
    loss_focal = FocalLoss()
    for iter, refs, logicano, normal in zip(
        tqdm_obj,
        train_ref_dataloader_infinite,
        logicano_dataloader_infinite,
        normal_dataloader_infinite,
    ):
        ref_images, label1 = refs
        normal_image, label2 = normal
        logicano_image = logicano["image"]
        logicano_gt = logicano["gt"]
        _, _, orig_height, orig_width = logicano_gt.shape
        normal_gt = torch.zeros(
            size=(1, 1, orig_height, orig_width), dtype=logicano_gt.dtype
        )

        ref_images = ref_images.to(device)
        normal_image = normal_image.to(device)
        logicano_image = logicano_image.to(device)
        logicano_gt = logicano_gt.to(device)
        normal_gt = normal_gt.to(device)

        image_batch = torch.cat([ref_images, normal_image, logicano_image])
        predicted_masks = model_stg2(
            image_batch
        )  # [2, 2, 256, 256], (1) logical_ano, (2) normal, both softmaxed
        predicted_masks = F.interpolate(
            predicted_masks, (orig_height, orig_width), mode="bilinear"
        )
        gt_masks = torch.cat([logicano_gt, normal_gt], dim=0)  # [2, 1, 256, 256]

        # TODO: [LATER] add other loss functions?
        loss = loss_focal(predicted_masks, gt_masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 50 == 0:
            print(
                "iter [{}/{}], loss:{:.4f}".format(iter, args.iters_stg2, loss.item())
            )

    torch.save(
        model_stg2.state_dict(), os.path.join(train_output_dir, f"model_stg2.pth")
    )

    """
    --[EVALUATION]--:
    use validation set
    """
    model_stg2.eval()
    with torch.no_grad():
        # obtain ref features from train set using p%
        # TODO: [LATER] better algorithms than random?
        ref_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=math.floor(len(train_data) * 0.1),
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )
        for imgs, label in ref_dataloader:
            imgs = imgs.to(device)
            ref_features = model_stg2(imgs, get_ref_features=True)  # [10%, 512, 8, 8]
            break  # we just need the first 10%

        # create validation dataloader
        validation_path = "datasets/loco/" + args.subdataset + "/validation"
        validation_set = ImageFolder(validation_path)

        # calculate quantiles using validation set
        maps_structure = []
        maps_logic = []
        for raw_image, label in tqdm(validation_set):
            orig_width = raw_image.width
            orig_height = raw_image.height
            image = transform_data(args.image_size)(raw_image)
            image = image.unsqueeze(0)
            image = image.to(device)  # [bs, 3, 256, 256]

            _, map_structure, map_logic = predict(image, model_stg2, ref_features, args)

            maps_structure.append(map_structure)
            maps_logic.append(map_logic)
        maps_structure = torch.cat(maps_structure)
        maps_logic = torch.cat(maps_logic)
        q_structure_start = torch.quantile(maps_structure, q=0.9)
        q_structure_end = torch.quantile(maps_structure, q=0.995)
        q_logic_start = torch.quantile(maps_logic, q=0.9)
        q_logic_end = torch.quantile(maps_logic, q=0.995)

    print(f"q_structure_start: {q_structure_start}")
    print(f"q_structure_end: {q_structure_end}")
    print(f"q_logic_start: {q_logic_start}")
    print(f"q_logic_end: {q_logic_end}")

    """
    --[EVALUATION]--:
    create dataloader
    """
    test_path = "datasets/loco/" + args.subdataset + "/test"
    test_data = ImageFolderWithPath(test_path)

    """
    --[EVALUATION]--:
    evaluating
    """

    model_stg2.eval()
    with torch.no_grad():
        for raw_image, path in test_data:
            # path: 'datasets/loco/breakfast_box/test/good/000.png'
            orig_width = raw_image.width
            orig_height = raw_image.height
            image = transform_data(args.image_size)(raw_image)
            image = image.unsqueeze(0)
            image = image.to(device)  # [bs, 3, 256, 256]

            # TODO: just for debugging
            if args.use_validation:
                map_combined, map_structure, map_logic = predict(
                    image,
                    model_stg2,
                    ref_features,
                    args,
                    q_structure_start=q_structure_start,
                    q_structure_end=q_structure_end,
                    q_logic_start=q_logic_start,
                    q_logic_end=q_logic_end,
                )
            else:
                map_combined, map_structure, map_logic = predict(
                    image,
                    model_stg2,
                    ref_features,
                    args,
                )

            # TODO: revert to map_combined in the future
            map_structure = F.interpolate(
                map_structure, (orig_height, orig_width), mode="bilinear"
            )
            map_structure = (
                map_structure[0, 0].cpu().numpy()
            )  # ready to be saved into .tiff format

            defect_class = os.path.basename(os.path.dirname(path))

            if test_output_dir is not None:
                img_nm = os.path.split(path)[1].split(".")[0]
                if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_dir, defect_class))
                file = os.path.join(test_output_dir, defect_class, img_nm + ".tiff")
                tifffile.imwrite(file, map_structure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seeds", type=int, default=[42], nargs="+")
    parser.add_argument("--batch_size_stg1", type=int, default=16)
    parser.add_argument(
        "--num_logicano",
        type=int,
        default=10,
        help="number of real logical anomalies used in training",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--iters_stg1", type=int, default=3000)
    parser.add_argument("--iters_stg2", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default=f"outputs/output_{timestamp}")
    parser.add_argument("--dataset", type=str, default="mvtec_loco")
    parser.add_argument(
        "--subdataset",
        default="breakfast_box",
        choices=[
            "breakfast_box",
            "juice_bottle",
            "pushpins",
            "screw_bag",
            "splicing_connectors",
        ],
        help="sub-datasets of Mvtec LOCO",
    )
    parser.add_argument("--stg1_ckpt", type=str)
    parser.add_argument("--use_validation", action="store_true")

    args = parser.parse_args()
    args.output_dir = args.output_dir + f"_[{subdataset_mapper[args.subdataset]}]"

    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in args.seeds:
        train(args, seed)
