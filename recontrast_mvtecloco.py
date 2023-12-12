import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
from models.resnet import wide_resnet50_2
from models.de_resnet import (
    de_wide_resnet50_2,
)
from models.recontrast import ReContrast
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


def transform_data(size):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    return data_transforms


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

    """
    --[STAGE 1]--:
    preparing dataset
    """

    train_path = "datasets/loco/" + args.subdataset + "/train"
    train_data = ImageFolder(root=train_path, transform=transform_data(args.image_size))
    print(f"train image number: {len(train_data)}")
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
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)
    model = ReContrast(
        encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder
    )
    model.train(mode=True, encoder_bn_train=True)

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
        en, de = model(img)  # en: {en_freeze, en}, de: {recon_en, recon_en_freeze}

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
                "iter [{}/{}], loss:{:.4f}".format(iter, args.iters_stg1, loss.item())
            )
    torch.save(model.state_dict(), os.path.join(train_output_dir, f"model_stg1.pth"))
    # visualize(model, test_dataloader, device, _class_=args.subdataset, save_name=args.save_name)

    # validation_path = "datasets/loco/" + args.subdataset + "/validation"  # TODO: use it

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
    model.eval()
    with torch.no_grad():
        for raw_image, path in test_data:
            # path: 'datasets/loco/breakfast_box/test/good/000.png'
            orig_width = raw_image.width
            orig_height = raw_image.height
            image = transform_data(args.image_size)(raw_image)
            image = image.unsqueeze(0)

            image = image.to(device)  # [bs, 3, 256, 256]

            en, de = model(image)
            anomaly_map = np.zeros((orig_height, orig_width))
            for fs, ft in zip(en, de):
                a_map = 1 - F.cosine_similarity(fs, ft)
                a_map = torch.unsqueeze(a_map, dim=1)  # [bs, 1, res, res]
                a_map = F.interpolate(
                    a_map,
                    size=(orig_height, orig_width),
                    mode="bilinear",
                    align_corners=True,
                )
                a_map = a_map[0, 0, :, :].to("cpu").detach().numpy()
                anomaly_map += a_map
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            defect_class = os.path.basename(os.path.dirname(path))

            if test_output_dir is not None:
                img_nm = os.path.split(path)[1].split(".")[0]
                if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_dir, defect_class))
                file = os.path.join(test_output_dir, defect_class, img_nm + ".tiff")
                tifffile.imwrite(file, anomaly_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seeds", type=int, default=[42], nargs="+")
    parser.add_argument("--batch_size_stg1", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--iters_stg1", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default=f"outputs_{timestamp}")
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
    args = parser.parse_args()
    args.output_dir = args.output_dir + f"_[{subdataset_mapper[args.subdataset]}]"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in args.seeds:
        train(args, seed)
