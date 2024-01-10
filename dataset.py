from torchvision import transforms
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageOps
import random
import math

torch.multiprocessing.set_sharing_strategy("file_system")


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


def transform_gt_into_tensor():
    gt_transforms = transforms.Compose(
        [
            # transforms.Resize((size, size)),
            transforms.ToTensor()
        ]
    )
    return gt_transforms


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
        ]
    )
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            # transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    return data_transforms


class LogicalAnomalyDataset(Dataset):
    def __init__(self, num_logicano, subdataset, image_size) -> None:
        super().__init__()
        self.image_size = image_size
        logical_anomaly_path = "datasets/loco/" + subdataset + "/test/logical_anomalies"
        logical_anomaly_gt_path = (
            "datasets/loco/" + subdataset + "/ground_truth/logical_anomalies"
        )
        all_logical_anomalies = sorted(os.listdir(logical_anomaly_path))
        selected_indices = [
            x.split(".png")[0]
            for x in random.sample(
                all_logical_anomalies,
                k=math.floor(num_logicano * len(all_logical_anomalies)),
            )
        ]
        self.images = [logical_anomaly_path + f"/{idx}.png" for idx in selected_indices]
        # TODO: [LATER] test with geometric augmentions to images in the future
        self.gt = [
            glob.glob(logical_anomaly_gt_path + f"/{idx}/*.png")
            for idx in selected_indices
        ]

    def __len__(self):
        return len(self.images)

    def transform_image(self, path):
        img = Image.open(path)
        img = img.convert("RGB")
        return transform_data(self.image_size)(img)

    def transform_gt(self, paths):
        overall_gt = None  # purpose is to determine all negative (normal) pixels
        individual_gts = []
        for each_path in paths:
            gt = Image.open(each_path)
            gt = np.array(gt)
            gt = torch.tensor(gt)
            gt = gt.unsqueeze(0)
            if overall_gt is not None:
                overall_gt = torch.logical_or(overall_gt, gt)
            else:
                overall_gt = gt

            individual_gts.append(gt)

        overall_gt = overall_gt.bool().to(torch.float32)
        return overall_gt, individual_gts

    def __getitem__(self, index):
        img_path = self.images[index]
        image = self.transform_image(img_path)

        gt_paths = self.gt[index]
        overall_gt, individual_gts = self.transform_gt(gt_paths)

        # overall_gt.shape: [1, orig.height, orig.width]

        sample = {
            "image": image,
            "overall_gt": overall_gt,
            "individual_gts": individual_gts,
            "img_path": img_path,
        }
        return sample


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == "train":
            self.img_path = os.path.join(root, "train")
        else:
            self.img_path = os.path.join(root, "test")
            self.gt_path = os.path.join(root, "ground_truth")
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        (
            self.img_paths,
            self.gt_paths,
            self.labels,
            self.types,
        ) = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == "good":
                img_paths = glob.glob(
                    os.path.join(self.img_path, defect_type) + "/*.png"
                ) + glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(["good"] * len(img_paths))
            else:
                img_paths = glob.glob(
                    os.path.join(self.img_path, defect_type) + "/*.png"
                ) + glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(
            gt_tot_paths
        ), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = (
            self.img_paths[idx],
            self.gt_paths[idx],
            self.labels[idx],
            self.types[idx],
        )
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MVTecSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, mask_path, transform, gt_transform, phase):
        if phase == "train":
            self.img_path = os.path.join(root, "train")
        else:
            self.img_path = os.path.join(root, "test")
            self.gt_path = os.path.join(root, "ground_truth")
            self.mask_path = os.path.join(mask_path, "test")

        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        (
            self.img_paths,
            self.gt_paths,
            self.mask_paths,
            self.labels,
            self.types,
        ) = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        mask_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == "good":
                img_paths = glob.glob(
                    os.path.join(self.img_path, defect_type) + "/*.png"
                )
                mask_paths = glob.glob(
                    os.path.join(self.mask_path, defect_type) + "/*.png"
                )

                img_paths.sort()
                mask_paths.sort()

                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                mask_tot_paths.extend(mask_paths)

                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(["good"] * len(img_paths))
            else:
                img_paths = glob.glob(
                    os.path.join(self.img_path, defect_type) + "/*.png"
                )
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                mask_paths = glob.glob(
                    os.path.join(self.mask_path, defect_type) + "/*.png"
                )

                img_paths.sort()
                gt_paths.sort()
                mask_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                mask_tot_paths.extend(mask_paths)

                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(
            gt_tot_paths
        ), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, mask_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = (
            self.img_paths[idx],
            self.gt_paths[idx],
            self.labels[idx],
            self.types[idx],
        )
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        mask = Image.open(self.mask_paths[idx])
        mask = self.gt_transform(mask)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, mask, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == "train":
            self.img_path = os.path.join(root, "train")
        else:
            self.img_path = os.path.join(root, "test")
            self.gt_path = os.path.join(root, "ground_truth")
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        (
            self.img_paths,
            self.gt_paths,
            self.labels,
            self.types,
        ) = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == "good":
                img_paths = glob.glob(
                    os.path.join(self.img_path, defect_type) + "/*.png"
                )
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(["good"] * len(img_paths))
            else:
                img_paths = glob.glob(
                    os.path.join(self.img_path, defect_type) + "/*.png"
                )
                gt_paths = glob.glob(
                    os.path.join(self.gt_path, defect_type) + "/*/000.png"
                )
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(
            gt_tot_paths
        ), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = (
            self.img_paths[idx],
            self.gt_paths[idx],
            self.labels[idx],
            self.types[idx],
        )
        img = Image.open(img_path).convert("RGB")
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == "train":
            self.img_path = os.path.join(root, "train")
        else:
            self.img_path = os.path.join(root, "test")
        self.transform = transform
        self.phase = phase
        # load dataset
        (
            self.img_paths,
            self.labels,
        ) = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == "NORMAL":
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == "train":
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label, img_path
