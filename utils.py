import json
import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
)
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
import os
from functools import partial
import math
from tqdm import tqdm


class IndividualGTLoss(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        defects_config_path = os.path.join(
            "datasets/loco/", args.subdataset, "defects_config.json"
        )
        defects = json.load(open(defects_config_path))
        self.config = {e["pixel_value"]: e for e in defects}

        self.gamma = 2
        self.smooth = 1e-5
        self.size_average = True

    def forward(self, predicted, gts):
        loss_per_gt = []
        for gt in gts:
            # gt.shape: [1, 1, orig.h, orig.w]
            # find unique config for the gt
            unique_values = sorted(torch.unique(gt).detach().cpu().numpy())
            pixel_type = unique_values[-1]
            pixel_detail = self.config[pixel_type]
            saturation_threshold = pixel_detail["saturation_threshold"]
            relative_saturation = pixel_detail["relative_saturation"]

            # calculate saturation_area (max pixels needed)
            bool_array = gt.cpu().numpy().astype(np.bool_)
            defect_area = np.sum(bool_array)
            saturation_area = (
                int(saturation_threshold * defect_area)
                if relative_saturation
                else np.minimum(saturation_threshold, defect_area)
            )

            # apply modified focal_loss
            num_class = predicted.shape[0]
            predicted = predicted.view(predicted.shape[0], -1)
            predicted = predicted.transpose(0, 1)  # shape: (H*W, 2)

            gt = gt.bool().to(torch.float32)
            gt = gt.squeeze(0)
            gt = gt.view(gt.shape[0], -1)
            gt = gt.transpose(0, 1)

            idx = gt.cpu().long()
            one_hot_key = torch.FloatTensor(gt.size(0), num_class).zero_()
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            if one_hot_key.device != predicted.device:
                one_hot_key = one_hot_key.to(predicted.device)
            if self.smooth:
                one_hot_key = torch.clamp(
                    one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
                )
            pt = (one_hot_key * predicted).sum(1) + self.smooth

            # use mask to only calculate loss of positive pixels
            mask = (gt == 1).squeeze(1)
            pt = torch.masked_select(pt, mask)

            logpt = pt.log()
            loss = -1 * torch.pow((1 - pt), self.gamma) * logpt

            print(loss)
            print(loss.shape)
            print("-----------")
            saturated_loss_values, _ = torch.topk(
                loss, k=saturation_area, largest=False
            )
            print(saturated_loss_values)
            print(saturated_loss_values.shape)
            print("=========**********=========**********")

            loss_per_gt.append(saturated_loss_values)

        loss_per_gt = torch.cat(loss_per_gt, dim=0)
        if self.size_average:
            loss_per_gt = loss_per_gt.mean()
        return loss_per_gt


class FocalLoss(torch.nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py

    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    """

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma = 2
        self.smooth = 1e-5
        self.size_average = True

    def forward(self, logit, target):
        # logit.shape: [bs, 2, h, w], sum of dim 1 is 1, because softmaxed
        # target.shape: [bs, 1, h, w], values are either 0 or 1, in float data type
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))  # flatten to [N*h*w, C]
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)  # [N*h*w, 1]

        idx = target.cpu().long()  # [N*h*w, 1]

        one_hot_key = torch.FloatTensor(
            target.size(0), num_class
        ).zero_()  # [N*h*w, C], all 0s
        one_hot_key = one_hot_key.scatter_(
            1, idx, 1
        )  # [N*h*w, C], with the right C marked with 1, and the other marked with 0
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )  # with values changed from {0, 1} to {smooth, 1-smooth}

        pt = (one_hot_key * logit).sum(1) + self.smooth

        # USE mask to only calculate loss of negative pixels
        mask = (target == 0).squeeze(1)
        pt = torch.masked_select(pt, mask)

        logpt = pt.log()

        loss = -1 * torch.pow((1 - pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


def modify_grad(x, inds, factor=0.0):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        if stop_grad:
            loss += (
                torch.mean(
                    1
                    - cos_loss(
                        a[item].view(a[item].shape[0], -1).detach(),
                        b[item].view(b[item].shape[0], -1),
                    )
                )
                * weight[item]
            )
        else:
            loss += (
                torch.mean(
                    1
                    - cos_loss(
                        a[item].view(a[item].shape[0], -1),
                        b[item].view(b[item].shape[0], -1),
                    )
                )
                * weight[item]
            )
    return loss


def global_cosine_hm(a, b, alpha=1.0, factor=0.0):
    # hard mining
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


def region_cosine(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode="mul", log=False):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == "mul":
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map[0, 0, :, :].to("cpu").detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True, max_ratio=0):
    """

    :param model:
    :param dataloader:
    :param device:
    :param _class_:
    :param calc_pro:
    :param max_ratio: if 0, use the max value of anomaly map as the image anomaly score.
     if 0.1, use the mean of max 10% anomaly map value, etc.
    :return:
    """
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(
                        compute_pro(
                            gt.squeeze(0).cpu().numpy().astype(int),
                            anomaly_map[np.newaxis, :, :],
                        )
                    )

            if max_ratio <= 0:
                sp_score = anomaly_map.max()
            else:
                anomaly_map = anomaly_map.ravel()
                sp_score = np.sort(anomaly_map)[
                    -int(anomaly_map.shape[0] * max_ratio) :
                ]
                sp_score = sp_score.mean()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def evaluation_batch(
    model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0
):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, cls in dataloader:
            img = img.to(device)
            if reg_calib:
                if hasattr(model, "require_cls"):
                    output = model(img, cls)
                else:
                    output = model(img)
                en, de, reg = output[0], output[1], output[2]
            else:
                output = model(img)
                en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

            if reg_calib:
                reg_mean = reg[:, 0].view(-1, 1, 1, 1)
                reg_max = reg[:, 1].view(-1, 1, 1, 1)
                anomaly_map = (anomaly_map - reg_mean) / (reg_max - reg_mean)
                # anomaly_map = (anomaly_map - reg_max) / (reg_max - reg_mean)

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][
                    :, : int(anomaly_map.shape[1] * max_ratio)
                ]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, ap_px, ap_sp


def evaluation_loco(model, dataloader, device, _class_=None, calc_pro=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            defect_type_list.extend(defect_type)

        gt_list_sp = np.array(gt_list_sp)
        pr_list_sp = np.array(pr_list_sp)
        defect_type_list = np.array(defect_type_list)
        auroc = roc_auc_score(gt_list_sp, pr_list_sp)
        auroc_logic = roc_auc_score(
            gt_list_sp[
                np.logical_or(
                    defect_type_list == "good", defect_type_list == "logical_anomalies"
                )
            ],
            pr_list_sp[
                np.logical_or(
                    defect_type_list == "good", defect_type_list == "logical_anomalies"
                )
            ],
        )
        auroc_struct = roc_auc_score(
            gt_list_sp[
                np.logical_or(
                    defect_type_list == "good",
                    defect_type_list == "structural_anomalies",
                )
            ],
            pr_list_sp[
                np.logical_or(
                    defect_type_list == "good",
                    defect_type_list == "structural_anomalies",
                )
            ],
        )
        auroc_mean = (auroc_logic + auroc_struct) / 2

    return auroc, auroc_logic, auroc_struct, auroc_mean


def evaluation_mask(model, dataloader, device, _class_=None, calc_pro=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, mask, _ in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            if mask.max() > 0:
                # mask = binary_dilation(mask[0, 0].cpu().numpy().astype(int), iterations=2)
                mask = mask[0, 0].cpu().numpy().astype(int)
                anomaly_map = anomaly_map * mask
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(
                        compute_pro(
                            gt.squeeze(0).cpu().numpy().astype(int),
                            anomaly_map[np.newaxis, :, :],
                        )
                    )
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def evaluation_noseg(model, dataloader, device, _class_=None, reduction="max"):
    model.eval()
    gt_list_sp = []
    pr_list_sp = []
    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt_list_sp.append(label.item())
            if reduction == "max":
                pr_list_sp.append(np.max(anomaly_map))
            elif reduction == "mean":
                pr_list_sp.append(np.mean(anomaly_map))

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh)
        f1 = f1_score(gt_list_sp, pr_list_sp >= thresh)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_sp, f1, acc


def visualize(model, dataloader, device, _class_="None", save_name="save"):
    model.eval()
    save_dir = os.path.join("./visualize", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img = (img * 255).astype("uint8")
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = (
                img_path[0].split("/")[-2]
                + "_"
                + img_path[0].split("/")[-1].replace(".png", "")
            )
            cv2.imwrite(save_dir_class + "/" + name + "_seg.png", heatmap)
            cv2.imwrite(save_dir_class + "/" + name + "_cam.png", hm_on_img)

    return


def visualize_noseg(model, dataloader, device, _class_="None", save_name="save"):
    model.eval()
    save_dir = os.path.join("./visualize", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img = (img * 255).astype("uint8")
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = (
                img_path[0].split("/")[-2]
                + "_"
                + img_path[0].split("/")[-1].replace(".png", "")
            )
            cv2.imwrite(save_dir_class + "/" + name + "_seg.png", heatmap)
            cv2.imwrite(save_dir_class + "/" + name + "_cam.png", hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_="None", save_name="save"):
    model.eval()
    save_dir = os.path.join("./visualize", save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(
                anomaly_map,
                dsize=(size[0].item(), size[1].item()),
                interpolation=cv2.INTER_NEAREST,
            )

            save_dir_class = os.path.join(
                save_dir, str(_class_), "test", defect_type[0]
            )
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split("/")[-1].replace(".png", "")
            cv2.imwrite(save_dir_class + "/" + name + ".tiff", anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append(
            {"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True
        )

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma**2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size // 2,
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)
