import math
import cv2
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from torch import optim
from tqdm import tqdm
import tifffile
import os
from datetime import datetime
from functools import partial
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from dataset import transform_data, LogicalAnomalyDataset, MyDummyDataset
from utils import FocalLoss, IndividualGTLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


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
    # if args.debug_mode_2 and q_structure_start is not None:
    #     max_index = model_stg2(
    #         image, get_ref_features=False, ref_features=ref_features, args=args
    #     )
    #     return max_index
    # else:
    en, de, pred_mask = model_stg2(
        image, get_ref_features=False, ref_features=ref_features, args=args
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
    train_data = ImageFolderWithPath(
        root=train_path, transform=transform_data(args.image_size)
    )
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
        model_stg1_dict = model_stg1.state_dict()
        torch.save(model_stg1_dict, os.path.join(train_output_dir, f"model_stg1.pth"))

    """
    --[STAGE 2]--:
    preparing datasets
    """
    if args.fixed_ref:
        used_ref_count = math.floor(len(train_data) * args.fixed_ref_percent)
        train_data_list = list(train_data)
        random.shuffle(train_data_list)
        train_ref_dataloader = torch.utils.data.DataLoader(
            train_data_list[:used_ref_count],
            batch_size=32,
            shuffle=False,
            # num_workers=1,
            pin_memory=True,
        )
    else:
        train_ref_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=8,
            shuffle=True,
            # num_workers=1,
            pin_memory=True,
        )
    train_ref_dataloader_infinite = InfiniteDataloader(train_ref_dataloader)

    normal_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        # num_workers=1,
        pin_memory=True,
    )
    logicano_data = LogicalAnomalyDataset(
        logicano_select=args.logicano_select,
        num_logicano=args.num_logicano,
        percent_logicano=args.percent_logicano,
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
    normal_dataloader_infinite = InfiniteDataloader(normal_dataloader)

    """
    --[STAGE 2]--:
    preparing model
    """

    if args.stg1_ckpt is None:
        # load from current model_stg1
        model_stg1.eval()
    else:
        # load from local file
        encoder, bn = wide_resnet50_2(pretrained=False)
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
        model_stg1_dict = torch.load(args.stg1_ckpt, map_location=device)
        model_stg1.load_state_dict(model_stg1_dict)
        model_stg1 = model_stg1.to(device)
        model_stg1.eval()

    model_stg2 = LogicalMaskProducer(
        model_stg1=model_stg1,
        args=args,
    )
    model_stg2 = model_stg2.to(device)

    if args.debug_mode_3:
        # test_path = "datasets/loco/" + args.subdataset + "/test"
        # test_data = ImageFolderWithPath(test_path)

        train_path = "datasets/loco/" + args.subdataset + "/train"
        train_data = ImageFolderWithPath(root=train_path)

        model_stg2.eval()
        normal_features = []
        # logicano_features = []
        # structureano_features = []
        with torch.no_grad():
            for raw_image, path in train_data:
                image = transform_data(args.image_size)(raw_image)
                image = image.unsqueeze(0)
                image = image.to(device)
                x = model_stg2(image, args)
                ano_type = path.split("/")[-2]
                if ano_type == "good":
                    normal_features.append(x)
                # elif ano_type == "logical_anomalies":
                #     logicano_features.append(x)
                # elif ano_type == "structural_anomalies":
                #     structureano_features.append(x)
                else:
                    raise Exception("Unimplemented ano_type")

        normal_features = torch.cat(normal_features, dim=0)
        # logicano_features = torch.cat(logicano_features, dim=0)
        # structureano_features = torch.cat(structureano_features, dim=0)
        print(normal_features.shape)
        # print(logicano_features.shape)
        # print(structureano_features.shape)
        save_dir_name = f"feature_maps_{args.subdataset}/"
        os.makedirs(save_dir_name)
        with open(os.path.join(save_dir_name, "normal_features.t"), "wb") as f:
            torch.save(normal_features, f)
        # with open(os.path.join(save_dir_name, "logic_anomaly_features.t"), "wb") as f:
        #     torch.save(logicano_features, f)
        # with open(
        #     os.path.join(save_dir_name, "structure_anomaly_features.t"), "wb"
        # ) as f:
        #     torch.save(structureano_features, f)

        exit()

    if args.fixed_ref:
        # obtain ref's ref_features early in the process
        model_stg2.eval()
        ref_features = []
        ref_path_names = []
        for imgs, path in train_ref_dataloader:
            imgs = imgs.to(device)
            batch_ref_features = model_stg2(
                imgs, get_ref_features=True, args=args
            )  # [10%, 512, 8, 8]
            ref_features.append(batch_ref_features)
            ref_path_names = ref_path_names + list(path)

        ref_features = torch.cat(ref_features, dim=0)
        num_ref = ref_features.shape[0]

        # find closest ref here
        logicanos_for_train = []
        for logicano in logicano_dataloader:
            max_sim = -1000
            max_index = None

            logicano_image = logicano["image"]
            logicano_image = logicano_image.to(device)
            logicano_image = model_stg2.model_stg1.encoder(logicano_image)
            logicano_image = model_stg2.model_stg1.bottleneck(logicano_image)

            for i in range(num_ref):
                ref_feature = ref_features[i]

                if args.similarity_priority == "pointwise":
                    sim = F.cosine_similarity(ref_feature, logicano_image, dim=0).mean()
                # elif args.similarity_priority == "flatten":
                #     sim = F.cosine_similarity(
                #         torch.mean(ref, dim=(1, 2)),
                #         torch.mean(logicano, dim=(1, 2)),
                #         dim=0,
                #     )
                else:
                    raise Exception("Unimplemented similarity_priority")

                if sim > max_sim:
                    max_sim = sim
                    max_index = i
            logicano["image"] = logicano_image
            logicano["max_ref_index"] = max_index

            logicanos_for_train.append(logicano)
        normals_for_train = []
        for normal in normal_dataloader:
            max_sim = -1000
            max_index = None

            normal_image, img_path = normal
            normal_image = normal_image.to(device)
            normal_image = model_stg2.model_stg1.encoder(normal_image)
            normal_image = model_stg2.model_stg1.bottleneck(normal_image)

            for i in range(num_ref):
                ref_path = ref_path_names[i]
                ref_path = ref_path.split("/")[-1]
                img_path = img_path[0].split("/")[-1]
                if ref_path == img_path:
                    # should not be the same image
                    continue

                ref_feature = ref_features[i]

                if args.similarity_priority == "pointwise":
                    sim = F.cosine_similarity(ref_feature, normal_image, dim=0).mean()
                else:
                    raise Exception("Unimplemented similarity_priority")

                if sim > max_sim:
                    max_sim = sim
                    max_index = i

            normals_for_train.append(
                {"image": normal_image, "max_ref_index": max_index}
            )

        model_stg2.train()

    if args.stg2_ckpt is None:
        """
        --[STAGE 2]--:
        preparing optimizer
        """
        # TODO: check when you alter modules
        if args.compress_bn:
            model_parameters = list(model_stg2.bn_compressor.parameters()) + list(
                model_stg2.deconv.parameters()
            )
        else:
            model_parameters = list(model_stg2.self_att_module.parameters()) + list(
                model_stg2.deconv.parameters()
            )
        # list(model_stg2.channel_reducer.parameters())

        if args.optimizer == "Adam":
            optimizer = torch.optim.AdamW(
                model_parameters,
                lr=args.lr_stg2,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            )
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(
                model_parameters,
                lr=args.lr_stg2,
                momentum=0.9,
                weight_decay=0.00003,
            )
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=0.000001,
                T_max=args.iters_stg2,
            )
        else:
            raise Exception("Unimplemented optimiser")
        """
        --[STAGE 2]--:
        training
        """
        writer = SummaryWriter(
            log_dir=f"./runs/{timestamp}_{args.subdataset}_iter{args.iters_stg2}_{args.loss_mode}"
        )  # Writer will output to ./runs/ directory by default. You can change log_dir in here
        tqdm_obj = tqdm(range(args.iters_stg2))
        model_stg2.train()
        loss_focal = FocalLoss()
        loss_individual_gt = IndividualGTLoss(args)

        # if args.debug_mode:
        #     logicano_fixed = list(logicano_dataloader)[0 : args.debug_logicano_count]
        #     # "datasets/loco/breakfast_box/test/logical_anomalies/073.png"
        #     # "datasets/loco/juice_bottle/test/logical_anomalies/008.png"
        #     # "datasets/loco/splicing_connectors/test/logical_anomalies/073.png"
        #     # "datasets/loco/pushpins/test/logical_anomalies/073.png"
        #     # "datasets/loco/screw_bag/test/logical_anomalies/008.png"

        #     normal_dataloader = torch.utils.data.DataLoader(
        #         ImageFolderWithPath(
        #             root=train_path, transform=transform_data(args.image_size)
        #         ),
        #         batch_size=1,
        #         shuffle=False,
        #         # num_workers=1,
        #         pin_memory=True,
        #     )
        #     if args.debug_normal_count == "all":
        #         normal_fixed = normal_dataloader
        #     elif args.debug_normal_count == "follow":
        #         normal_fixed = list(normal_dataloader)[0 : args.debug_logicano_count]
        #     else:
        #         raise Exception("Unimplemented debug_normal_count")
        #     # "datasets/loco/breakfast_box/train/good/000.png"

        #     logicano_fixed_dataloader_infinite = InfiniteDataloader(logicano_fixed)
        #     normal_fixed_dataloader_infinite = InfiniteDataloader(normal_fixed)
        if args.fixed_ref:
            logicanos_for_train = MyDummyDataset(logicanos_for_train)
            normals_for_train = MyDummyDataset(normals_for_train)

            logicanos_for_train_dataloader = torch.utils.data.DataLoader(
                logicanos_for_train,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )

            normals_for_train_dataloader = torch.utils.data.DataLoader(
                normals_for_train,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )

            logicano_dataloader_infinite = InfiniteDataloader(
                logicanos_for_train_dataloader
            )
            normal_dataloader_infinite = InfiniteDataloader(
                normals_for_train_dataloader
            )

        for iter, refs, logicano, normal in zip(
            tqdm_obj,
            train_ref_dataloader_infinite,
            logicano_dataloader_infinite,
            normal_dataloader_infinite
            # logicano_fixed_dataloader_infinite
            # if args.debug_mode
            # else logicano_dataloader_infinite,
            # normal_fixed_dataloader_infinite
            # if args.debug_mode
            # else normal_dataloader_infinite,
        ):
            if args.fixed_ref:
                logicano_image = logicano["image"][
                    0
                ]  # already feature map, [1, 2048, 8, 8]
                overall_gt = logicano["overall_gt"][0]  # [1, 1, orig.h, orig.w]
                individual_gts = logicano["individual_gts"][0]
                logicano_max_ref_index = logicano["max_ref_index"][0]
                logicano_ref = ref_features[logicano_max_ref_index]
                _, _, orig_height, orig_width = overall_gt.shape

                overall_gt = overall_gt.to(device)
                individual_gts = [item.to(device) for item in individual_gts]

                normal_image = normal["image"][0]  # already feature map
                normal_max_ref_index = normal["max_ref_index"][0]
                normal_ref = ref_features[normal_max_ref_index]

                logicano_input = torch.cat([logicano_ref, logicano_image[0]])
                normal_input = torch.cat([normal_ref, normal_image[0]])

                input = torch.stack(
                    [logicano_input, normal_input], dim=0
                )  # shape: [2, 4096, 8, 8]

                predicted_masks = model_stg2(input, args=args)
            else:
                ref_images, label1 = refs
                normal_image, label2 = normal
                logicano_image = logicano["image"]
                overall_gt = logicano["overall_gt"]  # [1, 1, orig.h, orig.w]
                individual_gts = logicano["individual_gts"]
                _, _, orig_height, orig_width = overall_gt.shape

                ref_images = ref_images.to(device)
                normal_image = normal_image.to(device)
                logicano_image = logicano_image.to(device)
                overall_gt = overall_gt.to(device)
                individual_gts = [item.to(device) for item in individual_gts]

                if args.logicano_only:
                    image_batch = torch.cat([ref_images, logicano_image])
                else:
                    image_batch = torch.cat([ref_images, logicano_image, normal_image])

                predicted_masks = model_stg2(
                    image_batch, args=args
                )  # [2, 2, 256, 256], bs(1) logical_ano, bs(2) normal, both softmaxed

            predicted_masks = F.interpolate(
                predicted_masks, (orig_height, orig_width), mode="bilinear"
            )

            if not args.logicano_only:
                normal_gt = torch.zeros(
                    size=(1, 1, orig_height, orig_width), dtype=overall_gt.dtype
                )
                normal_gt = normal_gt.to(device)
                overall_gt = torch.cat(
                    [overall_gt, normal_gt], dim=0
                )  # [2, 1, orig.h, orig.w]

            # loss_focal for overall negative target pixels only
            loss_overall_negative = loss_focal(predicted_masks, overall_gt)
            # loss for positive pixels in individual gts
            loss_individual_positive = loss_individual_gt(
                predicted_masks[0], individual_gts
            )
            loss = loss_overall_negative + loss_individual_positive
            writer.add_scalar("Loss/train", loss, iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.optimizer == "SGD":
                scheduler.step()

            if iter % 20 == 0:
                print(
                    "iter [{}/{}], loss:{:.4f}".format(
                        iter, args.iters_stg2, loss.item()
                    )
                )
        writer.flush()  # Call flush() method to make sure that all pending events have been written to disk
        writer.close()  # if you do not need the summary writer anymore, call close() method.
        model_stg2_dict = model_stg2.state_dict()
        torch.save(model_stg2_dict, os.path.join(train_output_dir, f"model_stg2.pth"))
    else:
        model_stg2_dict = torch.load(args.stg2_ckpt, map_location=device)
        model_stg2.load_state_dict(model_stg2_dict)

    # # compare key values
    # keys_with_diff = []
    # for key in model_stg1_dict.keys():
    #     outcome = torch.all(
    #         torch.eq(
    #             model_stg1_dict[key],
    #             model_stg2_dict["model_stg1." + key],
    #         )
    #     )
    #     if outcome == False:
    #         keys_with_diff.append(key)
    # print("------keys_with_diff------")
    # print(keys_with_diff)
    # print("------[END] keys_with_diff------")

    """
    --[DEBUG_MODE]--
    """

    # if args.debug_mode:
    #     heatmap_alpha = 0.5

    #     def normalizeData(data, minval, maxval):
    #         return (data - minval) / (maxval - minval)

    #     model_stg2.eval()
    #     with torch.no_grad():
    #         ref_dataloader = torch.utils.data.DataLoader(
    #             train_data,
    #             batch_size=math.floor(len(train_data) * 0.1),
    #             shuffle=True,
    #             num_workers=4,
    #             drop_last=False,
    #         )
    #         for imgs, label in ref_dataloader:
    #             imgs = imgs.to(device)
    #             ref_features = model_stg2(
    #                 imgs, get_ref_features=True, args=args
    #             )  # [10%, 512, 8, 8]
    #             break  # we just need the first 10%
    #         debug_heatmap_folder = f"./debug_heatmaps_{args.subdataset}_{timestamp}/"
    #         os.makedirs(debug_heatmap_folder)

    #         # logic anomaly heatmap
    #         for each_logicano in logicano_fixed:
    #             logicano_image = each_logicano["image"]
    #             logicano_image = logicano_image.to(device)

    #             _, _, map_logic_logicano = predict(
    #                 logicano_image, model_stg2, ref_features, args
    #             )
    #             map_logic_logicano = F.interpolate(
    #                 map_logic_logicano, (orig_height, orig_width), mode="bilinear"
    #             )
    #             map_logic_logicano = map_logic_logicano[0, 0].cpu().numpy()
    #             pred_mask_logicano = np.uint8(
    #                 normalizeData(
    #                     map_logic_logicano,
    #                     np.min(map_logic_logicano),
    #                     np.max(map_logic_logicano),
    #                 )
    #                 * 255
    #             )
    #             heatmap_logicano = cv2.applyColorMap(
    #                 pred_mask_logicano, cv2.COLORMAP_JET
    #             )
    #             path_name = each_logicano["img_path"][0]
    #             raw_img_logicano = np.array(cv2.imread(path_name, cv2.IMREAD_COLOR))
    #             overlay_logicano = (
    #                 heatmap_logicano * heatmap_alpha
    #                 + raw_img_logicano * (1.0 - heatmap_alpha)
    #             )
    #             cv2.imwrite(
    #                 os.path.join(
    #                     debug_heatmap_folder,
    #                     f"{args.subdataset}_logicano_{os.path.basename(path_name).split('.png')[0]}_iter_{args.iters_stg2}_{args.loss_mode}.jpg",
    #                 ),
    #                 overlay_logicano,
    #             )
    #         for each_normal in normal_fixed:
    #             normal_image, path_name = each_normal
    #             normal_image = normal_image.to(device)

    #             # normal image heatmap
    #             _, _, map_logic_normal = predict(
    #                 normal_image, model_stg2, ref_features, args
    #             )
    #             map_logic_normal = F.interpolate(
    #                 map_logic_normal, (orig_height, orig_width), mode="bilinear"
    #             )
    #             map_logic_normal = map_logic_normal[0, 0].cpu().numpy()
    #             pred_mask_normal = np.uint8(
    #                 normalizeData(
    #                     map_logic_normal,
    #                     np.min(map_logic_normal),
    #                     np.max(map_logic_normal),
    #                 )
    #                 * 255
    #             )
    #             heatmap_normal = cv2.applyColorMap(pred_mask_normal, cv2.COLORMAP_JET)

    #             raw_img_normal = np.array(
    #                 cv2.imread(
    #                     path_name[0],
    #                     cv2.IMREAD_COLOR,
    #                 )
    #             )
    #             overlay_normal = heatmap_normal * heatmap_alpha + raw_img_normal * (
    #                 1.0 - heatmap_alpha
    #             )

    #             cv2.imwrite(
    #                 os.path.join(
    #                     debug_heatmap_folder,
    #                     f"{args.subdataset}_normal_{os.path.basename(path_name[0]).split('.png')[0]}_iter_{args.iters_stg2}_{args.loss_mode}.jpg",
    #                 ),
    #                 overlay_normal,
    #             )

    #     exit()

    """
    --[EVALUATION]--:
    use validation set
    """
    model_stg2.eval()
    with torch.no_grad():
        # obtain ref features from train set using p%
        # if args.debug_mode_2:
        #     train_data = ImageFolderWithPath(
        #         root=train_path, transform=transform_data(args.image_size)
        #     )

        if not args.fixed_ref:
            ref_dataloader = torch.utils.data.DataLoader(
                train_data,
                batch_size=math.floor(len(train_data) * 0.1),
                shuffle=True,
                num_workers=4,
                drop_last=False,
            )
            for imgs, label in ref_dataloader:
                imgs = imgs.to(device)
                ref_features = model_stg2(
                    imgs, get_ref_features=True, args=args
                )  # [10%, 512, 8, 8]
                # if args.debug_mode_2:
                #     ref_path = list(label)
                break  # we just need the first 10%

        # if args.debug_mode_2:
        #     q_structure_start = torch.tensor(0.1)
        #     q_structure_end = torch.tensor(0.9)
        #     q_logic_start = torch.tensor(0.1)
        #     q_logic_end = torch.tensor(0.9)
        # else:

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
    # if args.debug_mode_2:
    #     closest_inspect_folder = (
    #         f"closest_{timestamp}_{subdataset_mapper[args.subdataset]}"
    #     )
    #     os.makedirs(closest_inspect_folder)
    model_stg2.eval()
    with torch.no_grad():
        for raw_image, path in test_data:
            # path: 'datasets/loco/breakfast_box/test/good/000.png'
            orig_width = raw_image.width
            orig_height = raw_image.height
            image = transform_data(args.image_size)(raw_image)
            image = image.unsqueeze(0)
            image = image.to(device)  # [bs, 3, 256, 256]

            # if args.debug_mode_2:
            #     max_index = predict(
            #         image,
            #         model_stg2,
            #         ref_features,
            #         args,
            #         q_structure_start=q_structure_start,
            #         q_structure_end=q_structure_end,
            #         q_logic_start=q_logic_start,
            #         q_logic_end=q_logic_end,
            #     )
            #     test_img = Image.open(path)
            #     test_img = test_img.resize((256, 256))
            #     path_divided = path.split("/")
            #     ano_type = path_divided[-2]
            #     img_index = path_divided[-1].split(".png")[0]
            #     test_img.save(
            #         os.path.join(closest_inspect_folder, f"{ano_type}_{img_index}.png"),
            #         "PNG",
            #     )

            #     ref_img_path = ref_path[max_index]
            #     ref_img = Image.open(ref_img_path)
            #     ref_img = ref_img.resize((256, 256))
            #     ref_path_divided = ref_img_path.split("/")
            #     ref_index = ref_path_divided[-1].split(".png")[0]
            #     ref_img.save(
            #         os.path.join(
            #             closest_inspect_folder,
            #             f"{ano_type}_{img_index}_ref_{ref_index}.png",
            #         ),
            #         "PNG",
            #     )

            # else:
            map_combined, _, _ = predict(
                image,
                model_stg2,
                ref_features,
                args,
                q_structure_start=q_structure_start,
                q_structure_end=q_structure_end,
                q_logic_start=q_logic_start,
                q_logic_end=q_logic_end,
            )

            map_combined = F.interpolate(
                map_combined, (orig_height, orig_width), mode="bilinear"
            )
            map_combined = (
                map_combined[0, 0].cpu().numpy()
            )  # ready to be saved into .tiff format

            defect_class = os.path.basename(os.path.dirname(path))

            if test_output_dir is not None:
                img_nm = os.path.split(path)[1].split(".")[0]
                if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_dir, defect_class))
                file = os.path.join(test_output_dir, defect_class, img_nm + ".tiff")
                tifffile.imwrite(file, map_combined)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seeds", type=int, default=[42], nargs="+")
    parser.add_argument("--batch_size_stg1", type=int, default=16)
    parser.add_argument(
        "--logicano_select",
        type=str,
        choices=["percent", "absolute"],
        default="absolute",
    )
    parser.add_argument(
        "--percent_logicano",
        type=float,
        default=0.1,
        help="proportion of real logical anomalies used in training",
    )
    parser.add_argument(
        "--num_logicano",
        type=int,
        default=10,
        help="number of real logical anomalies used in training",
    )

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--iters_stg1", type=int, default=3000)
    parser.add_argument("--iters_stg2", type=int, default=3000)
    parser.add_argument("--lr_stg2", type=float, default=0.0002)
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
    parser.add_argument("--stg2_ckpt", type=str)
    parser.add_argument(
        "--logicano_only",
        action="store_true",
        help="if true, then only use one logical anomaly during stg2 training",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="if true, then enter debug_mode",
    )
    parser.add_argument(
        "--debug_mode_2",
        action="store_true",
        help="if true, then enter debug_mode_2",
    )
    parser.add_argument(
        "--debug_mode_3",
        action="store_true",
        help="for Curtis",
    )
    parser.add_argument("--debug_logicano_count", type=int, default=5)
    parser.add_argument(
        "--debug_normal_count",
        type=str,
        choices=["all", "follow"],
        default="all",
        help="if all, use all normal images; if follow, use same number as debug_logicano_count",
    )
    parser.add_argument(
        "--loss_mode",
        default="extreme",
        choices=["extreme", "average"],
        help="decides whether to use min or mean with ref to calculate loss",
    )
    parser.add_argument("--attn_count", type=int, default=4)
    parser.add_argument(
        "--attn_in_deconv",
        action="store_true",
        help="if true, then use attention in deconv module",
    )
    parser.add_argument(
        "--fixed_ref",
        action="store_true",
        help="if true, then use 10 percent of train images as fixed ref throughout training/validation/eval",
    )
    parser.add_argument("--fixed_ref_percent", type=float, default=0.1)
    parser.add_argument(
        "--similarity_priority",
        type=str,
        choices=["flatten", "pointwise"],
        default="flatten",
        help="how to calcualte similarity between ref and input",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "SGD"],
        default="Adam",
        help="choice of optimizer",
    )
    parser.add_argument(
        "--compress_bn",
        action="store_true",
        help="if true, then continue compress bottleneck feature to spatial dim of 1",
    )

    args = parser.parse_args()
    args.output_dir = args.output_dir + f"_[{subdataset_mapper[args.subdataset]}]"

    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in args.seeds:
        train(args, seed)
