from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        m.eval()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratio=1,  # to reduce spatial dim before K,V multiplication
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))

        return x


class DeConv(nn.Module):
    def __init__(
        self,
        attn_in_deconv=False,
        in_dim=4096,
        out_dim=2,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.attn_in_deconv = attn_in_deconv

        self.dec_1 = nn.ConvTranspose2d(
            in_channels=in_dim, out_channels=512, kernel_size=2, stride=2
        )
        self.dec_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=64, kernel_size=4, stride=4
        )
        self.dec_3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=8, kernel_size=2, stride=2
        )
        self.dec_4 = nn.ConvTranspose2d(
            in_channels=8, out_channels=out_dim, kernel_size=2, stride=2
        )
        if self.attn_in_deconv:
            self.attn_module_1 = nn.ModuleList(
                [SelfAttentionBlock(dim=512) for j in range(4)]
            )
            self.attn_module_2 = nn.ModuleList(
                [SelfAttentionBlock(dim=64) for j in range(4)]
            )

    def forward(self, x):
        x = self.dec_1(x)
        x = self.relu(x)
        if self.attn_in_deconv:
            # pass through attention module
            B, C, H, W = x.shape  # ( 2 256 16 16)
            x = x.reshape(B, C, -1)
            x = x.permute(0, 2, 1)
            for blk in self.attn_module_1:
                x = blk(x, H, W)
            x = x.permute(0, 1, 2)
            x = x.reshape(B, C, H, W)
        else:
            x = self.dropout(x)

        x = self.dec_2(x)
        x = self.relu(x)
        if self.attn_in_deconv:
            # pass through attention module
            B, C, H, W = x.shape  # ( 2 32 64 64 )
            x = x.reshape(B, C, -1)
            x = x.permute(0, 2, 1)
            for blk in self.attn_module_2:
                x = blk(x, H, W)
            x = x.permute(0, 1, 2)
            x = x.reshape(B, C, H, W)
        else:
            x = self.dropout(x)

        x = self.dec_3(x)
        x = self.relu(x)

        x = self.dec_4(x)
        return x


class LogicalMaskProducer(nn.Module):
    def __init__(
        self,
        model_stg1,
        logicano_only=False,
        loss_mode="extreme",
        attn_count=4,
        attn_in_deconv=False,
    ) -> None:
        super().__init__()
        # choices
        self.logicano_only = logicano_only
        self.loss_mode = loss_mode
        self.attn_count = attn_count
        self.attn_in_deconv = attn_in_deconv

        # from stg1
        self.model_stg1 = model_stg1

        # from stg2
        self.channel_reducer = nn.Linear(in_features=2048, out_features=512)
        self.self_att_module = nn.ModuleList(
            [SelfAttentionBlock(dim=4096) for j in range(self.attn_count)]
        )
        self.deconv = DeConv(attn_in_deconv=self.attn_in_deconv)

        # prevent gradients
        for param in self.model_stg1.parameters():
            param.requires_grad = False

    def forward(self, x, get_ref_features=False, ref_features=None, args=None):
        # x.shape: [bs, 3, 256, 256]

        if self.training:
            """
            train mode
            """
            # extract features with pretrained stg1 model
            with torch.no_grad():
                x = self.model_stg1.encoder(x)
                x = self.model_stg1.bottleneck(x)  # [bs, 2048, 8, 8]

            if self.loss_mode == "extreme":
                if args.fixed_ref:
                    refs = ref_features
                else:
                    refs = x[:-2]  # [bs-2, 2048, 8, 8]

                logicano = x[-2]
                normal = x[-1]

                num_ref = refs.shape[0]
                max_logicano_sim = -1000
                max_logicano_index = None
                max_normal_sim = -1000
                max_normal_index = None
                for i in range(num_ref):
                    ref = refs[i]

                    if args.similarity_priority == "pointwise":
                        logicano_sim = F.cosine_similarity(ref, logicano, dim=0).mean()
                    elif args.similarity_priority == "flatten":
                        logicano_sim = F.cosine_similarity(
                            torch.mean(ref, dim=(1, 2)),
                            torch.mean(logicano, dim=(1, 2)),
                            dim=0,
                        )
                    else:
                        raise Exception("Unimplemented similarity_priority")
                    if logicano_sim > max_logicano_sim:
                        max_logicano_sim = logicano_sim
                        max_logicano_index = i

                    if args.similarity_priority == "pointwise":
                        normal_sim = F.cosine_similarity(ref, normal, dim=0).mean()
                    elif args.similarity_priority == "flatten":
                        normal_sim = F.cosine_similarity(
                            torch.mean(ref, dim=(1, 2)),
                            torch.mean(normal, dim=(1, 2)),
                            dim=0,
                        )
                    else:
                        raise Exception("Unimplemented similarity_priority")

                    if normal_sim > max_normal_sim:
                        max_normal_sim = normal_sim
                        max_normal_index = i

                # create two new inputs, and upsample them
                logicano_input = torch.cat(
                    [refs[max_logicano_index], logicano]
                )  # shape: [4096, 8, 8]
                normal_input = torch.cat(
                    [refs[max_normal_index], normal]
                )  # shape: [4096, 8, 8]

                x = torch.stack(
                    [logicano_input, normal_input], dim=0
                )  # shape: [2, 4096, 8, 8]

                # pass through attention module
                B, C, H, W = x.shape
                x = x.reshape(B, C, -1)
                x = x.permute(0, 2, 1)
                for blk in self.self_att_module:
                    x = blk(x, H, W)
                x = x.permute(0, 1, 2)
                x = x.reshape(B, C, H, W)  # [2, 4096, 8, 8]

                x = self.deconv(x)
                x = torch.softmax(x, dim=1)
                return x
            elif self.loss_mode == "average":
                raise Exception("Haven't implemented loss_mode==average yet")
                # refs = x[:-2]  # [bs-2, 512, 8, 8]
                # logicano = x[-2]
                # normal = x[-1]
                # num_ref = refs.shape[0]

                # logicano_input = [torch.cat([ref, logicano]) for ref in refs]
                # logicano_input = torch.stack(
                #     logicano_input, dim=0
                # )  # shape: [8, 1024, 8, 8]
                # normal_input = [torch.cat([ref, normal]) for ref in refs]
                # normal_input = torch.stack(
                #     normal_input, dim=0
                # )  # shape: [8, 1024, 8, 8]
                # intermediate_input = torch.cat(
                #     [logicano_input, normal_input], dim=0
                # )  # shape: [16, 1024, 8, 8]
                # output = self.deconv(intermediate_input)
                # output_logicano = output[:num_ref]
                # output_logicano = output_logicano.mean(dim=0)  # (2, 256, 256)
                # output_normal = output[num_ref:]
                # output_normal = output_normal.mean(dim=0)  # (2, 256, 256)
                # output = torch.stack([output_logicano, output_normal], dim=0)
                # output = torch.softmax(output, dim=1)
                # return output

            # # reduce channel dimension
            # x = x.permute(0, 2, 3, 1)
            # x = self.channel_reducer(x)
            # x = x.permute(0, 3, 1, 2)  # [bs, 512, 8, 8]

            # # pass through attention module
            # B, C, H, W = x.shape
            # x = x.reshape(B, C, -1)
            # x = x.permute(0, 2, 1)
            # for blk in self.self_att_module:
            #     x = blk(x, H, W)
            # x = x.permute(0, 1, 2)
            # x = x.reshape(B, C, H, W)  # [bs, 512, 8, 8]

            # if self.loss_mode == "extreme":
            #     if self.logicano_only:
            #         refs = x[:-1]  # [bs-1, 512, 8, 8]
            #         logicano = x[-1]
            #         num_ref = refs.shape[0]
            #         max_logicano_sim = -1000
            #         max_logicano_index = None
            #         for i in range(num_ref):
            #             ref = refs[i]
            #             # logicano_sim = F.cosine_similarity(ref, logicano, dim=0).mean()
            #             logicano_sim = F.cosine_similarity(
            #                 torch.mean(ref, dim=(1, 2)),
            #                 torch.mean(logicano, dim=(1, 2)),
            #                 dim=0,
            #             )
            #             if logicano_sim > max_logicano_sim:
            #                 max_logicano_sim = logicano_sim
            #                 max_logicano_index = i
            #         logicano_input = torch.cat([refs[max_logicano_index], logicano])
            #         intermediate_input = logicano_input.unsqueeze(0)
            #     else:
            #         refs = x[:-2]  # [bs-2, 512, 8, 8]
            #         logicano = x[-2]
            #         normal = x[-1]
            #         num_ref = refs.shape[0]
            #         max_logicano_sim = -1000
            #         max_logicano_index = None
            #         max_normal_sim = -1000
            #         max_normal_index = None
            #         for i in range(num_ref):
            #             ref = refs[i]
            #             # logicano_sim = F.cosine_similarity(ref, logicano, dim=0).mean()
            #             logicano_sim = F.cosine_similarity(
            #                 torch.mean(ref, dim=(1, 2)),
            #                 torch.mean(logicano, dim=(1, 2)),
            #                 dim=0,
            #             )
            #             if logicano_sim > max_logicano_sim:
            #                 max_logicano_sim = logicano_sim
            #                 max_logicano_index = i
            #             # normal_sim = F.cosine_similarity(ref, normal, dim=0).mean()
            #             normal_sim = F.cosine_similarity(
            #                 torch.mean(ref, dim=(1, 2)),
            #                 torch.mean(normal, dim=(1, 2)),
            #                 dim=0,
            #             )

            #             if normal_sim > max_normal_sim:
            #                 max_normal_sim = normal_sim
            #                 max_normal_index = i

            #         # create two new inputs, and upsample them
            #         logicano_input = torch.cat([refs[max_logicano_index], logicano])
            #         normal_input = torch.cat([refs[max_normal_index], normal])
            #         intermediate_input = torch.stack(
            #             [logicano_input, normal_input], dim=0
            #         )
            #     output = self.deconv(
            #         intermediate_input
            #     )  # [1or2, 2, 256, 256], (1) logical_ano, (2) normal

            #     output = torch.softmax(output, dim=1)
            #     return output
            # elif self.loss_mode == "average":
            #     if self.logicano_only:
            #         refs = x[:-1]  # [bs-1, 512, 8, 8]
            #         logicano = x[-1]
            #         logicano_input = [torch.cat([ref, logicano]) for ref in refs]
            #         intermediate_input = torch.stack(
            #             logicano_input, dim=0
            #         )  # shape: [8, 1024, 8, 8]
            #         output = self.deconv(intermediate_input)
            #         output = output.mean(dim=0)
            #         output = output.unsqueeze(0)
            #         return output
            #     else:
            #         refs = x[:-2]  # [bs-2, 512, 8, 8]
            #         logicano = x[-2]
            #         normal = x[-1]
            #         num_ref = refs.shape[0]

            #         logicano_input = [torch.cat([ref, logicano]) for ref in refs]
            #         logicano_input = torch.stack(
            #             logicano_input, dim=0
            #         )  # shape: [8, 1024, 8, 8]
            #         normal_input = [torch.cat([ref, normal]) for ref in refs]
            #         normal_input = torch.stack(
            #             normal_input, dim=0
            #         )  # shape: [8, 1024, 8, 8]
            #         intermediate_input = torch.cat(
            #             [logicano_input, normal_input], dim=0
            #         )  # shape: [16, 1024, 8, 8]
            #         output = self.deconv(intermediate_input)
            #         output_logicano = output[:num_ref]
            #         output_logicano = output_logicano.mean(dim=0)  # (2, 256, 256)
            #         output_normal = output[num_ref:]
            #         output_normal = output_normal.mean(dim=0)  # (2, 256, 256)
            #         output = torch.stack([output_logicano, output_normal], dim=0)
            #         output = torch.softmax(output, dim=1)
            #         return output
        else:
            """
            eval mode
            """
            if get_ref_features:  # to obtain 10% ref features from train set
                # extract features with pretrained stg1 model
                with torch.no_grad():
                    x = self.model_stg1.encoder(x)
                    x = self.model_stg1.bottleneck(x)  # [bs, 2048, 8, 8]
                    return x

                    # # reduce channel dimension
                    # x = x.permute(0, 2, 3, 1)
                    # x = self.channel_reducer(x)
                    # x = x.permute(0, 3, 1, 2)

                    # # pass through attention module
                    # B, C, H, W = x.shape
                    # x = x.reshape(B, C, -1)
                    # x = x.permute(0, 2, 1)
                    # for blk in self.self_att_module:
                    #     x = blk(x, H, W)
                    # x = x.permute(0, 1, 2)
                    # x = x.reshape(B, C, H, W)  # [10%, 512, 8, 8]
                    # return x
            else:  # eval on each test image
                with torch.no_grad():
                    assert ref_features is not None, "ref_features should not be None"
                    # structural branch
                    stg1_en, stg1_de = self.model_stg1(x)

                    # logical branch
                    x = self.model_stg1.encoder(x)
                    x = self.model_stg1.bottleneck(x)  # [bs, 2048, 8, 8], bs==1 ??

                    if self.loss_mode == "extreme":
                        # find closest ref
                        num_ref = ref_features.shape[0]
                        max_sim = -1000
                        max_index = None

                        for i in range(num_ref):
                            ref = ref_features[i]
                            if args.similarity_priority == "pointwise":
                                sim = F.cosine_similarity(ref, x[0], dim=0).mean()
                            elif args.similarity_priority == "flatten":
                                sim = F.cosine_similarity(
                                    torch.mean(ref, dim=(1, 2)),
                                    torch.mean(x[0], dim=(1, 2)),
                                    dim=0,
                                )
                            else:
                                raise Exception("Unimplemented similarity_priority")
                            if sim > max_sim:
                                max_sim = sim
                                max_index = i

                        x = torch.cat([ref_features[max_index], x[0]]).unsqueeze(
                            0
                        )  # [1, 4096, 8, 8]

                        # pass through attention module
                        B, C, H, W = x.shape
                        x = x.reshape(B, C, -1)
                        x = x.permute(0, 2, 1)
                        for blk in self.self_att_module:
                            x = blk(x, H, W)
                        x = x.permute(0, 1, 2)
                        x = x.reshape(B, C, H, W)  # [1, 4096, 8, 8]

                        x = self.deconv(x)
                        x = torch.softmax(x, dim=1)

                        return (stg1_en, stg1_de, x)
                    elif self.loss_mode == "average":
                        raise Exception("Haven't implemented loss_mode==average yet")
                        # intermediate_input = [
                        #     torch.cat([ref, x[0]]) for ref in ref_features
                        # ]
                        # intermediate_input = torch.stack(
                        #     intermediate_input, dim=0
                        # )  # shape: [10%, 1024, 8, 8]
                        # pred_mask = self.deconv(intermediate_input)
                        # pred_mask = pred_mask.mean(dim=0)
                        # pred_mask = pred_mask.unsqueeze(0)
                        # pred_mask = torch.softmax(pred_mask, dim=1)
                        # return (
                        #     stg1_en,
                        #     stg1_de,
                        #     pred_mask,
                        # )

    def train(self, mode=True):
        self.training = mode
        self.model_stg1.apply(set_bn_eval)
        if mode is True:
            self.model_stg1.train(False)
            self.channel_reducer.train(True)
            self.self_att_module.train(True)
            self.deconv.train(True)
        else:
            self.model_stg1.train(False)
            self.channel_reducer.train(False)
            self.self_att_module.train(False)
            self.deconv.train(False)
        return self


class ReContrast(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_freeze,
        bottleneck,
        decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(
            x
        )  # if input_size=256, then [[bs, 256, 64, 64], [bs, 512, 32, 32], [bs, 1024, 16, 16]]
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)

        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return (
            en_freeze + en,
            de,
        )  # de's first half is recons of en, second half is recons of en_freeze

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self
