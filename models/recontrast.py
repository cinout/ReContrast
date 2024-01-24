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
        compress_bn=False,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.attn_in_deconv = attn_in_deconv
        self.compress_bn = compress_bn

        if compress_bn:
            self.dec_1 = nn.ConvTranspose2d(
                in_channels=in_dim, out_channels=512, kernel_size=4, stride=4
            )
            self.dec_2 = nn.ConvTranspose2d(
                in_channels=512, out_channels=64, kernel_size=4, stride=4
            )
            self.dec_3 = nn.ConvTranspose2d(
                in_channels=64, out_channels=8, kernel_size=4, stride=4
            )
            self.dec_4 = nn.ConvTranspose2d(
                in_channels=8, out_channels=out_dim, kernel_size=4, stride=4
            )
        else:
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


class BN_Compressor(nn.Module):
    def __init__(self, in_dim=4096) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=2048, kernel_size=2, stride=2, padding=0
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=2048, out_channels=1024, kernel_size=2, stride=2, padding=0
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class LogicalMaskProducer(nn.Module):
    def __init__(
        self,
        model_stg1,
        args,
    ) -> None:
        super().__init__()
        # choices
        self.logicano_only = args.logicano_only
        self.loss_mode = args.loss_mode
        self.attn_count = args.attn_count
        self.attn_in_deconv = args.attn_in_deconv
        self.compress_bn = args.compress_bn

        # from stg1
        self.model_stg1 = model_stg1

        # from stg2
        # self.channel_reducer = nn.Linear(in_features=2048, out_features=512)

        if self.compress_bn:
            self.bn_compressor = BN_Compressor(in_dim=4096)
        else:
            self.self_att_module = nn.ModuleList(
                [SelfAttentionBlock(dim=4096) for j in range(self.attn_count)]
            )

        self.deconv = DeConv(
            attn_in_deconv=self.attn_in_deconv,
            in_dim=1024 if self.compress_bn else 4096,
            compress_bn=self.compress_bn,
        )

        # prevent gradients
        for param in self.model_stg1.parameters():
            param.requires_grad = False

    def forward(self, x, get_ref_features=False, ref_features=None, args=None):
        if self.training:
            """
            train mode
            """

            if self.loss_mode == "extreme":
                if args.fixed_ref:
                    pass
                else:
                    # extract features with pretrained stg1 model
                    with torch.no_grad():
                        x = self.model_stg1.encoder(x)
                        x = self.model_stg1.bottleneck(x)  # [bs, 2048, 8, 8]
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
                            logicano_sim = F.cosine_similarity(
                                ref, logicano, dim=0
                            ).mean()
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

                if self.compress_bn:
                    x = self.bn_compressor(x)  # [2, 4096, 1, 1]
                else:
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
            else:
                raise Exception("Haven't implemented loss_mode==average yet")

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

            else:  # eval on each test image
                with torch.no_grad():
                    # structural branch
                    stg1_en, stg1_de = self.model_stg1(x)

                    # logical branch
                    x = self.model_stg1.encoder(x)
                    x = self.model_stg1.bottleneck(x)  # [bs, 2048, 8, 8], bs==1 ??

                    if args.debug_mode_3:
                        return x

                    assert ref_features is not None, "ref_features should not be None"
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

                        if self.compress_bn:
                            x = self.bn_compressor(x)  # [2, 4096, 1, 1]
                        else:
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
                    else:
                        raise Exception("Haven't implemented loss_mode==average yet")

    def train(self, mode=True):
        self.training = mode
        self.model_stg1.apply(set_bn_eval)
        if mode is True:
            self.model_stg1.train(False)
            # self.channel_reducer.train(True)
            if self.compress_bn:
                self.bn_compressor.train(True)
            else:
                self.self_att_module.train(True)
            self.deconv.train(True)
        else:
            self.model_stg1.train(False)
            # self.channel_reducer.train(False)
            if self.compress_bn:
                self.bn_compressor.train(False)
            else:
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
