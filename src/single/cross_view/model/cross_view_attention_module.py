import torch
import torch.nn as nn


class BiLateralAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(BiLateralAttentionModule, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=(in_channels, 1, 1))
        self.max_pool = nn.MaxPool3d(kernel_size=(in_channels, 1, 1))

    def forward(self, FLp, FRp):
        AP_FLp = self.avg_pool(FLp)
        MP_FLp = self.max_pool(FLp)
        AP_FRp = self.avg_pool(FRp)
        MP_FRp = self.max_pool(FRp)

        concat = torch.cat([AP_FLp, MP_FLp, AP_FRp, MP_FRp], dim=1)
        attention_map = self.conv(concat)

        return attention_map


class BiProjectionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(BiProjectionAttentionModule, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, FsCC, FsMLO):
        AP_FsCC = self.avg_pool(FsCC)
        MP_FsCC = self.max_pool(FsCC)
        AP_FsMLO = self.avg_pool(FsMLO)
        MP_FsMLO = self.max_pool(FsMLO)

        concat = torch.cat([AP_FsCC, MP_FsCC, AP_FsMLO, MP_FsMLO], dim=1)
        attention_map = self.mlp(concat.view(concat.size()[0], -1)).unsqueeze(-1).unsqueeze(-1)

        return attention_map


class CrossViewAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(CrossViewAttentionModule, self).__init__()

        self.bi_lateral = BiLateralAttentionModule(in_channels)
        self.bi_projection = BiProjectionAttentionModule(in_channels)

    def forward(self, FLCC, FRCC, FLMLO, FRMLO):
        cc_attention_array = self.bi_lateral(FLCC, FRCC)
        mlo_attention_array = self.bi_lateral(FLMLO, FRMLO)
        left_attention_vector = self.bi_projection(FLCC, FLMLO)
        right_attention_vector = self.bi_projection(FRCC, FRMLO)

        FLCC_out = (1 + cc_attention_array) * (1 + left_attention_vector) * FLCC
        FRCC_out = (1 + cc_attention_array) * (1 + right_attention_vector) * FRCC
        FLMLO_out = (1 + mlo_attention_array) * (1 + left_attention_vector) * FLMLO
        FRMLO_out = (1 + mlo_attention_array) * (1 + right_attention_vector) * FRMLO

        return FLCC_out, FRCC_out, FLMLO_out, FRMLO_out


if __name__ == "__main__":
    FLCC_input = torch.rand(4, 16, 100, 80)
    FRCC_input = torch.rand(4, 16, 100, 80)
    FLMLO_input = torch.rand(4, 16, 100, 80)
    FRMLO_input = torch.rand(4, 16, 100, 80)

    model = CrossViewAttentionModule(in_channels=16)
    FLCC_out, FRCC_out, FLMLO_out, FRMLO_out = model(FLCC_input, FRCC_input, FLMLO_input, FRMLO_input)

    assert FLCC_out.shape == FLCC_input.shape
    assert FRCC_out.shape == FRCC_input.shape
    assert FLMLO_out.shape == FLMLO_input.shape
    assert FRMLO_out.shape == FRMLO_input.shape
