import torch
from torch import nn
import torch.nn.functional as F



k = 64

class slicesattention(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.conv_w = nn.Conv2d(in_channel * 12, 12, 3, padding=1)  # 12 = N (N=12)
        self.conv_w1 = nn.Conv2d(12, 12, 3, padding=1)
        self.conv_w2 = nn.Conv2d(12, 12, 3, padding=1)
        self.pool_avg_w = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=False)
        self.ConvLSTM = nn.Conv2d(in_channel * 12, out_channel, kernel_size=3, padding=1)


    def forward(self,x):
        input = x
        fs_features =torch.cat(torch.chunk(input,12,dim=0),dim=1)
        weight = self.conv_w(fs_features)
        weight = self.conv_w1(weight)
        weight = self.conv_w2(weight)

        # att
        weight = self.pool_avg_w(weight)
        weight =torch.mul(F.softmax(weight,dim=1),12)
        weight = weight.transpose(0,1)
        res = torch.mul(input,weight)
        res = torch.cat(torch.chunk(res,12,dim=0),dim=1)
        res = self.ConvLSTM(res)

        return res

class Encoder(nn.Module):
    def __init__(self, rgb_model,fs_model,depth_model):
        super(Encoder, self).__init__()
        self.rgb_backbone = rgb_model
        self.depth_backbone = depth_model
        self.fs_backbone = fs_model
        self.relu = nn.ReLU(inplace=True)

        slices = []
        slices.append(slicesattention(64, 64))
        slices.append(slicesattention(128, 128))
        slices.append(slicesattention(256, 256))
        slices.append(slicesattention(512, 512))
        slices.append(slicesattention(512, 512))
        self.Slices = nn.ModuleList(slices)

        rgb_cp=[]
        rgb_cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        rgb_cp.append(nn.Sequential(nn.Conv2d(128, 128, 5, 1, 2), self.relu, nn.Conv2d(128, 128, 5, 1, 2), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        rgb_cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        rgb_cp.append(nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        rgb_cp.append(nn.Sequential(nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.rgb_CP = nn.ModuleList(rgb_cp)

        fs_cp = []
        fs_cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        fs_cp.append(nn.Sequential(nn.Conv2d(128, 128, 5, 1, 2), self.relu, nn.Conv2d(128, 128, 5, 1, 2), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        fs_cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        fs_cp.append(nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        fs_cp.append(nn.Sequential(nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.fs_CP = nn.ModuleList(fs_cp)

        depth_cp = []
        depth_cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        depth_cp.append(nn.Sequential(nn.Conv2d(128, 128, 5, 1, 2), self.relu, nn.Conv2d(128, 128, 5, 1, 2), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        depth_cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        depth_cp.append(nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        depth_cp.append(nn.Sequential(nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.depth_CP = nn.ModuleList(depth_cp)

    def forward(self, input_rgb,input_fs,input_depth):
        feature_extract = []

        rgb = self.rgb_backbone(input_rgb)
        depth = self.depth_backbone(input_depth)
        fs = self.fs_backbone(input_fs)
        # x={}
        for i in range(5):
            fs['x'+str(i+1)] = self.Slices[i](fs['x'+str(i+1)])
            rgb['x'+str(i+1)] = self.rgb_CP[i](rgb['x'+str(i+1)])
            fs['x' + str(i + 1)] = self.fs_CP[i](fs['x' + str(i + 1)])
            depth['x' + str(i + 1)] = self.depth_CP[i](depth['x' + str(i + 1)])
        for i in range(5):
            feature_extract.append(torch.cat((rgb['x'+str(i+1)],fs['x'+str(i+1)],depth['x'+str(i+1)]),dim=0))
        return feature_extract  # list of tensor that compress model output

class spatial_channel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.relu = nn.ReLU(inplace=False)
        # spatial attention
        self.conv_sp1 = nn.Conv2d(out_channel, int(out_channel/4), 3, padding=1)
        self.conv_sp2 = nn.Conv2d(int(out_channel/4), int(out_channel/4), 3, padding=1)
        self.conv_sp4 = nn.Conv2d(int(out_channel/4),out_channel,3,padding=1)

        # channel attention
        self.Average_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = nn.Conv2d(in_channel,int(in_channel/3),kernel_size=1)
        self.conv_fc2 = nn.Conv2d(int(in_channel/3),in_channel,kernel_size=1)

        self.dowm = nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        self.bn_dowm = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        input = x


        # channel attention
        channel_weight = self.Average_pooling(input)
        channel_weight1 = self.relu(self.conv_fc1(channel_weight))
        channel_weight2 = F.sigmoid(self.conv_fc2(channel_weight1))
        channel_att = input+input*channel_weight2
        channel_att = self.relu(self.bn_dowm(self.dowm(channel_att)))

        # spatial attention
        spatial_weight = self.relu(self.conv_sp1(channel_att))
        spatial_weight = self.relu(self.conv_sp2(spatial_weight))
        spatial_weight = F.sigmoid(self.conv_sp4(spatial_weight))
        spatial_att = channel_att + channel_att * spatial_weight

        return spatial_att

class cross_fusion(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.fs_sc = spatial_channel(in_channel,in_channel)
        self.rgb_sc = spatial_channel(in_channel, in_channel)
        self.depth_sc = spatial_channel(in_channel, in_channel)

        self.fuse_rgb = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn_fuse_rgb = nn.BatchNorm2d(out_channel)

        self.fuse_depth = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn_fuse_depth = nn.BatchNorm2d(out_channel)


        self.fuse = nn.Conv2d(3*in_channel,out_channel,kernel_size=3,padding=1)
        self.bn_fuse = nn.BatchNorm2d(out_channel)


    def forward(self,rgb_feature,fs_feature,depth_feature):
        depth_att_features = self.depth_sc(depth_feature)
        rgb_att_features = self.rgb_sc(rgb_feature)

        fs_feature = fs_feature*(depth_att_features+rgb_att_features)+fs_feature
        fs_feature = self.fs_sc(fs_feature)

        fused_rgb = self.relu(self.bn_fuse_rgb(self.fuse_rgb(rgb_att_features*fs_feature+rgb_att_features)))
        fused_depth = self.relu(self.bn_fuse_depth(self.fuse_depth(depth_att_features * fs_feature + depth_att_features)))

        return torch.cat((fused_rgb,fs_feature,fused_depth),dim=0)


class CMLayer(nn.Module):
    def __init__(self):
        super(CMLayer, self).__init__()
        attetions = []
        attetions.append(cross_fusion(k, k))
        attetions.append(cross_fusion(k, k))
        attetions.append(cross_fusion(k, k))
        attetions.append(cross_fusion(k, k))
        attetions.append(cross_fusion(k, k))
        self.attention = nn.ModuleList(attetions)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            temp_rgb, temp_fs, temp_depth = torch.split(list_x[i], [1, 1, 1], dim=0)
            fused = self.attention[i](temp_rgb, temp_fs, temp_depth)
            resl.append(fused)
        return resl


class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

class BConv3(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,padding):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=kernel_size,padding=padding)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=kernel_size,padding=padding)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, kernel_size=kernel_size,padding=padding)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self,x):
        input = x
        out = self.relu(self.bn1(self.conv1(input)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out

class edge_fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.cat = BConv3(3 * k, k, 3, 1)
        self.con_and = nn.Conv2d(k, k, kernel_size=1)
        self.fuse = BConv3(2 * k, k, 3, 1)

    def forward(self,x):
        input = x
        edge_rgb, edge_fs, edge_depth = torch.split(input, [1, 1, 1], dim=0)
        and_edge = self.con_and(edge_rgb * edge_fs * edge_depth)
        cat_edge = self.cat(torch.cat((edge_rgb, edge_fs, edge_depth), dim=1))
        edge = self.fuse(torch.cat((and_edge, cat_edge), dim=1))
        return edge

class edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

        self.edge_extract_1 = BConv3(2 * k, k, 3, 1)
        self.edge_extract_2 = BConv3(2 * k, k, 3, 1)
        self.edge_extract_3 = BConv3(2 * k, k, 3, 1)
        self.edge_extract_4 = BConv3(2 * k, k, 3, 1)

        self.edge_fuse_1 = edge_fuse()
        self.edge_fuse_2 = edge_fuse()
        self.edge_fuse_3 = edge_fuse()
        self.edge_fuse_4 = edge_fuse()

        self.edge_score_1 = nn.Conv2d(k, 1, 1, 1)
        self.edge_score_2 = nn.Conv2d(k, 1, 1, 1)
        self.edge_score_3 = nn.Conv2d(k, 1, 1, 1)
        self.edge_score_4 = nn.Conv2d(k, 1, 1, 1)

        self.sal_fuse = BConv3(3 * k, k, 3, 1)
        self.up_1 = BConv3(2 * k, k, 5, 2)
        self.up_2 = BConv3(2 * k, k, 5, 2)
        self.up_3 = BConv3(2 * k, k, 5, 2)
        self.up_4 = BConv3(2 * k, k, 5, 2)

    def forward(self, edge, region):
        o_region = region
        region = F.interpolate(region, scale_factor=2, mode='bilinear', align_corners=True)
        edge_features_4 = self.edge_extract_4(torch.cat((edge[3],region),dim=1))
        region = F.interpolate(region, scale_factor=2, mode='bilinear', align_corners=True)
        edge_features_3 = self.edge_extract_3(torch.cat((edge[2],region),dim=1))
        region = F.interpolate(region, scale_factor=2, mode='bilinear', align_corners=True)
        edge_features_2 = self.edge_extract_2(torch.cat((edge[1],region),dim=1))
        region = F.interpolate(region, scale_factor=2, mode='bilinear', align_corners=True)
        edge_features_1 = self.edge_extract_1(torch.cat((edge[0],region),dim=1))

        edge_features_1 = self.edge_fuse_1(edge_features_1)
        edge_features_2 = self.edge_fuse_2(edge_features_2)
        edge_features_3 = self.edge_fuse_3(edge_features_3)
        edge_features_4 = self.edge_fuse_4(edge_features_4)


        edge_1 = self.edge_score_1(edge_features_1)
        edge_2 = self.edge_score_2(edge_features_2)
        edge_3 = self.edge_score_3(edge_features_3)
        edge_4 = self.edge_score_4(edge_features_4)
        final_edge = []
        final_edge.append(F.interpolate(edge_1, (256, 256), mode='bilinear', align_corners=True))
        final_edge.append(F.interpolate(edge_2, (256, 256), mode='bilinear', align_corners=True))
        final_edge.append(F.interpolate(edge_3, (256, 256), mode='bilinear', align_corners=True))
        final_edge.append(F.interpolate(edge_4, (256, 256), mode='bilinear', align_corners=True))


        final_region = self.sal_fuse(torch.cat((torch.split(o_region,[1,1,1],dim=0)),dim=1))
        final_region = F.interpolate(final_region, scale_factor=2, mode='bilinear', align_corners=True)
        final_region = self.up_1(torch.cat((final_region,edge_features_4),dim=1))
        final_region = F.interpolate(final_region, scale_factor=2, mode='bilinear', align_corners=True)
        final_region = self.up_2(torch.cat((final_region, edge_features_3), dim=1))
        final_region = F.interpolate(final_region, scale_factor=2, mode='bilinear', align_corners=True)
        final_region = self.up_3(torch.cat((final_region, edge_features_2), dim=1))
        final_region = F.interpolate(final_region, scale_factor=2, mode='bilinear', align_corners=True)
        final_region = self.up_4(torch.cat((final_region, edge_features_1), dim=1))

        return final_region,final_edge

class CrossEdge(nn.Module):
    def __init__(self, Encoder, cm_layers, side_edge, coarse_score_layers, final_score_layers):
        super(CrossEdge, self).__init__()
        self.Encoder = Encoder
        self.score_coarse = coarse_score_layers
        self.score_final = final_score_layers
        self.cm = cm_layers
        self.side_edge = side_edge

    def forward(self, allfocus,inputs_focal,depth):
        x = self.Encoder(input_rgb=allfocus, input_fs=inputs_focal, input_depth=depth)
        x_cm = self.cm(x)
        s_coarse = self.score_coarse(x[4])
        sal_features,edge = self.side_edge(x_cm[0:4], x_cm[4])
        s_final = self.score_final(sal_features)


        return s_final,s_coarse,edge


def build_model(rgb_model, fs_model, depth_model):
    return CrossEdge(Encoder(rgb_model, fs_model, depth_model), CMLayer(), edge(), ScoreLayer(k), ScoreLayer(k))

