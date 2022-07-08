import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from utils import ALLDataset
import cv2
from MEANet import build_model
from backbones.VGG import VGGNet

def test(weight_path, net, test_dataloader, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.load_state_dict(torch.load(weight_path))

    net.eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        prev_time = datetime.now()
        print(prev_time)
        for index, (allfocus,depth,fs,names) in enumerate(test_dataloader):

            basize, dime, height, width = fs.size()
            inputs_focal = fs.view(1, basize, dime, height, width).transpose(0, 1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, 12, dim=2), dim=1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, basize, dim=0), dim=1).squeeze()

            allfocus = allfocus.to(device)
            depth = depth.to(device)
            inputs_focal = inputs_focal.to(device)
            sal_final, sal_coarse, edge = net(allfocus, inputs_focal, depth)
            name = names[0]
            print(name)
            pre_sal = torch.sigmoid(sal_final)
            pre_sal = pre_sal.permute(1, 0, 2, 3)[0]
            pre_sal = pre_sal.cpu().detach().numpy().copy()
            pre_sal = pre_sal[0]
            pre_sal = pre_sal * 255
            cv2.imwrite(save_path + name.split('.')[0]+'.png', pre_sal)
        cur_time = datetime.now()
        print(cur_time)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    rgb_model = VGGNet(requires_grad=True)
    fs_model = VGGNet(requires_grad=True)
    depth_model = VGGNet(requires_grad=True)
    MEANet = build_model(rgb_model=rgb_model, fs_model=fs_model, depth_model=depth_model)

    data_path = 'dataset/test/'
    weight_name = 'MEANet'

    weight_path = 'trained_weight/' + weight_name + '.pth'
    save_path = 'results/' + weight_name + '/'
    dataset = ['DUTLF', 'LFSD', 'HFUT', 'Lytro']
    for data in dataset:
        test_dataloader = DataLoader(
            ALLDataset(transform=False, location=data_path + data + '/', crop=False, train=False), batch_size=1,
            shuffle=False)
        test(weight_path, MEANet, test_dataloader, save_path + data + '/')