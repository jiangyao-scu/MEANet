from backbones.VGG import VGGNet
from MEANet import build_model
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from utils import ALLDataset
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os


def train(net, train_dataloader, device, optimizer, train_sampler, writer, epo_num=50, local_rank=0,
          weight_save_path='weights/'):


    if local_rank == 0:
        if not os.path.exists(weight_save_path):
            os.makedirs(weight_save_path)

        num_params = 0
        for p in net.parameters():
            num_params += p.numel()
        print(net)
        print("The number of parameters: {}".format(num_params))
        prev_time = datetime.now()

    optimizer.zero_grad()
    BATCH_SIZE = 2
    aveGrad = 0
    net.train()
    r_sal_loss = 0

    for epo in range(epo_num):
        train_sampler.set_epoch(epo)
        for index, (allfocus, depth, fs, GT, contour, names) in enumerate(train_dataloader):
            basize, dime, height, width = fs.size()
            inputs_focal = fs.view(1, basize, dime, height, width).transpose(0, 1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, 12, dim=2), dim=1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, basize, dim=0), dim=1).squeeze()

            sal_label_coarse = F.interpolate(GT, (16, 16), mode='bilinear', align_corners=True)
            sal_label_coarse = torch.cat((sal_label_coarse, sal_label_coarse, sal_label_coarse), dim=0)

            allfocus = allfocus.to(device)
            inputs_focal = inputs_focal.to(device)
            depth = depth.to(device)
            GT = GT.to(device)
            contour = contour.to(device)
            sal_label_coarse = sal_label_coarse.to(device)

            sal_final, sal_coarse, edge = net(allfocus=allfocus, inputs_focal=inputs_focal, depth=depth)

            loss = 0
            loss = loss + F.binary_cross_entropy_with_logits(sal_final, GT, reduction='sum') + \
                   256 * F.binary_cross_entropy_with_logits(sal_coarse, sal_label_coarse, reduction='sum')
            for i in range(len(edge)):
                loss = loss + F.binary_cross_entropy_with_logits(edge[i], contour, reduction='sum')


            sal_loss = loss / BATCH_SIZE
            r_sal_loss += sal_loss.data
            sal_loss.backward()

            aveGrad += 1
            if aveGrad % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            if local_rank == 0:
                if np.mod(index, 10) == 0:
                    print('epoch {}, {}/{}, lr={}, train loss is {}'.
                          format(epo, index, len(train_dataloader), optimizer.param_groups[0]['lr'], sal_loss))
                    writer.add_scalar('training loss', r_sal_loss * BATCH_SIZE / 10,
                                      epo * len(train_dataloader) + index)
                    r_sal_loss = 0

        if local_rank == 0:
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            prev_time = cur_time
            print('%s' % (time_str))

            if epo < 40 and np.mod(epo, 10) == 9:
                torch.save(net.module.state_dict(), weight_save_path + 'Cross_model_{}.pth'.format(epo))
                print(weight_save_path + 'Cross_model_{}.pth'.format(epo))
            elif epo >= 40:
                torch.save(net.module.state_dict(), weight_save_path + 'Cross_model_{}.pth'.format(epo))
                print(weight_save_path + 'Cross_model_{}.pth'.format(epo))


def main(train_data_path, weight_save_path, log_path, lr):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    writer = SummaryWriter(log_path)

    data = ALLDataset(transform=False, location=train_data_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    train_dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, sampler=train_sampler)

    rgb_model = VGGNet(requires_grad=True)
    fs_model = VGGNet(requires_grad=True)
    depth_model = VGGNet(requires_grad=True)
    MEANet = build_model(rgb_model=rgb_model, fs_model=fs_model, depth_model=depth_model)
    MEANet.to(device)
    MEANet = torch.nn.parallel.DistributedDataParallel(MEANet, device_ids=[local_rank],
                                                       output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(MEANet.parameters(), lr=lr)

    train(MEANet, train_dataloader, device, optimizer, epo_num=50, local_rank=local_rank,
          weight_save_path=weight_save_path, train_sampler=train_sampler, writer=writer)


if __name__ == '__main__':
    train_data_path = 'xxx/'
    weight_save_path = 'MEANet/'
    log_path = 'log/MEANet'
    lr = 1e-4

    main(train_data_path, weight_save_path, log_path, lr)
