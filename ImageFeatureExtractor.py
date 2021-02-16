import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
from tqdm import tqdm
import json
import os
import sys
from torchsummary import summary
import torch.nn.functional as F

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.net = resnet50(pretrained=True).cuda()
        # summary(net, (3,224,224))
        # sys.exit()
        # net = list(net.children())[:-1]
        self.cnn = nn.Sequential(
                                # nn.Conv2d(256, 512, 1), 
                                # nn.AvgPool2d(2,2), 
                                nn.Conv2d(512, 1024, 1), 
                                nn.AvgPool2d(2,2), 
                                nn.Conv2d(1024, 2048, 1), 
                                nn.AvgPool2d(2,2)).cuda()
        # freeze cnn
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, img):
        feature = self.net.conv1(img)
        feature = self.net.bn1(feature)
        feature = self.net.relu(feature)
        feature = self.net.maxpool(feature)

        feature = self.net.layer1(feature)
        feature = self.net.layer2(feature)
        feature = self.cnn(feature)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        return feature

if __name__ == '__main__':
    cnn = ImageFeatureExtractor()
    cnn.to("cuda:0")
    # for target bbox
    targ_bbox = json.load(open('./data/pfn-picbbox.json'))
    for image in tqdm(targ_bbox):
        file_name = image['file_path'] + '_' + str(image['targ_id'])
        tmpimg = np.load(os.path.join('./data/target_bbox/', file_name +'.npy'))
        tmpimg = tmpimg.reshape([1, 3, 224, 224])
        tmpimg = torch.from_numpy(tmpimg.copy()).to("cuda:0")
        feat = cnn(tmpimg)
        device2 = torch.device('cpu')
        feat = feat.to(device2)
        feat = feat.reshape([1, 2048]).detach().numpy()
        # print(feat.shape)
        np.save("./data/target_bbox_feat_layer2_avg/" + file_name, feat)
    
    # for des
    input = []
    with open('data/pfn-pic/labels/ort.jsonl', 'r') as fin:
        for line in fin:
            input.append(json.loads(line))
    des = ['_tl', '_tr', '_bl', '_br']
    for image in tqdm(input):
        for i in des:
            file_name = image['image_file'] + i
            tmpimg = np.load(os.path.join('./data/des_bbox/', file_name +'.npy'))
            tmpimg = tmpimg.reshape([1, 3, 224, 224])
            tmpimg = torch.from_numpy(tmpimg.copy()).to("cuda:0")
            feat = cnn(tmpimg)
            device2 = torch.device('cpu')
            feat = feat.to(device2)
            feat = feat.reshape([1, 2048]).detach().numpy()
            # print(feat.shape)
            np.save("./data/des_bbox_feat_layer2_avg/" + file_name, feat)

    # for context bboxes
    # input = []
    # with open('data/pfn-pic/labels/ort.jsonl', 'r') as fin:
    #     for line in fin:
    #         input.append(json.loads(line))
    # for image in tqdm(input):
    #     file_name = image['image_file']
    #     bbox_file = './data/context_bboxes/' + file_name +'.npy'
    #     bboxes = np.load(bbox_file)
    #     bbox_att = []
    #     # print(bboxes.shape)  
    #     for bbox in bboxes:
    #         tmpimg = bbox.reshape([1, 3, 224, 224])
    #         tmpimg = torch.from_numpy(tmpimg.copy().astype(np.float32))
    #         feat = cnn(tmpimg)
    #         device2 = torch.device('cpu')
    #         feat = feat.to(device2)
    #         feat = feat.reshape([1, 2048])
    #         feat = np.array(feat)
    #         bbox_att.append(feat)
    #     # print(feat.shape)
    #     # print(feats.shape)
    #     np.save("./data/pfn-picbu_att_resnet/" + file_name, bbox_att)