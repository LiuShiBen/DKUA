
import PIL.Image as Image
import time
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.my_tools import *
import numpy as np
from torch.nn import functional as F
from scipy.linalg import sqrtm
from einops import rearrange

class Trainer:
    def __init__(self, args, model, optimizer, num_classes,
                 data_loader_train, training_phase, add_num=0, margin=0.0):
        self.num_task_experts = args.num_task_experts  #5
        self.model = model
        self.model.cuda()
        self.data_loader_train = data_loader_train
        self.training_phase = training_phase
        self.add_num = add_num
        self.gamma = 0.5
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.T = 2
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.train_iters = len(self.data_loader_train)
        self.optimizer = optimizer
        self.epoch = args.epochs
        self.cov_common = []
        self.cov_mean = []


    def train(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_base = AverageMeter()
        losses_kl = AverageMeter()
        losses_sm = AverageMeter()
        losses_af = AverageMeter()

        end = time.time()
        self.model.train()
        #print("training_step:", training_stepping)
        if self.training_phase == 1:
            for i in range(len(self.data_loader_train)):
                train_inputs = self.data_loader_train.next()
                data_time.update(time.time() - end)
                imgs, targets, cids, domains = self._parse_data(train_inputs)
                # print("imgs:", imgs.shape, targets.shape)
                targets += self.add_num
                # Current network output
                cls_out, com_feat, _, _ = self.model(imgs, self.training_phase)
                loss_ce = self.CE_loss(cls_out, targets)
                loss_tp = self.Hard_loss(com_feat, targets)
                loss = loss_ce + loss_tp

                #loss = tp_task + ce_task
                losses_base.update(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Loss_base {:.3f} ({:.3f})\t'
                          .format(epoch, i + 1, self.train_iters,
                                  batch_time.val, batch_time.avg,
                                  losses_base.val, losses_base.avg))
        if 1 < self.training_phase <= 5:
            feat_task_all = []
            with torch.no_grad():
                for i in range(len(self.data_loader_train)):
                    train_inputs = self.data_loader_train.next()
                    data_time.update(time.time() - end)
                    imgs, targets, cids, domains = self._parse_data(train_inputs)
                    _, com_feat, _, task_out = self.model(imgs, self.training_phase)
                    # print("feat_task:", feat_task[0].shape)
                    feat_task_all.append(task_out)
            # feat = torch.stack(feat_task1, dim=0)
            feat_task_all[0].shape
            feat = torch.stack(feat_task_all, dim=0)
            feat = rearrange(feat, 'a b c d -> (a b) c d')
            #print("feat:", feat.shape)
            self.cov_common, self.cov_mean = self.update_common_covariance(feat, self.training_phase)

            for i in range(len(self.data_loader_train)):
                train_inputs = self.data_loader_train.next()
                data_time.update(time.time() - end)
                imgs, targets, cids, domains = self._parse_data(train_inputs)
                # print("imgs:", imgs.shape, targets.shape)
                targets += self.add_num
                # Current network output
                cls_out, com_feat, current_feat, task_out = self.model(imgs, self.training_phase)
                # corss-entroy loss of new samples
                loss_ce = self.CE_loss(cls_out, targets)
                # triplet loss of new samples
                loss_tp = self.Hard_loss(com_feat, targets)
                loss_sm = self.cal_similarity(current_feat[0], self.cov_common)
                loss_af = self.cal_aff(task_out, com_feat[0], self.training_phase)
                loss_kl = self.kl_loss(task_out, self.training_phase)

                loss = loss_ce + loss_tp  + loss_kl + loss_sm + loss_af # + (loss_i2t + loss_t2i) / 2
                losses_sm.update(loss_sm)
                losses_af.update(loss_af)
                losses_kl.update(loss_kl)
                losses_base.update(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Loss_base {:.3f} ({:.3f})\t'
                          'Loss_KL {:.3f} ({:.3f})\t'
                          .format(epoch, i + 1, self.train_iters,
                                  batch_time.val, batch_time.avg,
                                  losses_base.val, losses_base.avg,
                                  losses_kl.val, losses_kl.avg)
        #print("sucessful.....", self.training_phase)

    def update_common_covariance(self, comom_feat, training_step):
        """
        if training_step == 2:
            task1_feat = comom_feat[:, :, 0].to("cuda")
            task2_feat = comom_feat[:, :, 1].to("cuda")

            task1_cov = torch.cov(task1_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task1_mean = task1_feat.mean(dim=0)

            task2_cov = torch.cov(task2_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task2_mean = task2_feat.mean(dim=0)

            weight_old_1 = 1 / 2
            weight_new_2 = 1 / 2
            common_cov = task1_cov * weight_old_1 + task2_cov * weight_new_2
            common_mean = task1_mean * weight_new_2 + task2_mean * weight_new_2

            return common_cov, common_mean

        elif training_step == 3:
            task1_feat = comom_feat[:, :, 0].to("cuda")
            task2_feat = comom_feat[:, :, 1].to("cuda")
            task3_feat = comom_feat[:, :, 2].to("cuda")

            task1_cov = torch.cov(task1_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task1_mean = task1_feat.mean(dim=0)

            task2_cov = torch.cov(task2_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task2_mean = task2_feat.mean(dim=0)

            task3_cov = torch.cov(task3_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task3_mean = task3_feat.mean(dim=0)

            weight_old_1 = 1/2
            weight_new_2 = 1/2
            common_cov_12 = task1_cov * weight_old_1 + task2_cov * weight_new_2
            common_mean_12 = task1_mean * weight_old_1 + task2_mean * weight_new_2

            weight_old_12 = 2/3
            weight_new_3 = 1/3
            common_cov = common_cov_12 * weight_old_12 + task3_cov * weight_new_3
            common_mean = common_mean_12 * weight_old_12 + task3_mean * weight_new_3

            return common_cov, common_mean

        elif training_step == 4:
            task1_feat = comom_feat[:, :, 0].to("cuda")
            task2_feat = comom_feat[:, :, 1].to("cuda")
            task3_feat = comom_feat[:, :, 2].to("cuda")
            task4_feat = comom_feat[:, :, 3].to("cuda")

            task1_cov = torch.cov(task1_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task1_mean = task1_feat.mean(dim=0)

            task2_cov = torch.cov(task2_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task2_mean = task2_feat.mean(dim=0)

            task3_cov = torch.cov(task3_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task3_mean = task3_feat.mean(dim=0)

            task4_cov = torch.cov(task4_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task4_mean = task4_feat.mean(dim=0)

            weight_old_1 = 1 / 2
            weight_new_2 = 1 / 2
            common_cov_12 = task1_cov * weight_old_1 + task2_cov * weight_new_2
            common_mean_12 = task1_mean * weight_old_1 + task2_mean * weight_new_2

            weight_old_12 = 2 / 3
            weight_new_3 = 1 / 3
            common_cov_123 = common_cov_12 * weight_old_12 + task3_cov * weight_new_3
            common_mean_123 = common_mean_12 * weight_old_12 + task3_mean * weight_new_3

            weight_old_123 = 3 / 4
            weight_new_4 = 1 / 4
            common_cov = common_cov_123 * weight_old_123 + task4_cov * weight_new_4
            common_mean = common_mean_123 * weight_old_123 + task4_mean * weight_new_4

            return common_cov, common_mean

        elif training_step == 5:
            task1_feat = comom_feat[:, :, 0].to("cuda")
            task2_feat = comom_feat[:, :, 1].to("cuda")
            task3_feat = comom_feat[:, :, 2].to("cuda")
            task4_feat = comom_feat[:, :, 3].to("cuda")
            task5_feat = comom_feat[:, :, 4].to("cuda")

            task1_cov = torch.cov(task1_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task1_mean = task1_feat.mean(dim=0)

            task2_cov = torch.cov(task2_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task2_mean = task2_feat.mean(dim=0)

            task3_cov = torch.cov(task3_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task3_mean = task3_feat.mean(dim=0)

            task4_cov = torch.cov(task4_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task4_mean = task4_feat.mean(dim=0)

            task5_cov = torch.cov(task5_feat.T)  # PyTorch的cov输入需是 (n_features, n_samples)
            task5_mean = task5_feat.mean(dim=0)

            weight_old_1 = 1 / 2
            weight_new_2 = 1 / 2
            common_cov_12 = task1_cov * weight_old_1 + task2_cov * weight_new_2
            common_mean_12 = task1_mean * weight_old_1 + task2_mean * weight_new_2

            weight_old_12 = 2 / 3
            weight_new_3 = 1 / 3
            common_cov_123 = common_cov_12 * weight_old_12 + task3_cov * weight_new_3
            common_mean_123 = common_mean_12 * weight_old_12 + task3_mean * weight_new_3

            weight_old_123 = 3 / 4
            weight_new_4 = 1 / 4
            common_cov_1234 = common_cov_123 * weight_old_123 + task4_cov * weight_new_4
            common_mean_1234 = common_mean_123 * weight_old_123 + task4_mean * weight_new_4

            weight_old_1234 = 4 / 5
            weight_new_5 = 1 / 5
            common_cov = common_cov_1234 * weight_old_1234 + task5_cov * weight_new_5
            common_mean = common_mean_1234 * weight_old_1234 + task5_mean * weight_new_5

            return common_cov, common_mean

    def update_covariance(self, dataload, old_cov=None, traing_phase=1):
        feat_task_all = []
        # self.model.eval()
        with torch.no_grad():
            for i in range(len(dataload)):
                train_inputs = self.data_loader_train.next()
                imgs, targets, cids, domains = self._parse_data(train_inputs)
                cls_base, cls_task, feat_base, feat_task = self.model(imgs, self.training_phase)
                # print("feat_task:", feat_task[0].shape)
                feat_task_all.append(feat_task[0])
        new_features = torch.stack(feat_task_all, dim=0).reshape(-1, 384)

        new_features = new_features.to('cuda')
        new_cov = torch.cov(new_features.T)  # PyTorch的cov输入需是 (n_features, n_samples)

        if traing_phase == 1:
            common_cov = new_cov
        if traing_phase > 1:

            weight_old = 1.0 - (1.0 / traing_phase)
            weight_new = 1.0 / traing_phase
            common_cov = old_cov * weight_old + new_cov * weight_new
        return common_cov

    def cal_similarity(self, x, cov):
        new_features = x.cuda()
        new_cov = torch.cov(new_features.T)  # PyTorch的cov输入需是 (n_features, n_samples)
        new_cov = new_cov @ new_cov.T + 1e-6 * torch.eye(384, device='cuda')  # 强制正定

        old_cov = cov.cuda()
        old_cov = old_cov @ old_cov.T + 1e-6 * torch.eye(384, device='cuda')  # 强制正定

        new_cov_norm = F.softmax(new_cov / 0.1, dim=1)
        old_cov_norm = F.softmax(old_cov / 0.1, dim=1)

        # divergence += self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)
        new_cov_norm_log = torch.log(new_cov_norm)
        loss = self.KLDivLoss(new_cov_norm_log, old_cov_norm)
        return loss

    def cal_aff(self, task_feat, com_feat, training_step):
        loss = []

        if training_step == 2:
            task1_feat = task_feat[:, :, 0].to("cuda")
            task2_feat = task_feat[:, :, 1].to("cuda")

            task1_to_common = self.cosine_similarity(task1_feat, com_feat)
            task1_to_common = F.softmax(task1_to_common / 0.1, dim=1)
            task1_to_common = self.get_normal_affinity(task1_to_common)

            task2_to_common = self.cosine_similarity(task2_feat, com_feat)
            task2_to_common = F.softmax(task2_to_common / 0.1, dim=1)
            task2_to_common = self.get_normal_affinity(task2_to_common)
            task2_to_common_log = torch.log(task2_to_common)
            loss = self.KLDivLoss(task2_to_common_log, task1_to_common)

        elif training_step == 3:
            task1_feat = task_feat[:, :, 0].to("cuda")
            task2_feat = task_feat[:, :, 1].to("cuda")
            task3_feat = task_feat[:, :, 2].to("cuda")

            task1_to_common = self.cosine_similarity(task1_feat, com_feat)
            task1_to_common = F.softmax(task1_to_common / 0.1, dim=1)
            task1_to_common = self.get_normal_affinity(task1_to_common)

            task2_to_common = self.cosine_similarity(task2_feat, com_feat)
            task2_to_common = F.softmax(task2_to_common / 0.1, dim=1)
            task2_to_common = self.get_normal_affinity(task2_to_common)

            task3_to_common = self.cosine_similarity(task3_feat, com_feat)
            task3_to_common = F.softmax(task3_to_common / 0.1, dim=1)
            task3_to_common = self.get_normal_affinity(task3_to_common)
            task3_to_common_log = torch.log(task3_to_common)

            loss31 = self.KLDivLoss(task3_to_common_log, task1_to_common)
            loss32 = self.KLDivLoss(task3_to_common_log, task2_to_common)

            #loss = (loss31 + loss32) / 2
            loss = loss31 + loss32

        elif training_step == 4:
            task1_feat = task_feat[:, :, 0].to("cuda")
            task2_feat = task_feat[:, :, 1].to("cuda")
            task3_feat = task_feat[:, :, 2].to("cuda")
            task4_feat = task_feat[:, :, 3].to("cuda")

            task1_to_common = self.cosine_similarity(task1_feat, com_feat)
            task1_to_common = F.softmax(task1_to_common / 0.1, dim=1)
            task1_to_common = self.get_normal_affinity(task1_to_common)

            task2_to_common = self.cosine_similarity(task2_feat, com_feat)
            task2_to_common = F.softmax(task2_to_common / 0.1, dim=1)
            task2_to_common = self.get_normal_affinity(task2_to_common)

            task3_to_common = self.cosine_similarity(task3_feat, com_feat)
            task3_to_common = F.softmax(task3_to_common / 0.1, dim=1)
            task3_to_common = self.get_normal_affinity(task3_to_common)

            task4_to_common = self.cosine_similarity(task4_feat, com_feat)
            task4_to_common = F.softmax(task4_to_common / 0.1, dim=1)
            task4_to_common = self.get_normal_affinity(task4_to_common)
            task4_to_common_log = torch.log(task4_to_common)

            loss41 = self.KLDivLoss(task4_to_common_log, task1_to_common)
            loss42 = self.KLDivLoss(task4_to_common_log, task2_to_common)
            loss43 = self.KLDivLoss(task4_to_common_log, task3_to_common)

            #loss = (loss41 + loss42 + loss43) / 3
            loss = loss41 + loss42 + loss43

        elif training_step == 5:
            task1_feat = task_feat[:, :, 0].to("cuda")
            task2_feat = task_feat[:, :, 1].to("cuda")
            task3_feat = task_feat[:, :, 2].to("cuda")
            task4_feat = task_feat[:, :, 3].to("cuda")
            task5_feat = task_feat[:, :, 4].to("cuda")

            task1_to_common = self.cosine_similarity(task1_feat, com_feat)
            task1_to_common = F.softmax(task1_to_common / 0.1, dim=1)
            task1_to_common = self.get_normal_affinity(task1_to_common)

            task2_to_common = self.cosine_similarity(task2_feat, com_feat)
            task2_to_common = F.softmax(task2_to_common / 0.1, dim=1)
            task2_to_common = self.get_normal_affinity(task2_to_common)

            task3_to_common = self.cosine_similarity(task3_feat, com_feat)
            task3_to_common = F.softmax(task3_to_common / 0.1, dim=1)
            task3_to_common = self.get_normal_affinity(task3_to_common)

            task4_to_common = self.cosine_similarity(task4_feat, com_feat)
            task4_to_common = F.softmax(task4_to_common / 0.1, dim=1)
            task4_to_common = self.get_normal_affinity(task4_to_common)

            task5_to_common = self.cosine_similarity(task5_feat, com_feat)
            task5_to_common = F.softmax(task5_to_common / 0.1, dim=1)
            task5_to_common = self.get_normal_affinity(task5_to_common)
            task5_to_common_log = torch.log(task5_to_common)

            loss51 = self.KLDivLoss(task5_to_common_log, task1_to_common)
            loss52 = self.KLDivLoss(task5_to_common_log, task2_to_common)
            loss53 = self.KLDivLoss(task5_to_common_log, task3_to_common)
            loss54 = self.KLDivLoss(task5_to_common_log, task4_to_common)

            #loss = (loss51 + loss52 + loss53 + loss54) / 4
            loss = loss51 + loss52 + loss53 + loss54

        else:
           pass
        return loss

    def cal_statistics(self, feature):
        B,C,K = feature.shape
        #print(B,C,K)
        features = feature[:,:,0]
        for i in range(1, K):
            #print(features.shape, feature[:,:,i].shape, i)
            features = torch.cat((features, feature[:,:,i]), dim=1)
        features = features.permute(1, 0).cpu()
        mean = torch.mean(features, dim = 0)
        sigma = torch.cov(features.T)
        return mean, sigma

    def kl_loss(self, task_feat, training_step):
        loss = []
        if training_step==2:
            task1_feat = task_feat[:, :, 0]
            task2_feat = task_feat[:, :, 1]
            loss1 = self.cosine_distance(task1_feat, task2_feat)
            # print("distance_matrix:", distance_matrix.shape)
            loss = torch.mean(loss1)

        elif training_step==3:
            task1_feat = task_feat[:, :, 0]
            task2_feat = task_feat[:, :, 1]
            task3_feat = task_feat[:, :, 2]

            loss31 = self.cosine_distance(task3_feat, task1_feat)
            loss32 = self.cosine_distance(task3_feat, task2_feat)
            loss = (torch.mean(loss31) + torch.mean(loss32)) / 2.0

        elif training_step==4:
            task1_feat = task_feat[:, :, 0]
            task2_feat = task_feat[:, :, 1]
            task3_feat = task_feat[:, :, 2]
            task4_feat = task_feat[:, :, 3]

            loss41 = self.cosine_distance(task4_feat, task1_feat)
            loss42 = self.cosine_distance(task4_feat, task2_feat)
            loss43 = self.cosine_distance(task4_feat, task3_feat)
            loss = (torch.mean(loss41) + torch.mean(loss42) + torch.mean(loss43)) / 3
        elif training_step==5:
            task1_feat = task_feat[:, :, 0]
            task2_feat = task_feat[:, :, 1]
            task3_feat = task_feat[:, :, 2]
            task4_feat = task_feat[:, :, 3]
            task5_feat = task_feat[:, :, 4]

            loss51 = self.cosine_distance(task5_feat, task1_feat)
            loss52 = self.cosine_distance(task5_feat, task2_feat)
            loss53 = self.cosine_distance(task5_feat, task3_feat)
            loss54 = self.cosine_distance(task5_feat, task4_feat)
            loss = (torch.mean(loss51) + torch.mean(loss52) + torch.mean(loss53) + torch.mean(loss54)) / 4
        else:
           pass
        return loss
 
    def get_normal_affinity(self,x, Norm=0.1):
        pre_matrix_origin=self.cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def CE_loss(self, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)  #ID loss
        return loss_ce

    def Tri_loss(self, s_features, targets):
        fea_loss = []
        for i in range(len(s_features)):
            loss_tr = self.criterion_triple(s_features[i], s_features[i], targets) #tri loss
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss)# / len(fea_loss)
        return loss_tr

    def Hard_loss(self, s_features, targets):
        fea_loss = []
        for i in range(0, len(s_features)):
            loss_tr = self.trip_hard(s_features[i], targets)[0]
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss)# / len(fea_loss)
        return loss_tr

    def cosine_distance(self, input1, input2):
        """Computes cosine distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat

    def cosine_similarity(self, input1, input2):
        """Computes cosine distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = torch.mm(input1_normed, input2_normed.t())
        return distmat
    




