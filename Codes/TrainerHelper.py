"""
working_directory
       |____experiment_name
                |____models
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable

def dice_loss_3d():
    pass

class Trainer(object):
    """This code was probably adapted from:
    https://github.com/AruniRC/resnet-face-pytorch/blob/master/train.py"""
    def __init__(self,
                 cuda,
                 model,
                 model_pth=None,  # FILEPATH_MODEL_LOAD
                 optimizer=None,
                 train_loader=None,
                 test_loader=None,  # valid_loader
                 lmk_num=None,
                 train_root_dir=None,  # project_dir where 'models' dir is.
                 out=None,  # working directory > experiment name
                 max_epoch=None,
                 batch_size=None,
                 size_average=False,
                 interval_validate=None,
                 compete=False,
                 onlyEval=False):
        self.cuda = cuda
        self.model = model
        self.model_pth = model_pth
        self.optim = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lmk_num = lmk_num
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.train_root_dir = train_root_dir
        self.out = out
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        self.timestamp_start = time.time()
        # fixme: what are these?
        self.interval_validate = interval_validate
        self.size_average = size_average
        self.epoch = 0  # epoch to start train / start_epoch
        self.iteration = 0
        self.best_mean_iu = 0
        self.model_save_criteria = 0
        # if os.path.exists(self.model_pth):  # if FILEPATH_MODEL_LOAD is not None:
        #     if self.cuda:
        #         # self.model.load_state_dict(torch.load(self.model_pth))
        #         checkpoint = torch.load(self.model_pth)  # checkpoint in pytorch terminology
        #         self.model.load_state_dict(checkpoint['checkpoint_latest']['model_state_dict'])
        #         self.optim.load_state_dict(checkpoint['checkpoint_latest']['optimizer_state_dict'])
        #         checkpoint_best = checkpoint['best_model_state']
        #         # loss_valid_min = train_states_best['loss_valid_min']
        #         model_save_criteria = checkpoint_best['model_save_criteria']
        #     else:
        #         # self.model.load_state_dict(torch.load(model_pth))
        #         self.model.load_state_dict(torch.load(self.model_pth, map_location=lambda storage, location: storage))
        # else:
        #     checkpoint = {}
        #     model_save_criteria = 0  # np.inf

    def train(self):
        self.model.train()
        out = os.path.join(self.out, 'visualization')
        if not os.path.exists(out):
            os.makedirs(out)
        log_file = os.path.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')
        for batch_idx, (data, target, sub_name, sub_resource) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=100, leave=False):  # epoch will be updated by train_epoch()
            # print(sub_name[0])
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred = self.model(data)
            self.optim.zero_grad()  # remove existing/previous gradient
            if sub_resource[0] == 0:
                loss, loss_organ = dice_loss_3d(pred, target, sub_resource)
                print('epoch=%d, batch_idx=%d, loss=%.6f \n' % (self.epoch, batch_idx, loss.data))
                print('epoch=%d, batch_idx=%d, dice_oragan=%.6f' % (self.epoch, batch_idx, loss_organ.cpu().data.numpy()))
                fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data))
                loss.backward()
                self.optim.step()
            elif sub_resource[0] == 1:
                loss, loss_liver = dice_loss_3d(pred, target, sub_resource)
                print('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))
                print('epoch=%d, batch_idx=%d, dice_liver=%.6f' % (
                self.epoch, batch_idx, loss_liver.cpu().data.numpy()[0]))
                fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))
                loss.backward()
                self.optim.step()
            elif sub_resource[0] == 2:
                loss, loss_pancreas = dice_loss_3d(pred, target, sub_resource)
                print('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))
                print('epoch=%d, batch_idx=%d, dice_pancreas=%.6f' % (
                self.epoch, batch_idx, loss_pancreas.cpu().data.numpy()[0]))
                fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))
                loss.backward()
                self.optim.step()
            elif sub_resource[0] == 3:
                loss, loss_spleen = dice_loss_3d(pred, target, sub_resource)
                print('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))
                print('epoch=%d, batch_idx=%d, dice_spleen=%.6f' % (
                self.epoch, batch_idx, loss_spleen.cpu().data.numpy()[0]))
                fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))
                loss.backward()
                self.optim.step()
        # loss.backward()
        # self.optim.step()
        # fixme: where is the loss, mean loss
        fv.close()

    def validate(self, checkpoint, model_save_criteria):
        self.model.eval()  # train
        out = os.path.join(self.out, 'seg_output')
        # out_vis = os.path.join(self.out, 'visualization')
        # results_epoch_dir = os.path.join(out, 'epoch_%04d' % self.epoch)
        # if not os.path.exists(results_epoch_dir):
        #     os.makedirs(results_epoch_dir)
        result_dice_file = os.path.join(out, '{}_dice_result_list.txt'.format(self.epoch))
        dice_coe_list = open(result_dice_file, 'w')
        with torch.no_grad():
            for batch_idx, (data, target, sub_name, sub_resource) in tqdm.tqdm(
                    # enumerate(self.test_loader), total=len(self.test_loader),
                    enumerate(self.test_loader), total=len(self.test_loader),
                    desc='Valid epoch=%d' % self.epoch, ncols=100, leave=False):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target, volatile=True)
                pred = self.model(data)
                ### This section for computing the dice coefficients for testing
                loss, loss_organ = dice_loss_3d(pred, target, sub_resource)
                print('epoch=%d, batch_idx=%d, dice_organ=%.6f' % (self.epoch, batch_idx, loss_organ.cpu().data.numpy()))
                # write to file
                dice_coe_list.write(str(loss_organ.cpu().data.numpy()) + '\n')
        chosen_criteria = loss  # fixme: chosen_criteria
        # Saving the best performing model
        if chosen_criteria < model_save_criteria:
            model_save_criteria = chosen_criteria
            checkpoint_best = {
                'epoch': self.epoch + 1,
                'arch': self.model.__class__.__name__,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'model_save_criteria': model_save_criteria
            }
            checkpoint['best_model_state'] = checkpoint_best
        dice_coe_list.close()

    def train_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train', ncols=100):
            self.epoch = epoch      # increments the epoch of Trainer
            out = os.path.join(self.out, 'models')
            if not os.path.exists(out):
                os.makedirs(out)
            # model_pth = #'%s/model_epoch_%04d.pth' % (out, epoch)
            if os.path.exists(self.model_pth):      # if FILEPATH_MODEL_LOAD is not None:
                if self.cuda:
                    checkpoint = torch.load(self.model_pth)     # checkpoint in pytorch terminology
                    self.model.load_state_dict(checkpoint['checkpoint_latest']['model_state_dict'])
                    self.optim.load_state_dict(checkpoint['checkpoint_latest']['optimizer_state_dict'])
                    checkpoint_best = checkpoint['best_model_state']
                    # loss_valid_min = train_states_best['loss_valid_min']
                    self.model_save_criteria = checkpoint_best['model_save_criteria']
                else:
                    checkpoint = torch.load(self.model_pth, map_location=lambda storage, location: storage)
                    self.model.load_state_dict(checkpoint['checkpoint_latest']['model_state_dict'])
                    self.optim.load_state_dict(checkpoint['checkpoint_latest']['optimizer_state_dict'])
                    checkpoint_best = checkpoint['best_model_state']
                    # loss_valid_min = train_states_best['loss_valid_min']
                    self.model_save_criteria = checkpoint_best['model_save_criteria']
                    # https://pytorch.org/docs/stable/generated/torch.load.html
                # if epoch % 5 == 0:
                # self.validate()
            else:
                checkpoint = {}
                model_save_criteria = self.model_save_criteria
                self.train()
                if epoch % 1 == 0:
                    self.validate(checkpoint, model_save_criteria)
                torch.save(self.model.state_dict(), self.model_pth)

    def test_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Test', ncols=100):
            self.epoch = epoch
            train_root_dir = os.path.join(self.train_root_dir, 'models')
            # model_pth = '%s/model_epoch_%04d.pth' % (train_root_dir, epoch)
            if self.cuda:
                self.model.load_state_dict(torch.load(self.model_pth))
            else:
                # self.model.load_state_dict(torch.load(model_pth))
                self.model.load_state_dict(torch.load(self.model_pth, map_location=lambda storage, location: storage))
            if epoch % 1 == 0:
                self.validate_liver()


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

