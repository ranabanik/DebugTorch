import numpy as np
import os
from glob import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
from Utilities import get_sub_list, Subset, Seg_three_label, dice_loss_three
import matplotlib.pyplot as plt
import pickle
import nibabel as nib

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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
        running_loss = 0
        out = os.path.join(self.out, 'visualization')
        if not os.path.exists(out):
            os.makedirs(out)
        log_file = os.path.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')
        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=100, leave=False):
            # print(sub_name[0])
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred = self.model(data)
            self.optim.zero_grad()  # remove existing/previous gradient
            # if sub_resource[0] == 0:
            loss, dice, _, _, _ = dice_loss_three(pred, target) #, sub_resource)
            running_loss += loss.item()
            print('epoch=%d, batch_idx=%d, loss=%.6f \n' % (self.epoch, batch_idx, loss.data))
            print('epoch=%d, batch_idx=%d, dice_organ=%.6f' % (self.epoch, batch_idx, dice.cpu().data.numpy()))
            fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data))
            loss.backward()
            self.optim.step()
        train_loss = running_loss/len(self.train_loader)
        fv.close()
        return train_loss

    def validate(self, checkpoint, model_save_criteria):
        self.model.eval()  # train
        # out = os.path.join(self.out, 'seg_output')
        # out_vis = os.path.join(self.out, 'visualization')
        # results_epoch_dir = os.path.join(out, 'epoch_%04d' % self.epoch)
        # if not os.path.exists(results_epoch_dir):
        #     os.makedirs(results_epoch_dir)
        result_dice_file = os.path.join(out, '{}_dice_result_list.txt'.format(self.epoch))
        dice_coe_list = open(result_dice_file, 'w')
        with torch.no_grad():
            for batch_idx, (data, target, sub_name) in tqdm.tqdm(
                    # enumerate(self.test_loader), total=len(self.test_loader),
                    enumerate(self.test_loader), total=len(self.test_loader),
                    desc='Valid epoch=%d' % self.epoch, ncols=100, leave=False):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target, volatile=True)
                pred = self.model(data)
                ### This section for computing the dice coefficients for testing
                loss, dice, _, _, _ = dice_loss_three(pred, target) #, sub_resource)
                print('epoch=%d, batch_idx=%d, dice_organ=%.6f' % (self.epoch, batch_idx, dice.cpu().data.numpy()))
                # write to file
                dice_coe_list.write(str(dice.cpu().data.numpy()) + '\n')
        chosen_criteria = loss
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
            if self.model_pth is not None and epoch!=0:
                # #os.path.exists(self.model_pth):      # if FILEPATH_MODEL_LOAD is not None:
                # print("Happens")
                if self.cuda:
                    print("Happens")
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

class Trainer_v2(object):
    """
    initiates training and returns:
    └── TrainerTrial_2021-07-24-02-37-43
        ├── loss_acc.pickle
        ├── trainvalid_loss.txt
        └── UNet3D.pth
    +-----------------------------------------------------------------------------------+
    | >> A = Trainer_v2(                                                                |
    |           cuda=use_cuda, --> True(GPU) or False(CPU)                              |
    |           model=model,  --> any PyTorch nn.Module                                 |
    |           optimizer=opt,                                                          |
    |           train_loader=train_loader,                                              |
    |           valid_loader=valid_loader,                                              |
    |           trainer_root_dir=r'/media/banikr2/DATA/banikr_D_drive/model/OutTrainer',|
    |           max_epoch=10,                                                           |
    |           batch_size=2,                                                           |
    |           experiment_name='TrainerTrial') --> str that defines experiment or None |
    |           mode_criteria=0 --> for dice/accuracy. np.inf for any loss              |
    | >> A.train_epoch()                                                                |
    +-----------------------------------------------------------------------------------+
    This code was inspired from:
    https://github.com/AruniRC/resnet-face-pytorch/blob/master/train.py"""
    # fixme: plot/spit results
    # --------------------------------------------------------------------------
    def __init__(self, cuda, model, optimizer, train_loader, valid_loader, test_loader,
                 trainer_root_dir, max_epoch, batch_size,
                 model_save_criteria=0,
                 experiment_name=None,
                 lr_decay_epoch=None,
                 lmk_num=None):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.trainer_root_dir = trainer_root_dir
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.model_save_criteria = model_save_criteria
        self.lr_decay_epoch = lr_decay_epoch
        self.lmk_num = lmk_num
        self.experiment_name = str(experiment_name)
        self.TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
        if self.test_loader is None:
            if self.experiment_name is not None:
                self.subdir = os.path.join(self.trainer_root_dir, self.experiment_name + '_{}'.format(self.TIME_STAMP))
            else:
                self.subdir = os.path.join(self.trainer_root_dir, 'default_{}'.format(self.TIME_STAMP))
            os.makedirs(self.subdir)
        # else:
        #     self.subdir = subdir
        if self.test_loader is None:
            print("Should")
            self.model_pth = os.path.join(self.subdir, self.model.__class__.__name__ + '.pth')
            self.loss_log = os.path.join(self.subdir, 'trainvalid_loss.txt')
            self.FILEPATH_LOG = os.path.join(self.subdir, 'loss_acc.pickle')
        self.epoch = 0  # epoch to start train / start_epoch
        self.iteration = 0
        self.best_mean_iu = 0
        self.train_losses = []
        self.train_dices = []     # accuracy can be dice or any other metic
        self.valid_losses = []
        self.valid_dices = []
        # self.lrlist = []
        self.checkpoint = {}
        if self.lr_decay_epoch is not None:
            """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim, step_size=self.lr_decay_epoch, gamma=0.1)
    def train(self):
        self.model.train()
        running_loss = 0
        running_dice = 0
        fv = open(self.loss_log, 'a')
        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=100, leave=False):  # epoch will be updated by train_epoch()
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred = self.model(data)
            loss, dice, _, _, _ = dice_loss_three(pred, target) #, sub_resource)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is NaN while training')

            # Gradient descent
            self.optim.zero_grad()  # remove existing/previous gradient
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            running_dice += dice.item()
            print('epoch=%d  batch_idx=%d  loss=%.6f \n' % (self.epoch, batch_idx, loss.data))
            print('epoch=%d  batch_idx=%d  dice_organ=%.6f' % (self.epoch, batch_idx, dice.cpu().data.numpy()))
            fv.write('training---------')
            fv.write('epoch=%d  batch_idx=%d  loss=%.4f  dice=%.6f \n' % (self.epoch, batch_idx, loss.data, dice.data))
        if self.lr_decay_epoch is None:
            pass
        else:
            assert self.lr_decay_epoch < self.max_epoch
            self.scheduler.step()
        train_loss = running_loss / len(self.train_loader)
        train_dice = running_dice/ len(self.train_loader)
        self.train_losses.append(train_loss)
        self.train_dices.append(train_dice)
        # self.lrlist.append(self.optim.param_groups[0]['lr'])
        fv.close()

    def validate(self): #, model_save_criteria):
        self.model.eval()  # train
        running_loss = 0
        running_dice = 0
        fv = open(self.loss_log, 'a')
        with torch.no_grad():
            for batch_idx, (data, target, sub_name) in tqdm.tqdm(
                    # enumerate(self.test_loader), total=len(self.test_loader),
                    enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc='Valid epoch=%d' % self.epoch, ncols=100, leave=False):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target, volatile=True)
                pred = self.model(data)
                loss, dice, _, _, _ = dice_loss_three(pred, target) #, sub_resource) # dice --> mean(dice_fg)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is NaN while validating')
                running_loss += loss.item()
                running_dice += dice.item()
                print('epoch=%d  batch_idx=%d  dice_organ=%.6f' % (self.epoch, batch_idx, dice.data)) #.cpu().data.numpy()))
                fv.write('validating-------')
                fv.write('epoch=%d  batch_idx=%d  loss=%.4f  dice=%.6f \n' % (self.epoch, batch_idx, loss.data, dice.data))
        valid_loss = running_loss / len(self.valid_loader)
        valid_dice = running_dice / len(self.valid_loader)
        self.valid_losses.append(valid_loss)
        self.valid_dices.append(valid_dice)
        if valid_loss < self.model_save_criteria:
            self.model_save_criteria = valid_loss
            checkpoint_best = {
                'epoch': self.epoch,
                'arch': self.model.__class__.__name__,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'model_save_criteria': self.model_save_criteria
            }
            self.checkpoint['best_model_state'] = checkpoint_best
            torch.save(self.checkpoint, self.model_pth)
        fv.close()

    def train_epoch(self):
        "codes to start training epochs which includes batch epochs of train and validation"
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train Epoch', ncols=100):
            self.epoch = epoch      # increments the epoch of Trainer
            # checkpoint = {} # fixme: here checkpoint!!!
            # model_save_criteria = self.model_save_criteria
            self.train()
            if epoch % 1 == 0:
                self.validate() #checkpoint) #, self.model_save_criteria)
            checkpoint_latest = {
                'epoch': self.epoch,
                'arch': self.model.__class__.__name__,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'model_save_criteria': self.model_save_criteria
            }
            self.checkpoint['checkpoint_latest'] = checkpoint_latest
            torch.save(self.checkpoint, self.model_pth)
            print("\n Updating log...")
            log = {
                'loss_train': self.train_losses,
                'loss_valid': self.train_dices,
                'dice_train': self.valid_losses,
                'dice_valid': self.valid_dices
                }
            with open(self.FILEPATH_LOG, 'wb') as pfile:
                pickle.dump(log, pfile)

    def loadtest(self, test_loader, subdir, state=0):
        "state: 0 for best_state, state: 1 or None for latest_state"
        print(subdir, "+++")
        model_pth = glob(os.path.join(subdir, '*.pth'))[0]  # '%s/model_epoch_%04d.pth' % (train_root_dir, epoch)
        if state == 0:
            if self.cuda:
                checkpoint = torch.load(model_pth)
                self.model.load_state_dict(checkpoint['best_model_state']['model_state_dict'])
            else:
                checkpoint = torch.load(model_pth, map_location=lambda storage, location: storage)
                self.model.load_state_dict(checkpoint['best_model_state']['model_state_dict'])
        else:
            if self.cuda:
                checkpoint = torch.load(model_pth)
                self.model.load_state_dict(checkpoint['checkpoint_latest']['model_state_dict'])
            else:
                checkpoint = torch.load(model_pth, map_location=lambda storage, location: storage)
                self.model.load_state_dict(checkpoint['checkpoint_latest']['model_state_dict'])
        outDir = os.path.join(subdir, 'out')
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
                enumerate(test_loader), total=len(test_loader),
                desc='Test batch=%d' % self.epoch, ncols=100, leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred = self.model(data)
            lbl_pred = pred.data.max(1)[1].cpu().numpy()[:, :, :].astype('uint8')
            out_nii_file = os.path.join(outDir, '{}_{}.nii.gz'.format(sub_name, batch_idx))
            seg_img = nib.Nifti1Image(lbl_pred[0], affine=np.eye(4))
            nib.save(seg_img, out_nii_file)

class UNet3D(nn.Module): #from SLANT
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        bn = True
        bs = True
        super(UNet3D, self).__init__()
        self.ec0_1_32 = self.encoder(self.in_channel, 32, bias=bs, batchnorm=bn) #True
        self.ec7_256_512 = self.encoder(32, 64, bias=bs, batchnorm=bn)
        self.dc0_64_nClasses = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=True)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels), #BatchNorm2d
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0_1_32(x)
        e7 = self.ec7_256_512(e0)
        d0 = self.dc0_64_nClasses(e7)
        return d0

if __name__ == '__main__':
    # print(use_cuda)
    project_dir = r'/media/banikr2/DATA/banikr_D_drive/model'
    data_dir = r'/media/banikr2/DATA/emModels/CT-MR_batch2/TrialMove/patchdir/LowResHeadSegTrial'
    imgDir = os.path.join(data_dir, 'img')
    segDir = os.path.join(data_dir, 'seg')
    train_img_subs, train_img_files = get_sub_list(imgDir)
    train_seg_subs, train_seg_files = get_sub_list(segDir)
    # print(train_img_subs, train_img_files)
    train_dict = {}
    train_dict['img_subs'] = train_img_subs
    train_dict['img_files'] = train_img_files
    train_dict['seg_subs'] = train_seg_subs
    train_dict['seg_files'] = train_seg_files

    # print(train_dict)
    dataSet = Seg_three_label(train_dict, num_labels=4)
    train_dataset = Subset(dataSet, np.arange(0, 8))
    valid_dataset = Subset(dataSet, np.arange(8, 10))
    bSize = 2
    train_loader = DataLoader(train_dataset, batch_size=bSize, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=bSize, shuffle=False, num_workers=1)
    print("A")
    # x, y, z = next(iter(train_loader))
    # # Initialize model
    model = UNet3D(in_channel=1, n_classes=4).to(device)
    # out = model(x)
    #
    # print(x.shape, y.shape)
    # print(">>", out.shape)
    # Initialize optimizer
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    out = r'/media/banikr2/DATA/banikr_D_drive/model/OutTrainer'

    A = Trainer_v2(cuda=use_cuda,
                   model=model,
                   optimizer=opt,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   test_loader=None,
                   trainer_root_dir=r'/media/banikr2/DATA/banikr_D_drive/model/OutTrainer',
                   max_epoch=50,
                   batch_size=2,
                   experiment_name='Operation_all_clear',
                   lr_decay_epoch=20)
    A.train_epoch()
    # modelDir = r'/media/banikr2/DATA/banikr_D_drive/model/OutTrainer/Checklog_lr_decay_None_2021-07-25-00-50-38'
    # A.loadtest(valid_loader, modelDir, 0)
if __name__ != '__main__':
    model_pth = r'/media/banikr2/DATA/banikr_D_drive/model/OutTrainer/models'
    model_pth = glob(os.path.join(model_pth, '*.pt'))[0]
    print(model_pth)
    checkpoint = torch.load(model_pth)
    model = UNet3D(in_channel=1, n_classes=4).to(device)
    model.load_state_dict(checkpoint['checkpoint_latest']['model_state_dict'])
