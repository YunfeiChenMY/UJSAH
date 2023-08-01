import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from spectral_norm import spectral_norm as SpectralNorm


class ImgNet(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(ImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        b = 4096
        self.fc_encode1 = nn.Linear(img_feat_len, b)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc_encode3 = nn.Linear(b, b)
        self.fc_encode2 = nn.Linear(b, code_len)
        # self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        torch.nn.init.normal_(self.fc_encode2.weight, mean=0.0, std=1)


    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        # feat = F.relu(self.BN1(self.fc_encode1(x)))
        feat = self.relu(self.fc_encode1(x))
        # feat = F.relu(self.fc_encode3(feat))
        feat = self.fc_encode2(self.dp(feat))
        # hid = self.BN2(hid)
        code = self.tanh(1 * feat)#2

        return feat, feat, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        a = 2048
        self.fc1 = nn.Linear(txt_feat_len, a)
        # self.BN1 = nn.BatchNorm1d(a)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(a, a)
        # self.fc21 = nn.Linear(a, a)
        # self.fc22 = nn.Linear(a, a)
        self.fc3 = nn.Linear(a, code_len)
        # self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        # torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=1)

    def forward(self, x):
        # feat = F.relu(self.BN1(self.fc1(x)))
        feat = self.relu(self.fc1(x))
        # feat = self.relu(self.fc2(feat))
        # feat = ac(self.fc21(feat))
        # feat = ac(self.fc22(feat))
        # hid = self.BN2(self.fc3(feat))
        feat = self.fc3(self.dp(feat))
        code = self.tanh(10 * feat)#20
        return feat, feat, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class DeImgNet(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(DeImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode1 = nn.Linear(code_len, img_feat_len)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.BN1 = nn.BatchNorm1d(4096)
        self.fc_encode2 = nn.Linear(4096, 4096)
        self.fc_encode3 = nn.Linear(4096, img_feat_len)
        self.BN2 = nn.BatchNorm1d(img_feat_len)
        # self.fc_encode = nn.Linear(code_len, 4096)
        self.alpha = 1.0
        self.Dealpha = 0.5


    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        # hid = F.relu(self.BN1(self.fc_encode1(x)))
        hid = self.relu(self.fc_encode1(x))
        hid = self.fc_encode2(hid)
        hid = self.fc_encode3(hid)
        hid = self.fc_encode3(hid)

        # hid = self.fc_encode(x)
        code = hid #* self.Dealpha

        return x, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class DeTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(DeTxtNet, self).__init__()
        self.fc1 = nn.Linear(code_len, 1024)
        # self.BN1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, txt_feat_len)
        # self.BN2 = nn.BatchNorm1d(txt_feat_len)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.alpha = 1.0
        self.Dealpha = 0.5

    def forward(self, x):
        # feat = F.relu(self.BN1(self.fc1(x)))
        hid = self.relu(self.fc1(x))
        feat = self.fc2(hid)
        # hid = self.BN2(self.fc3(feat))
        hid = self.fc3(feat)
        code = hid #* self.Dealpha + self.Dealpha
        return hid, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class GenHash(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(GenHash, self).__init__()
        self.fc1 = nn.Linear(code_len * 2, code_len)
        # self.BN1 = nn.BatchNorm1d(code_len * 2)
        # # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(code_len * 2, code_len)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        self.Dealpha = 0.5
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1)

    def forward(self, x):
        # feat = F.relu(self.BN1(self.fc1(x)))
        # feat = F.relu(self.fc1(x))
        # hid = self.fc2(feat)
        # hid = self.BN2(self.fc3(feat))
        hid = self.fc1(x)
        code = torch.tanh(1 * self.BN2(hid)) #* self.Dealpha + self.Dealpha
        B = torch.sign(code)
        return code, B

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GeImgNet(nn.Module):
    def __init__(self, code_len):
        super(GeImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode1 = nn.Linear(code_len, 512)
        # self.fc_encode2 = nn.Linear(1024, 1024)
        self.fc_encode3 = nn.Linear(512, 4096)
        self.alpha = 1.0
        self.Dealpha = 1.0


    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        hid = self.fc_encode1(x)
        # hid = self.fc_encode2(hid)
        hid = self.fc_encode3(hid)
        code = F.relu(self.alpha * hid) #* self.Dealpha

        return x, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class GeTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(GeTxtNet, self).__init__()
        self.fc1 = nn.Linear(code_len, 512)
        # self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(512, txt_feat_len)
        self.alpha = 1.0
        self.Dealpha = 0.5

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        hid = self.fc3(hid)
        code = (F.tanh(self.alpha * hid)) #* self.Dealpha
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class Txt2Img(nn.Module):
    def __init__(self, txt_code_len, img_code_len):
        super(Txt2Img, self).__init__()
        self.fc1 = nn.Linear(txt_code_len, 1024)
        self.fc2 = nn.Linear(1024, img_code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class Img2Txt(nn.Module):
    def __init__(self, img_code_len, txt_code_len):
        super(Img2Txt, self).__init__()
        self.fc1 = nn.Linear(img_code_len, 1024)
        self.fc2 = nn.Linear(1024, txt_code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)



class DiscriminatorImg(nn.Module):
    """
    Discriminator network with PatchGAN.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
    """
    def __init__(self, code_len):
        super(DiscriminatorImg, self).__init__()
        self.fc_encode = nn.Linear(4096, code_len + 1)
        self.alpha = 1.0

    def forward(self, x):
        hid = self.fc_encode(x)
        code = F.tanh(self.alpha * hid)
        return code.squeeze()

class DiscriminatorText(nn.Module):
    """
    Discriminator network with PatchGAN.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
    """
    def __init__(self, code_len):
        super(DiscriminatorText, self).__init__()
        self.fc_encode = nn.Linear(1386, code_len + 1)
        self.alpha = 1.0

    def forward(self, x):
        hid = self.fc_encode(x)
        code = F.tanh(self.alpha * hid)
        return code.squeeze()


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, label, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor

    def __call__(self, prediction, label, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss