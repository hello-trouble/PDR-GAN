import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import stage2net
from . import networks
import torch.nn.functional as F
from torch.autograd import Variable

from util import util


class RefinedDCPModel(BaseModel):
    """
    This class implements the RefineDNet model, for learning single image dehazing without paired data.
    It adopts the basic backbone networks provided by CycleGAN.

    The model training requires '--dataset_mode unpaired' dataset.
    By default, it uses a '--netR_T unet_trans_256' U-Net refiner,
    a '--netR_J resnet_9blocks' ResNet refiner,
    and a '--netD basic' discriminator (PatchGAN introduced by pix2pix).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_G', type=float, default=0.05, help='weight for loss_G_single')
            parser.add_argument('--lambda_identity', type=float, default=1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rec_I', type=float, default=1, help='weight for loss_rec_I')
            parser.add_argument('--lambda_tv', type=float, default=1, help='weight for TV loss of refine_T')
            parser.add_argument('--lambda_vgg', type=float, default=0, help='weight for loss_vgg')
        
        parser.add_argument('--netR_T', type=str, default='unet_trans_256', help='specify generator architecture')
        parser.add_argument('--netR_J', type=str, default='resnet_9blocks', help='specify generator architecture')

        return parser

    def __init__(self, opt):
        """Initialize the RefineDNet class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.stage = opt.training_stage
        if self.stage ==0:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = ['D_single', 'G_single', 'rec_I', 'TV_T', 'idt_J', 'vgg']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            if self.isTrain:
                self.visual_names = ['real_I', 'dcp_T_vis', 'refine_T_vis', 'out_T_vis', 'dcp_J','refine_J', 'rec_I', 'rec_J','map_A', 'real_J', 'ref_real_J']
            else:
                self.visual_names = ['real_I', 'dcp_T_vis', 'refine_T_vis', 'out_T_vis', 'dcp_J','refine_J', 'rec_I', 'rec_J','map_A']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
            if self.isTrain:
                self.model_names = ['Refiner_T', 'Refiner_J', 'D','Stage2_RefUNet', 'Stage2_DepthGuidedD']
            else:  # during test time, only load Gs
                self.model_names = ['Refiner_T', 'Refiner_J','Stage2_RefUNet']
        if self.stage ==1:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = [ 'stage2_D_single',
                               'stage2_G_single', 'stage2_idt_J']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            if self.isTrain:
                self.visual_names = ['real_I', 'dcp_T_vis', 'refine_T_vis', 'out_T_vis', 'dcp_J', 'refine_J', 'rec_I',
                                     'rec_J', 'map_A', 'real_J', 'ref_real_J', 'stage2_ref_J']
            else:
                self.visual_names = ['real_I', 'dcp_T_vis', 'refine_T_vis', 'out_T_vis', 'dcp_J', 'refine_J', 'rec_I',
                                     'rec_J', 'map_A', 'stage2_ref_J']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
            if self.isTrain:
                self.model_names = ['Refiner_T', 'Refiner_J', 'D', 'Stage2_RefUNet', 'Stage2_DepthGuidedD']
            else:  # during test time, only load Gs
                self.model_names = ['Refiner_T', 'Refiner_J', 'Stage2_RefUNet']
        if self.stage == 2:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = ['D_single', 'G_single', 'rec_I', 'TV_T', 'idt_J', 'vgg', 'stage2_D_single',
                               'stage2_G_single', 'stage2_idt_J']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            if self.isTrain:
                self.visual_names = ['real_I', 'dcp_T_vis', 'refine_T_vis', 'out_T_vis', 'dcp_J', 'refine_J', 'rec_I',
                                     'rec_J', 'map_A', 'real_J', 'ref_real_J', 'stage2_ref_J']
            else:
                self.visual_names = ['real_I', 'dcp_T_vis', 'refine_T_vis', 'out_T_vis', 'dcp_J', 'refine_J', 'rec_I',
                                     'rec_J', 'map_A', 'stage2_ref_J']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
            if self.isTrain:
                self.model_names = ['Refiner_T', 'Refiner_J', 'D', 'Stage2_RefUNet', 'Stage2_DepthGuidedD']
            else:  # during test time, only load Gs
                self.model_names = ['Refiner_T', 'Refiner_J', 'Stage2_RefUNet']

        # define networks (both Generators and discriminators)
        self.netG_DCP = networks.init_net(networks.DCPDehazeGenerator(), gpu_ids=self.gpu_ids) # use default setting for DCP
        self.netRefiner_T = networks.define_G(opt.input_nc+1, 1, opt.ngf, opt.netR_T, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netRefiner_J = networks.define_G(opt.input_nc+opt.output_nc, opt.output_nc, opt.ngf, opt.netR_J, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netStage2_RefUNet = stage2net.define_strage_G(9,3,opt.init_type, opt.init_gain, self.gpu_ids)




        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netStage2_DepthGuidedD = stage2net.define_strage_D(3, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_I_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_J_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()
            self.criterionVGG = networks.VGGLoss() if self.opt.lambda_vgg > 0.0 else None
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netRefiner_T.parameters(), self.netRefiner_J.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            ###########################stage2####################################
            ###loss
            self.stage2_criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss in stage2.
            self.stage2_criterionMSE = torch.nn.MSELoss()
            self.stage2_criterionIdt = torch.nn.L1Loss()

            ###Adam
            self.stage2_optimizer_G = torch.optim.Adam(itertools.chain(self.netStage2_RefUNet.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.stage2_optimizer_D = torch.optim.Adam(self.netStage2_DepthGuidedD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.stage2_optimizers.append(self.stage2_optimizer_G)
            self.stage2_optimizers.append(self.stage2_optimizer_D)


        # display the architecture of each part
        # print(self.netRefiner_T)
        # print(self.netRefiner_J)
        # if self.isTrain:
        #     print(self.netD)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_I = input['haze'].to(self.device) # [-1, 1]
        self.image_paths = input['paths']

        if self.isTrain:
            self.real_J = input['clear'].to(self.device) # [-1, 1]


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        dcp_J, self.dcp_T, self.dcp_A = self.netG_DCP(self.real_I)

        #scale to [-1,1]
        self.dcp_J = (torch.clamp(dcp_J,0,1)-0.5)/0.5

        # output scale [0,1]
        self.refine_T, self.out_T = self.netRefiner_T(torch.cat((self.real_I, self.dcp_T), 1))
        self.refine_J = self.netRefiner_J(torch.cat((self.real_I, self.dcp_J), 1)) #ref_j

        # reconstruct haze image
        shape = self.refine_J.shape
        dcp_A_scale = self.dcp_A
        self.map_A = (dcp_A_scale).reshape((1,3,1,1)).repeat(1,1,shape[2],shape[3])

        refine_T_map = self.refine_T.repeat(1,3,1,1)
        self.rec_I = util.synthesize_fog(self.refine_J, refine_T_map, self.map_A)
        self.rec_J = util.reverse_fog(self.real_I, refine_T_map, self.map_A)

        ###stage2
        self.stage2_ref_J = self.netStage2_RefUNet(torch.cat((self.real_I, self.refine_J , self.rec_J), 1))


    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()


    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        # rescale to [-1,1] for visdom 
        self.refine_T_vis = (self.refine_T - 0.5)/0.5
        self.out_T_vis = (self.out_T - 0.5)/0.5
        self.dcp_T_vis = (self.dcp_T - 0.5)/0.5
        # self.map_A_vis = (self.map_A - 0.5)/0.5


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def backward_D(self):
        fake_J = self.fake_I_pool.query(self.refine_J)
        self.loss_D_single = self.backward_D_basic(self.netD, self.real_J, fake_J)

    def stage2_backward_D(self):

        fake_J = self.fake_J_pool.query(self.stage2_ref_J)
        # Real
        b, c, h, w = fake_J.size()
        depth_gt_real = Variable(torch.zeros(b, 1, h, w)).to(self.device)

        depth_real, pred_real = self.netStage2_DepthGuidedD(self.stage2_ref_J)
        loss_D_real = self.stage2_criterionGAN(pred_real, True)

        d_realdepth_loss = self.stage2_criterionMSE(depth_real, depth_gt_real)
        total_loss_real = loss_D_real + d_realdepth_loss

        # Fake
        depth_fake,pred_fake = self.netStage2_DepthGuidedD(fake_J.detach())
        loss_D_fake = self.stage2_criterionGAN(pred_fake, False)
        depth_fake = depth_fake.repeat(1, 3, 1, 1)
        d_fakedepth_loss = self.stage2_criterionMSE(depth_fake, 1 - self.refine_T.detach())
        total_loss_fake = loss_D_fake + d_fakedepth_loss

        # Combined loss and calculate gradients
        self.loss_stage2_D_single = (total_loss_real + total_loss_fake) * 0.5
        self.loss_stage2_D_single.backward()

   # 'stage2_D_single', 'stage2_G_single', 'stage2_idt_J'
    def stage2_backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_G = self.opt.lambda_G

        # Generator losses for rec_I and refine_J
        self.loss_stage2_G_single = self.stage2_criterionGAN(self.netStage2_DepthGuidedD(self.stage2_ref_J)[1], True) * lambda_G

        # Identity loss, ||refiner_J(real_J) - real_J||

        self.stage2_ref_real_J = self.netStage2_RefUNet(torch.cat((self.real_I, self.real_J,self.real_J), 1))

        self.loss_stage2_idt_J = self.stage2_criterionIdt(self.stage2_ref_real_J, self.real_J) * lambda_idt \
            if lambda_idt > 0.0 \
            else 0

        self.loss_stage2_G = self.loss_stage2_G_single  + self.loss_stage2_idt_J

        self.loss_stage2_G.backward(retain_graph=True)


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_tv = self.opt.lambda_tv
        lambda_G = self.opt.lambda_G
        lambda_rec_I = self.opt.lambda_rec_I
        lambda_vgg = self.opt.lambda_vgg

        # Generator losses for rec_I and refine_J
        self.loss_G_single = self.criterionGAN(self.netD(self.refine_J), True)*lambda_G

        # Reconstrcut loss
        self.loss_rec_I = self.criterionRec(self.rec_I, self.real_I) * lambda_rec_I

        # perecptual loss
        self.loss_vgg = self.criterionVGG(self.refine_J, self.dcp_J)*lambda_vgg if lambda_vgg > 0.0 else 0

        # TV loss
        self.loss_TV_T = self.criterionTV(self.out_T)*lambda_tv if lambda_tv > 0.0 else 0

        # Identity loss, ||refiner_J(real_J) - real_J||
        self.ref_real_J = self.netRefiner_J(torch.cat((self.real_I, self.real_J), 1))
        self.loss_idt_J = self.criterionIdt(self.ref_real_J, self.real_J)*lambda_idt \
                            if lambda_idt > 0.0 \
                            else 0
    
        self.loss_G = self.loss_G_single + self.loss_rec_I + self.loss_idt_J \
                     + self.loss_TV_T \
                     + self.loss_vgg
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

    def stage2_optimize_parameters(self):

        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netStage2_DepthGuidedD, False)  # Ds require no gradients when optimizing Gs
        self.stage2_optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.stage2_backward_G()  # calculate gradients for G_A and G_B
        self.stage2_optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad(self.netStage2_DepthGuidedD, True)
        self.stage2_optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.stage2_backward_D()  # calculate gradients for D_A
        self.stage2_optimizer_D.step()  # update D_A and D_B's weights

    def set_gradients(self, trainable):
        if len(self.gpu_ids) > 1:
            for param in self.netRefiner_T.module.parameters():
                param.requires_grad = trainable
            for param in self.netRefiner_J.module.parameters():
                param.requires_grad = trainable
            for param in self.netD.module.parameters():
                param.requires_grad = trainable
        else:
            for param in self.netRefiner_T.parameters():
                param.requires_grad = trainable
            for param in self.netRefiner_J.parameters():
                param.requires_grad = trainable
            for param in self.netD.parameters():
                param.requires_grad = trainable


    def trainable(self, model, trainable):
        for parameter in model.parameters():
            parameter.requires_grad = trainable

    def load_pretrained(self):

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for key, value in torch.load("./pretrained/60_net_Refiner_T.pth").items():
            name = 'module.' + key
            new_state_dict[name] = value
        # state_dict = torch.load('./pretrained/60_net_Refiner_T.pth', map_location=str(self.device))
        self.netRefiner_T.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        for key, value in torch.load("./pretrained/60_net_Refiner_J.pth").items():
            name = 'module.' + key
            new_state_dict[name] = value
        # state_dict = torch.load('./pretrained/60_net_Refiner_J.pth', map_location=str(self.device))
        self.netRefiner_J.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        for key, value in torch.load("./pretrained/60_net_D.pth").items():
            name = 'module.' + key
            new_state_dict[name] = value
        # state_dict = torch.load('./pretrained/60_net_D.pth', map_location=str(self.device))
        self.netD.load_state_dict(new_state_dict)