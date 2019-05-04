from __future__ import division
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
# Custom Modules
from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader

# Plot Parameters
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('data_dir',
                        help ='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('val_data_dir',
                        help ='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images'
                        )
    parser.add_argument('model_path', help ='path to the trained model')
    parser.add_argument('output_directory',
                        help ='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type = int, help = 'input height',
                        default = 128)
    parser.add_argument('--input_width', type = int, help = 'input width',
                        default = 256)
    parser.add_argument('--model', default = 'ConvLSTM')
                        
    parser.add_argument('--pretrained', default = False,
                        help = 'Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default = 'train',
                        help = 'mode: train or test (default: train)')
    parser.add_argument('--epochs', default = 50,
                        help = 'number of total epochs to run')
    parser.add_argument('--learning_rate', default = 1e-4,
                        help = 'initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default = 8,
                        help = 'mini-batch size (default: 4)')
    parser.add_argument('--adjust_lr', default = True,
                        help = 'apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default = 'cuda:0',
                        help = 'choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--print_images', default = False,
                        help ='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default = False,
                        help = 'print weights of every layer')
    parser.add_argument('--input_channels', default = 3,
                        help = 'Number of channels in input tensor')
    parser.add_argument('--num_workers', default = 8,
                        help = 'Number of workers in dataloader')
    parser.add_argument('--do_augmentation', default = False,
                        help = 'Do augmentation of images or not')
    parser.add_argument('--use_multiple_gpu', default = False)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):								
    """Sets the learning rate to the initial LR\
       decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:

    def __init__(self, args):
        self.args = args
        self.device = args.device  ###											
        self.model = get_model(args.model, input_channels = args.input_channels, pretrained = args.pretrained)
	self.model = self.model.double()		
										
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
	
	self.model = self.model.to(self.device) ###
	#self.model.cuda()

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(	n = 4,
                				SSIM_w = 0.85,
                				disp_gradient_w = 0.1, lr_w = 1).to(self.device)  ###
            self.optimizer = optim.Adam(self.model.parameters(),							
                                        lr = args.learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode,
                                                                 False, args.batch_size,
								 args.input_channels,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 4  ###
 
        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode,
                                                     args.do_augmentation, args.batch_size,
						     args.input_channels,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)
        if 'cuda' in self.device: ###
           torch.cuda.synchronize()


    def train(self):
        losses = []
        val_losses = []
	best_loss = float('Inf')
        best_val_loss = float('Inf')
        running_val_loss = 0.0
	torch.set_default_tensor_type('torch.DoubleTensor')

        
	self.model.eval()
	
        for data in self.val_loader:
            #data = to_device(data, self.device)
	    
	    left = data['left_video'].cuda()
	    right = data['right_video'].cuda()
            disps = self.model(left)
	    n_frames = left.size()[1]
	    loss = 0
            for idx in range(n_frames):
                    loss = loss + self.loss_function([disps[0][:,idx,:,:,:], disps[1][:,idx,:,:,:], disps[2][:,idx,:,:,:], disps[3][:,idx,:,:,:]], (left[:,idx,:,:,:], right[:,idx,:,:,:]))
            val_losses.append(loss.item())
            running_val_loss += loss.item()										
        running_val_loss /= self.val_n_img / self.args.batch_size
        print('Val_loss:', running_val_loss)
	prev_val_loss = running_val_loss
	count = 0
        for epoch in range(self.args.epochs):
	    print('Starting epoch....................', epoch+16)
            if self.args.adjust_lr:
            	adjust_learning_rate(self.optimizer, epoch,								
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for data in self.loader:
                # Load data
                # data = to_device(data, self.device)
		left = data['left_video'].cuda()
                right = data['right_video'].cuda()
		n_frames = left.size()[1]
                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
		loss = 0
                for idx in range(n_frames):
                    loss = loss + self.loss_function([disps[0][:,idx,:,:,:], disps[1][:,idx,:,:,:], disps[2][:,idx,:,:,:], disps[3][:,idx,:,:,:]], (left[:,idx,:,:,:], right[:,idx,:,:,:]))
                loss.backward()												
                self.optimizer.step()											
                losses.append(loss.item())

                # Print statistics
                if self.args.print_weights:										
                    j = 1
                    for (name, parameter) in self.model.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            plt.subplot(5, 9, j)
                            plt.hist(parameter.data.view(-1))
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    plt.show()

                running_loss += loss.item()

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                #data = to_device(data, self.device)
		left = data['left_video'].cuda()
                right = data['right_video'].cuda()
                disps = self.model(left)
		n_frames = left.size()[1]
		loss = 0
                for idx in range(n_frames):
                	loss = loss + self.loss_function([disps[0][:,idx,:,:,:], disps[1][:,idx,:,:,:], disps[2][:,idx,:,:,:], disps[3][:,idx,:,:,:]], (left[:,idx,:,:,:], right[:,idx,:,:,:]))
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            self.save(self.args.model_path[:-4] + str(epoch+16) + '_last.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.model_path[:-4] + str(epoch+16) + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')
	    if(abs(prev_val_loss-running_val_loss) < 0.001):
		count += 1
	    else:
		count = 0
	    if(count == 5):
		break;
	    #print(prev_val_loss,running_val_loss)
	    prev_val_loss = running_val_loss
        print ('Finished Training', best_val_loss)
        self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
	torch.set_default_tensor_type('torch.DoubleTensor')
        self.model.eval()
	no_of_frames = 3
        with torch.no_grad():
	    count = 0
            for (i, data) in enumerate(self.loader):
		count += 1
		data = data.cuda()
		left = data
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0]
		print(disp.size())
		np.save(self.output_directory + '/disparities_' +  str(count) + '.npy', disp)


def main(args):
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()

