from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    opt = TrainOptions().parse()

    opt.isTrain = False
    cudnn.benchmark = True
    opt.no_log =True
    opt.display_id=0
    opt.verbose = False

    datadir = "./reflection_data"

    # Define evaluation/test dataset

    eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'our_real'),enable_transforms=True)

    eval_dataloader_ceilnet = datasets.DataLoader(
        eval_dataset_ceilnet, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    engine = Engine(opt)

    """Main Loop"""
    result_dir = './results'

    # evaluate on synthetic test data from CEILNet
    res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table1', savedir=join(result_dir, 'our_real'))#