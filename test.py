import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time

# python test.py --dataroot your_path/deepfashion/fashion_resize --dirSem your_path/deepfashion --pairLst your_path/deepfashion/fashion-resize-pairs-test.csv 
# --checkpoints_dir ./checkpoints --results_dir ./results --name fashion_AdaGen_sty512_nres8_lre3_SS_fc_vgg_cxloss_ss_merge3 
# --model adgan --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no 
# --gpu_ids 0,1 --BP_input_nc 18 --no_flip --which_model_netG ADGen --which_epoch 800


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

opt.how_many = 999999
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime)
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]
    print(img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()




