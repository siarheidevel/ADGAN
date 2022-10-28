import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

# def add_path(path):
#     import sys
#     if path not in sys.path:
#         sys.path.insert(0, path)


# add_path('/home/deeplab/devel/ADGAN')

# python train.py --dataroot /home1/menyf/data/deepfashion/fashion_resize --dirSem /home1/menyf/data/deepfashion 
# --pairLst /home1/menyf/data/deepfashion/fashion-resize-pairs-train.csv --name fashion_adgan_test 
# --model adgan --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --dataset_mode keypoint --n_layers 3 --norm instance 
# --batchSize 6 --pool_size 0 --resize_or_crop no --gpu_ids 8,9 --BP_input_nc 18 --SP_input_nc 8 
# --no_flip --which_model_netG ADGen --niter 500 --niter_decay 500 
# --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1 --display_id 0

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
