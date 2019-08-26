gpu = '3'
random_seed = 0
video_path = '/ssd/GRID/lip'
train_list = 'data/unseen_train.txt'
val_list = 'data/unseen_val.txt'
anno_path = '/ssd/GRID/GRID_align_txt'
vid_padding = 75
txt_padding = 200
batch_size = 120
base_lr = 1e-4 * (batch_size/50)
num_workers = 8
max_epoch = 10000
display = 10
test_step = 1000
save_prefix = 'weights/LipNet_unseen'
is_optimize = False

weights = 'pretrain/LipNet_unseen_loss_0.3112999200820923_wer_0.13454311211664152_cer_0.06858054560148422.pt'
