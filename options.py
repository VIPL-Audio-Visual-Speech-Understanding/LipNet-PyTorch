gpu = '0'
random_seed = 0
video_path = 'lip/'
train_list = 'data/unseen_train.txt'
val_list = 'data/unseen_val.txt'
anno_path = 'GRID_align_txt'
vid_padding = 75
txt_padding = 200
batch_size = 120
base_lr = 1e-4
num_workers = 16
max_epoch = 10000
display = 10
test_step = 100
save_prefix = 'weights/LipNet_unseen'
is_optimize = False

weights = 'pretrain/LipNet_unseen_loss_0.3097497522830963_wer_0.13465623428858722_cer_0.0681201566447587.pt'
