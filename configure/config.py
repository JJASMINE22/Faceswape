# ===generator===
root_path = "文件根目录(绝对路径)"
image_size = (256, 256)  # set heigth to 100, width to 32
batch_size = 32
train_steps = 1000
validate_steps = 200
transform_kwargs = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4
}

scale_size=(5, 5)
input_shape=(128, 128)
padding_size=(16, 16)


# ===training===
Epoches = 100
per_sample_interval = 200
learning_rate = 5e-5
betas = {"beta_1": .5,
         "beta_2": .999}
ckpt_path = '目录\\checkpoint'
former_sample_path = '目录\\sample\\former_Batch{:d}.jpg'
latter_sample_path = '目录\\sample\\latter_Batch{:d}.jpg'
