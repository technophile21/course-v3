# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Image segmentation with CamVid

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *
from fastai.callbacks.hooks import *

path = untar_data(URLs.CAMVID)
path.ls()

path_lbl = path/'labels'
path_img = path/'images'

# ## Subset classes

# +
# path = Path('./data/camvid-small')

# def get_y_fn(x): return Path(str(x.parent)+'annot')/x.name

# codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
#     'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])

# src = (SegmentationItemList.from_folder(path)
#        .split_by_folder(valid='val')
#        .label_from_func(get_y_fn, classes=codes))

# bs=8
# data = (src.transform(get_transforms(), tfm_y=True)
#         .databunch(bs=bs)
#         .normalize(imagenet_stats))
# -

# ## Data

fnames = get_image_files(path_img)
fnames[:3]

lbl_names = get_image_files(path_lbl)
lbl_names[:3]

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))

img_l = lbl_names[0]
img_l = open_image(img_l)
img_l.show(figsize=(5,5), alpha=1)

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), alpha=1)
print(mask.shape)

src_size = np.array(mask.shape[1:])
src_size,mask.data

codes = np.loadtxt(path/'codes.txt', dtype=str); codes

# ## Datasets

size = src_size//2
bs=4  #8 (CUDA out of memory for 8)

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

data.show_batch(2, figsize=(10,7))

data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)

# ## Model

# +
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
# -

metrics=acc_camvid
# metrics=accuracy

wd=1e-2

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

lr_find(learn)
learn.recorder.plot()

lr=3e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

learn.save('camvid-stage-1')

learn.load('camvid-stage-1');

learn.show_results(rows=3, figsize=(8,9))

learn.unfreeze()

lrs = slice(lr/400,lr/4)

learn.fit_one_cycle(12, lrs, pct_start=0.8)

learn.save('camvid-stage-2-92');

# ## Go big

# You may have to restart your kernel and come back to this stage if you run out of memory, and may also need to decrease `bs`.

size = src_size
bs=3

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

learn.load('stage-2');

lr_find(learn)
learn.recorder.plot()

lr=1e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

learn.save('stage-1-big')

learn.load('stage-1-big');

learn.unfreeze()

lrs = slice(1e-6,lr/10)

learn.fit_one_cycle(10, lrs)

learn.save('stage-2-big')

learn.load('stage-2-big');

learn.show_results(rows=3, figsize=(10,10))



# ## fin


