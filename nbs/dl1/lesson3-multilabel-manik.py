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

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *

path = pathlib.Path('/Users/singhalmanik/Code/Datasets/keras-multi-label/dataset/')
path

#labels for path
def labelList(path):
    #print(path)
    comp = path.parts
    return comp[-2].split('_')


np.random.seed(42)
src = (ImageItemList.from_folder(path, extensions=['.jpg', '.jpeg', '.png'])
      .random_split_by_pct(0.2)
        .label_from_func(func = labelList)
      )
src

tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
bs = 32

data = (src.transform(tfms, size=128)
        .databunch(bs=bs).normalize(imagenet_stats)
       )

data.show_batch(3, figsize=(9,16))

arch = models.resnet34

acc_02 = partial(accuracy_thresh, thresh=0.2)
learner = create_cnn(data, arch, metrics=acc_02)

learner.lr_find()

learner.recorder.plot()

lr = 0.01

learner.fit_one_cycle(5, slice(lr))

learner.save('dress-1-res34')

learner.unfreeze

learner.lr_find()
learner.recorder.plot()

learner.fit_one_cycle(5, slice(1e-4, lr/5))

learner.save('dress-2-res34')

learner.recorder.plot_losses()

learner.show_results()

learner.summary()

learner.recorder.plot()

interep = ClassificationInterpretation.from_learner(learner)
interep.top_losses()

#

img = learner.data.train_ds[800][0]
show_image(img)
learner.predict(img)

learner.export('dress-resnet34')

test_path = Path('/Users/singhalmanik/Code/Datasets/keras-multi-label/examples/new')
test = ImageItemList.from_folder(test_path)
len(test)


#learner.data.test_ds = test
learn2 = load_learner(path, fname='dress-resnet34', test=test)
preds, _ = learn2.get_preds(ds_type=DatasetType.Test)

learn2.data.test_ds

thresh = 0.2
labelled_preds = [' '.join([learn2.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]

labelled_preds[:]


