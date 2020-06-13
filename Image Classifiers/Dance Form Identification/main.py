import os
from fastai import *
from fastai.vision import *
import  pandas as pd

input_path = 'dataset'
train_df = pd.read_csv(os.path.join(input_path,'train.csv'))
print(train_df.head())

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

np.random.seed(42)
src = (ImageList.from_csv(input_path, 'train.csv', folder='train')
       .split_by_rand_pct(0.2)
       .label_from_df())

data = (src.transform(tfms, size=128)
        .databunch(bs=64).normalize(imagenet_stats))

acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, models.resnet50, metrics=[acc_02, f_score])


