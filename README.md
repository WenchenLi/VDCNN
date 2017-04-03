# VDCNN

tensorflow implementation of Very Deep Convolutional Networks
for Text Classification

# RUN

## train
I have toy dataset rt_data_all.txt to get you started with the model.
 
```bash
# config the Training flags in main
python main.py
```

## predict
please take a look at predict.py. I have example for both english and chinese
```bash
# config the Training flags in main
python predict.py
```

### model and implementation details
- Convolutional block

![conv block](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/84ca430856a92000e90cd728445ca2241c10ddc3/3-Figure2-1.png)

- the basic whole model

![VDCNN](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/84ca430856a92000e90cd728445ca2241c10ddc3/2-Figure1-1.png)

- shortcut 

as shows in paper shortcut is not always helping, so we dont implement shortcut here yet, put it 
as future TODO.

![shortcut]( https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/84ca430856a92000e90cd728445ca2241c10ddc3/7-Table5-1.png)

above table :
Evolution of the train/test error on the Yelp Review Full data set for all depths, and with or without shortcut connections (ResNet).

- different depth,(K)max pooling 

as shows in the table, k max pooling not always helps, so keep to max pooling for now, mark KmaxPooling as TODO 

![depth](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/84ca430856a92000e90cd728445ca2241c10ddc3/5-Table4-1.png)

above Table : Testing error of our models on the 8 data sets. The deeper the networks the lower the error for all pooling types. No data preprocessing or augmentation is used.
 

## how to train your own model

### training input data format
each sentence should be separated by line.
for each line, it starts with the training sentence, followed by the label.
label should be started with   `__label__` just to be consistent with [fasttext](https://github.com/facebookresearch/fastText) input format
following is a few examples, where 'pos' and 'neg'  are labels.
```
the thing looks like a made-for-home-video quickie . __label__neg
effective but too-tepid biopic __label__pos
```

## TODO
2. on our own data[result]
3. uniform sample from different category[weighted sample][precision/recall/f1][give different weights to the positive and negative examples in the training criterion.]
3. add [n-gram cnn as comparision](https://github.com/dennybritz/cnn-text-classification-tf)
4. separate accuracy for each class 

2. prototxt config
4. optimize readin data queue
4. add shortcut
5. add kMaxPooling
6. fold for dynamic graph

# reference
[1][Conneau, Alexis et al. “Very Deep Convolutional Networks for Natural Language Processing.” CoRR abs/1606.01781 (2016): n. pag.](https://pdfs.semanticscholar.org/f797/fd44b9ddd5845611eb7a705ca9464a8819d1.pdf?_ga=1.122241998.496193353.1486868690)
