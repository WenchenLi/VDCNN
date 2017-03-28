# VDCNN

tensorflow implementation of Very Deep Convolutional Networks
for Text Classification

# RUN
```bash

```

## TODO
1. a benchmark datasets shows it's working
1. predictor
2. prototxt config
3. pragmatically add blocks

4. add shortcut
5. add kMaxPooling
6. fold for dynamic graph


### pretrain
- train char level nlp for english and chinese 

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
 
 
