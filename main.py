# Copyright 2017 The Wenchen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
run the training/dev/testing
"""


def save_check_point(mod_arg, mod_aux, pre, epoch):
    """
    Save model each epoch, load as:

    sym, arg_params, aux_params = \
        mx.model.load_checkpoint(model_prefix, n_epoch_load)

    # assign parameters
    mod.set_params(arg_params, aux_params)

    OR

    mod.fit(..., arg_params=arg_params, aux_params=aux_params,
            begin_epoch=n_epoch_load)
    """

    save_dict = {('arg:%s' % k): v for k, v in mod_arg.items()}
    save_dict.update({('aux:%s' % k): v for k, v in mod_aux.items()})
    param_name = '%s-%04d.pk' % (pre, epoch)
    pickle.dump(save_dict, open(param_name, "wb"))
    print('Saved checkpoint to \"%s\"' % param_name)


def load_check_point(file_name):
    # Load file
    print(file_name)
    save_dict = pickle.load(open(file_name, "rb"))
    # Extract data from save
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v

    # Recreate model
    cnn = create_vdcnn()
    mod = mx.mod.Module(cnn, context=ctx)

    # Bind shape
    mod.bind(data_shapes=[('data', DATA_SHAPE)],
             label_shapes=[('softmax_label', (BATCH_SIZE,))])

    # assign parameters from save
    mod.set_params(arg_params, aux_params)
    print('Model loaded from disk')

    return mod


def train_model(train_fname):
    # Create mx.mod.Module()
    cnn = create_vdcnn()
    mod = mx.mod.Module(cnn, context=ctx)

    # Bind shape
    mod.bind(data_shapes=[('data', DATA_SHAPE)],
             label_shapes=[('softmax_label', (BATCH_SIZE,))])

    # Initialise parameters and optimiser
    mod.init_params(mx.init.Normal(sigma=SD))
    mod.init_optimizer(optimizer='sgd',
                       optimizer_params={
                           "learning_rate": 0.01,
                           "momentum": 0.9,
                           "wd": 0.00001,
                           "rescale_grad": 1.0 / BATCH_SIZE
                       })

    # Load Data
    X_train, y_train = load_file('amazon_review_polarity_train.csv')

    # Train
    print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)
    print("started training")
    tic = time.time()

    # Evaluation metric:
    metric = mx.metric.Accuracy()

    # Train EPOCHS
    for epoch in range(EPOCHS):
        t = 0
        metric.reset()
        tic_in = time.time()
        for batch in load_data_frame(X_data=X_train,
                                     y_data=y_train,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True):
            # Push data forwards and update metric
            mod.forward_backward(batch)
            mod.update()
            mod.update_metric(metric, batch.label)

            # For training + testing
            # mod.forward(batch, is_train=True)
            # mod.update_metric(metric, batch.label)
            # Get weights and update
            # For training only
            # mod.backward()
            # mod.update()
            # Log every 50 batches = 128*50 = 6400
            t += 1
            if t % 50 == 0:
                train_t = time.time() - tic_in
                metric_m, metric_v = metric.get()
                print("epoch: %d iter: %d metric(%s): %.4f dur: %.0f" % (epoch, t, metric_m, metric_v, train_t))

        # Checkpoint
        arg_params, aux_params = mod.get_params()
        save_check_point(mod_arg=arg_params,
                         mod_aux=aux_params,
                         pre=train_fname,
                         epoch=epoch)
        print("Finished epoch %d" % epoch)

    print("Done. Finished in %.0f seconds" % (time.time() - tic))


def test_model(test_fname):
    # Load saved model:
    mod = load_check_point(test_fname)
    # assert mod.binded and mod.params_initialized

    # Load data
    X_test, y_test = load_file('amazon_review_polarity_test.csv')

    # Score accuracy
    metric = mx.metric.Accuracy()

    # Test batches
    for batch in load_data_frame(X_data=X_test,
                                 y_data=y_test,
                                 batch_size=len(y_test)):
        mod.forward(batch, is_train=False)
        mod.update_metric(metric, batch.label)

        metric_m, metric_v = metric.get()
        print("TEST(%s): %.4f" % (metric_m, metric_v))