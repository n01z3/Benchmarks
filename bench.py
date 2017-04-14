import mxnet as mx
import numpy as np
import importlib
import os, sys
import logging
import argparse

# logging.basicConfig(level=logging.DEBUG)

log_file = "logfile"
log_dir = "./"
log_file_full_name = os.path.join(log_dir, log_file)
head = '%(asctime)-15s Node[' + str(mx.kvstore.create("local").rank) + '] %(message)s'

logger = logging.getLogger()
handler = logging.FileHandler(log_file_full_name)
formatter = logging.Formatter(head)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

DATA_PATH = "./"
PART = 'gen_4k'
TRAIN_BIN = 'data.rec'
VAL_BIN = "val.rec"
DATA_SHAPE = (3, 224, 224)

# ====================================================

NUM_GPU = 1
BATCH_SIZE = 12
LR = 0.001

PROCESS = 'finetune'
START_EPOCH = 0
NET_TYPE = 'vgg19'

# ====================================================

kv = mx.kvstore.create("local")


def finetune_symbol(symbol, model, **kwargs):
    initializer = mx.initializer.Load(param=model.arg_params, default_init=mx.init.Uniform(0.001))
    new_model = mx.model.FeedForward(symbol=symbol, initializer=initializer, **kwargs)
    return new_model


def get_train_val(train_bin=TRAIN_BIN,
                  val_bin=VAL_BIN,
                  batch_size=BATCH_SIZE,
                  data_shape=DATA_SHAPE,
                  aug=True, mean=False):
    train_path = os.path.join(DATA_PATH, train_bin)
    val_path = os.path.join(DATA_PATH, val_bin)

    train = mx.io.ImageRecordIter(
        path_imgrec=train_path,
        mean_r=128.0 if mean else 0.0,
        mean_g=128.0 if mean else 0.0,
        mean_b=128.0 if mean else 0.0,

        shuffle=True,
        rand_crop=True,
        rand_mirror=True,

        # max_rotate_angle=10 if aug else 0,
        # max_aspect_ratio=0.25 if aug else 0,
        # max_shear_ratio=0.1 if aug else 0,

        # max_random_scale=1.0 if aug else 1.0,
        # min_random_scale=0.85 if aug else 1.0,

        # random_h=36 if aug else 0,
        # random_s=40 if aug else 0,
        # random_l=40 if aug else 0,

        prefetch_buffer=8,
        data_shape=data_shape,
        batch_size=batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        preprocess_threads=8)

    return train


def get_data(batch):
    data = np.int32(np.random.rand(20000, 3, 224, 224))
    label = np.random.randint(0, 1000, 20000)
    print(data.shape, label.shape)
    trn = mx.io.NDArrayIter(data=data, label=label, batch_size=batch, shuffle=True)
    out = mx.io.PrefetchingIter(trn)
    return out


def train(net_type, num_gpu=1, batch=BATCH_SIZE):
    # if num_gpu == 2:
    #     devs = [mx.gpu(0), mx.gpu(2)]
    # else:

    devs = []
    for i in range(num_gpu):
        devs.append(mx.gpu(i))

    sym, arg_params, aux_params = mx.model.load_checkpoint('models/%s' % net_type, 0)

    if net_type == 'vgg19':
        internals = sym.get_internals()
        fea_symbol = internals["fc6_output"]

        fc1 = mx.symbol.FullyConnected(data=fea_symbol, num_hidden=2531, name='fc1_1')
        sym = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    model = mx.mod.Module(context=devs, symbol=sym)

    eval_metrics = ['accuracy']
    optimizer_params = {'learning_rate': 0.001, 'momentum': 0.9, 'wd': 0.0001}
    batch_end_callbacks = [mx.callback.Speedometer(batch, 100)]
    checkpoint = mx.callback.do_checkpoint('tmp/-1')

    trn = get_data(batch)

    model.fit(trn,
              begin_epoch=0,
              num_epoch=1,
              eval_metric=eval_metrics,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True)


if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--net', type=str)
    args = parser.parse_args()
    n = args.n
    net = args.net
    print(n, net, 'ololo')
    batch = BATCH_SIZE * n

    logging.info('NET: %s, GPU: %d, BATCH: %d' % (net, n, batch))
    train(net_type=net, num_gpu=n, batch=batch)
