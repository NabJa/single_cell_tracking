"""
Train ResNet23.

Requires tensorflow.__version__ >= 2.0
"""

import argparse
from datetime import datetime
from pathlib import Path
import keras
import numpy as np
import pickle


def train(inp, out, img_shape, validation, batch_size):

    # Import moved to function for faster --help
    from src.mlt_detection.resnet23 import ResNet23
    from src.mlt_detection.data.dataset import Dataset

    inp, out = Path(inp), Path(out)

    logdir = out / str("logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(logdir))

    resnet23 = ResNet23(image_shape=img_shape, training=True)
    data = Dataset(str(inp))

    train_ids, val_ids = data.split_data(validation)

    history = resnet23.train(train=data.generate_train_data(batch_size=1),
                             val=data.generate_val_data(batch_size=1),
                             steps_per_epoch=100,
                             validation_steps=10,
                             epochs=1,
                             callbacks=[tensorboard_callback])

    # Save model and meta data
    resnet23.save_model(str(out.joinpath("model.h5")))
    np.save(str(out.joinpath("train_ids.npy")), train_ids)
    np.save(str(out.joinpath("val_ids.npy")), val_ids)
    pickle.dump(history.history, open(str(out.joinpath("history.p")), "wb"))


# TODO support training on TFRecord data
def _tf_record_path(x):
    x = Path(x)
    if not x.is_file():
        raise FileNotFoundError("Not a file: ", x)
    elif not (x.suffix == ".tfrecord"):
        raise ValueError(f"Given a {x.suffix} file. A .tfrecord file is needed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="REQUIRED Input path.")
    parser.add_argument("--output", required=True,
                        help="REQUIRED Output path.")
    parser.add_argument("--img_shape", type=tuple, default=(224, 224, 3),
                        help="OPTIONAL Input image shape. Third channel is added if not provided. "
                             "DEFAULT=(224, 224, 3)")
    parser.add_argument("--validation", default=0.2,
                        help="OPTIONAL Validation dataset name or fraction of train data. "
                             "DEFAULT=0.2 (20 percent of train data)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="OPTIONAL Batch size. DEFAULT=1")

    args = parser.parse_args()

    train(args.input, args.output, args.img_shape, args.validation, args.batch_size)
