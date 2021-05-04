# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import pickle

from fairmotion.data import amass_dip
from fairmotion.ops import conversions
from fairmotion.ops import motion as motion_ops
from fairmotion.tasks.motion_prediction import utils
from fairmotion.utils import utils as fairmotion_utils
from fairmotion.ops import conversions
import random

from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def split_into_windows(motion, window_size, stride, drop_on, threshold=4, drop_perc=0.0):
    """
    Split motion object into list of motions with length window_size with
    the given stride.
    """
    n_windows = (motion.num_frames() - window_size) // stride + 1
    motion_ws = [
        motion_ops.cut(motion, start, start + window_size)
        for start in stride * np.arange(n_windows)
    ]
    metadata = tuple()
    if drop_on == "vel":
        # create idx here because when list is deleted
        # the idx should shift
        i = 0
        n_motion_ws = len(motion_ws)
        for motion_obj in motion_ws:
            seq_deleted = False
            aa = conversions.R2A(motion_obj.rotations())
            for j in range(len(aa)):
                # skip 0th index where velocity cannot be computed
                if j == 0:
                    continue
                # element wise subtract
                vel = np.subtract(aa[j], aa[j-1])
                n_crosses = len(vel[np.where(vel < -threshold)]) + len(vel[np.where(vel > threshold)])
                if n_crosses:
                    del motion_ws[i]
                    seq_deleted = True
                    break
            # if item in list is deleted, counter should not move
            # because the next element takes the place of the deleted element
            if not seq_deleted:
                i += 1

        frames_deleted = n_motion_ws - len(motion_ws)
        if frames_deleted:
            logging.info("{} frame sequences deleted during filtering".format(frames_deleted))
    return  [m for m in motion_ws if random.random() > drop_perc]


def process_file(ftuple, create_windows, convert_fn, lengths, drop_on, drop_perc=0.0):
    src_len, tgt_len = lengths
    filepath, file_id = ftuple
    motion = amass_dip.load(filepath)
    motion.name = file_id
    metadata = None
    if create_windows is not None:
        window_size, window_stride = create_windows
        if motion.num_frames() < window_size:
            return [], []
        matrices = list()
        motions, metadata = split_into_windows(motion, window_size, window_stride, drop_on, drop_perc=drop_perc)
        for motion in motions:
            matrices.append(convert_fn(motion.rotations()))
    else:
        matrices = [convert_fn(motion.rotations())]

    return (
        [matrix[:src_len, ...].reshape((src_len, -1)) for matrix in matrices],
        [
            matrix[src_len : src_len + tgt_len, ...].reshape((tgt_len, -1))
            for matrix in matrices
        ],
        metadata,
    )

def process_split(
    all_fnames, output_path, rep, src_len, tgt_len, create_windows=None, drop_on=None, drop_perc=0.0
):
    """
    Process data into numpy arrays.

    Args:
        all_fnames: List of filenames that should be processed.
        output_path: Where to store numpy files.
        rep: If the output data should be rotation matrices, quaternions or
            axis angle.
        create_windows: Tuple (size, stride) of windows that should be
            extracted from each sequence or None otherwise.

    Returns:
        Some meta statistics (how many sequences processed etc.).
    """
    assert rep in ["aa", "rotmat", "quat"]
    convert_fn = utils.convert_fn_from_R(rep)
    logging.info("Paralleling Processes")
    data = fairmotion_utils.run_parallel(
        process_file,
        all_fnames,
        num_cpus=8,
        create_windows=create_windows,
        convert_fn=convert_fn,
        lengths=(src_len, tgt_len),
        drop_on=drop_on,
        drop_perc=drop_perc
    )
    logging.info("Paralleling Complete")
    src_seqs, tgt_seqs = [], []
    n_frame_seq_total = 0
    n_frame_seq_deleted = 0
    for worker_data in tqdm(data, ascii=True, desc="Processing Data"):
        s, t, metadata = worker_data
        if metadata:
            seq_total, seq_deleted = metadata
            n_frame_seq_total += seq_total
            n_frame_seq_deleted += seq_deleted
        src_seqs.extend(s)
        tgt_seqs.extend(t)
    logging.info(f"Processed {len(src_seqs)} sequences")
    if n_frame_seq_deleted:
        logging.info("n_seqs_total {}, n_seqs_deleted: {}".format(n_frame_seq_total, n_frame_seq_deleted))
        logging.info("pct deleted {}".format(n_frame_seq_deleted / n_frame_seq_total * 100))
    pickle.dump((src_seqs, tgt_seqs), open(output_path, "wb"))


def read_content(filepath):
    content = []
    with open(filepath) as f:
        for line in f:
            content.append(line.strip())
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Location of the downloaded and unpacked zip file. See "
        "https://amass.is.tue.mpg.de/dataset for dataset",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to store pickle files."
    )
    parser.add_argument(
        "--split-dir",
        default="./data",
        help="Where the text files defining the data splits are stored.",
    )
    parser.add_argument(
        "--rep",
        type=str,
        help="Angle representation to convert data to",
        choices=["aa", "quat", "rotmat"],
        default="aa",
    )
    parser.add_argument(
        "--src-len",
        type=int,
        default=120,
        help="Number of frames fed as input motion to the model",
    )
    parser.add_argument(
        "--tgt-len",
        type=int,
        default=24,
        help="Number of frames to be predicted by the model",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=180,
        help="Window size for test and validation, in frames.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=120,
        help="Window stride for test and validation, in frames. This is also"
        " used as training window size",
    )
    parser.add_argument(
        "--drop-on",
        type=str,
        default="",
        )
    parser.add_argument(
        "--drop-percent",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    train_files = read_content(
        os.path.join(args.split_dir, "training_fnames.txt")
    )
    validation_files = read_content(
        os.path.join(args.split_dir, "validation_fnames.txt")
    )
    test_files = read_content(os.path.join(args.split_dir, "test_fnames.txt"))

    train_ftuples = []
    test_ftuples = []
    validation_ftuples = []
    for filepath in tqdm(fairmotion_utils.files_in_dir(args.input_dir, ext="pkl"), ascii=True, desc="Sourcing Files"):
        db_name = os.path.split(os.path.dirname(filepath))[1]
        db_name = (
            "_".join(db_name.split("_")[1:])
            if "AMASS" in db_name
            else db_name.split("_")[0]
        )
        f = os.path.basename(filepath)
        file_id = "{}/{}".format(db_name, f)

        if file_id in train_files:
            train_ftuples.append((filepath, file_id))
        elif file_id in validation_files:
            validation_ftuples.append((filepath, file_id))
        elif file_id in test_files:
            test_ftuples.append((filepath, file_id))
        else:
            pass

    output_path = os.path.join(args.output_dir, args.rep)
    fairmotion_utils.create_dir_if_absent(output_path)

    logging.info("Processing training data...")
    if args.drop_on == "vel":
        logging.info("Dropping frame seqences on velocity")
    else:
        logging.info("DropOn set off - Perserving all frames")
    train_dataset = process_split(
        train_ftuples,
        os.path.join(output_path, "train.pkl"),
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
        drop_on=args.drop_on,
        drop_perc=args.drop_percent
    )

    logging.info("Processing validation data...")
    process_split(
        validation_ftuples,
        os.path.join(output_path, "validation.pkl"),
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )

    logging.info("Processing test data...")
    process_split(
        test_ftuples,
        os.path.join(output_path, "test.pkl"),
        rep=args.rep,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        create_windows=(args.window_size, args.window_stride),
    )
