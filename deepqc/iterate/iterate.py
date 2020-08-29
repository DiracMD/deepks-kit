import os
import sys
import numpy as np
from deepqc.utils import copy_file, save_yaml
from deepqc.task.workflow import Sequence, Iteration
from deepqc.iterate.template import make_scf, make_train


# args not specified here may cause error
DEFAULT_SCF_MACHINE = {
    "sub_size": 1, # how many systems is put in one task (folder)
    "group_size": 1, # how many tasks are submitted in one job
    "ingroup_parallel": 1, #how many tasks can run at same time in one job
    "dispatcher": None, # use default lazy-local slurm defined in task.py
    "resources": None, # use default 10 core defined in templete.py
    "python": "python" # use current python in path
}

# args not specified here may cause error
DEFAULT_TRN_MACHINE = {
    "dispatcher": None, # use default lazy-local slurm defined in task.py
    "resources": None, # use default 10 core defined in templete.py
    "python": "python" # use current python in path
}

SCF_ARGS_NAME = "scf_input.yaml"
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_TRN_NAME = "init_train.yaml"

DATA_TRAIN = "data_train"
DATA_TEST  = "data_test"
MODEL_FILE = "model.pth"

SCF_STEP_DIR = "00.scf"
TRN_STEP_DIR = "01.train"

RECORD = "RECORD"

DEFAUFT_SYS_TRN = "systems_train.raw"
DEFAUFT_SYS_TST = "systems_test.raw"


def assert_exist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No required file or directory: {path}")


def check_share_folder(data, name, share_folder="share"):
    # save data to share_folder/name. 
    # if data is None, check the existence in share.
    # if data is a file name, copy it to share.
    # if data is a dict, save it as an yaml file in share.
    dst_name = os.path.join(share_folder, name)
    if data is None:
        assert_exist(dst_name)
    elif isinstance(data, str) and os.path.exists(data):
        copy_file(data, dst_name)
    elif isinstance(data, dict):
        save_yaml(data, dst_name)
    else:
        raise ValueError(f"Invalid argument: {data}")


def check_arg_dict(data, default, strict=True):
    allowed = {k:v for k,v in data.items() if k in default}
    outside = {k:v for k,v in data.items() if k not in default}
    if outside:
        print(f"following ars are not in the default list: {list(outside.keys())}"
              +"and would be discarded" if strict else "but kept", file=sys.stderr)
    if strict:
        return {**default, **allowed}
    else:
        return {**default, **data}


def make_iterate(systems_train=None, systems_test=None,
                 niter=5, workdir=".", share_folder="share",
                 scf_args=None, scf_machine=None,
                 train_args=None, train_machine=None,
                 init_model=None, init_scf=None, init_train=None,
                 cleanup=False, strict=True):
    # check share folder contains required data
    if systems_train is None:
        systems_train = os.path.join(share_folder, DEFAUFT_SYS_TRN)
        assert_exist(systems_train) # must have training systems. test can be empty
    if systems_test is None:
        systems_test = os.path.join(share_folder, DEFAUFT_SYS_TST)
    check_share_folder(scf_args, SCF_ARGS_NAME, share_folder)
    check_share_folder(train_args, TRN_ARGS_NAME, share_folder)
    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)
    # make tasks
    scf_step = make_scf(
        systems_train=systems_train, systems_test=systems_test,
        train_dump=DATA_TRAIN, test_dump=DATA_TEST, no_model=False,
        workdir=SCF_STEP_DIR, share_folder=share_folder,
        source_arg=SCF_ARGS_NAME, source_model=MODEL_FILE,
        cleanup=cleanup, **scf_machine
    )
    train_step = make_train(
        source_train=DATA_TRAIN, source_test=DATA_TEST,
        restart=True, source_model=MODEL_FILE, 
        save_model=MODEL_FILE, source_arg=TRN_ARGS_NAME, 
        workdir=TRN_STEP_DIR, share_folder=share_folder,
        cleanup=cleanup, **train_machine
    )
    per_iter = Sequence([scf_step, train_step])
    iterate = Iteration(per_iter, niter, workdir=".", record_file=RECORD)
    # make init
    if init_model: # if set true or give str, check share/init/model.pth
        init_folder=os.path.join(share_folder, "init")
        check_share_folder(init_model, MODEL_FILE, init_folder)
        iterate.set_init_folder(init_folder)
    else:
        check_share_folder(scf_args, INIT_SCF_NAME, share_folder)
        check_share_folder(train_args, INIT_TRN_NAME, share_folder)
        scf_init = make_scf(
        systems_train=systems_train, systems_test=systems_test,
        train_dump=DATA_TRAIN, test_dump=DATA_TEST, no_model=True,
        workdir=SCF_STEP_DIR, share_folder=share_folder,
        source_arg=INIT_SCF_NAME, source_model=MODEL_FILE,
        cleanup=cleanup, **scf_machine
        )
        train_init = make_train(
            source_train=DATA_TRAIN, source_test=DATA_TEST,
            restart=False, source_model=MODEL_FILE, 
            save_model=MODEL_FILE, source_arg=INIT_TRN_NAME, 
            workdir=TRN_STEP_DIR, share_folder=share_folder,
            cleanup=cleanup, **train_machine
        )
        init_iter = Sequence([scf_init, train_init], workdir="iter.init")
        iterate.prepend(init_iter)
    return iterate
