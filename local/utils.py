import contextlib
import os

from hydra.utils import get_original_cwd


def run_stage(stage_num, start_stage, stop_stage, skip_stages):
    """Simple helper function to avoid boilerplate code for stages"""
    if skip_stages is None:
        skip_stages = []
    if (
        (start_stage <= stage_num)
        and (stop_stage >= stop_stage)
        and (stage_num not in skip_stages)
    ):
        return True
    else:
        return False


@contextlib.contextmanager
def use_orig_cwd():
    """simple helper context that used the current dir and not the hydra current
    exp dir."""
    exp_dir = os.getcwd()
    orig_dir = get_original_cwd()
    os.chdir(orig_dir)
    yield
    os.chdir(exp_dir)
