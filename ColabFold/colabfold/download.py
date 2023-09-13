import logging
import tarfile
from pathlib import Path

import subprocess
import appdirs

logger = logging.getLogger(__name__)

# The data dir location logic switches between a version with and one without "params" because alphafold
# always internally joins "params". (We should probably patch alphafold)
default_data_dir = Path(appdirs.user_cache_dir(__package__ or "colabfold"))


def download_alphafold_params(model_type: str, data_dir: Path = default_data_dir):
    params_dir = data_dir.joinpath("params")
    if model_type == "alphafold2_multimer_v3":
        weights_name = "alphafold_params_colab_2022-12-06.tar"
        success_marker = params_dir.joinpath(
            "download_complexes_multimer_v3_finished.txt"
        )
    elif model_type == "alphafold2_multimer_v2":
        raise NotImplementedError("Old multimer weights are unavailable -- contact support@latch.bio if needed")
    elif model_type == "alphafold2_multimer_v1":
        raise NotImplementedError("Old multimer weights are unavailable -- contact support@latch.bio if needed")
    else:
        weights_name = "alphafold_params_2021-07-14.tar"
        success_marker = params_dir.joinpath("download_finished.txt")

    if success_marker.is_file():
        return

    params_dir.mkdir(parents=True, exist_ok=True)

    params_dir.mkdir(parents=True, exist_ok=True)
    weights_command = [
        "aria2c",
        "--max-connection-per-server=16",
        f"https://latch-public.s3.us-west-2.amazonaws.com/colabfold/{weights_name}",
        "-d",
        str(params_dir),
    ]
    rval = subprocess.run(weights_command, check=True)
    if rval.returncode != 0:
        raise Exception("Error downloading alphafold weights")

    with tarfile.open(params_dir / weights_name, "r") as tar:
        tar.extractall(params_dir)

    (params_dir / weights_name).unlink()
    success_marker.touch()


if __name__ == "__main__":
    # TODO: Arg to select which one
    download_alphafold_params("alphafold2_multimer_v3")
    download_alphafold_params("alphafold2_ptm")
