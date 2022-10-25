import logging
import tarfile
from pathlib import Path

import appdirs
import tqdm
import subprocess

logger = logging.getLogger(__name__)

# The data dir location logic switches between a version with and one without "params" because alphafold
# always internally joins "params". (We should probably patch alphafold)
default_data_dir = Path(appdirs.user_cache_dir(__package__ or "colabfold"))


def download_alphafold_params(model_type: str, data_dir: Path = default_data_dir):
    import requests

    params_dir = data_dir.joinpath("params")
    if model_type == "AlphaFold2-multimer-v2":
        weights_name = "alphafold_params_colab_2022-03-02.tar"
        success_marker = params_dir.joinpath(
            "download_complexes_multimer-v2_finished.txt"
        )
    elif model_type == "AlphaFold2-multimer-v1":
        weights_name = "alphafold_params_colab_2021-10-27.tar"
        success_marker = params_dir.joinpath(
            "download_complexes_multimer-v1_finished.txt"
        )
    else:
        weights_name = "alphafold_params_2021-07-14.tar"
        success_marker = params_dir.joinpath("download_finished.txt")

    if success_marker.is_file():
        return

    params_dir.mkdir(parents=True, exist_ok=True)
    weights_command = [
        "aria2c",
        f"http://storage.googleapis.com/alphafold/{weights_name}",
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
    download_alphafold_params("AlphaFold2-multimer-v2")
    download_alphafold_params("AlphaFold2-ptm")
