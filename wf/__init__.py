import subprocess
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, List

from flytekit.core.annotation import FlyteAnnotation
from latch import large_gpu_task, message, workflow
from latch.resources.launch_plan import LaunchPlan
from latch.types import LatchDir, LatchFile
import shutil
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import string
import re


def _fmt_dir(bucket_path: str) -> str:
    if bucket_path[-1] == "/":
        return bucket_path[:-1]
    return bucket_path


@dataclass_json
@dataclass
class AAChain:
    name: str
    amino_acids: Annotated[
        str,
        FlyteAnnotation(
            {
                "display_name": "Amino Acid Chain",
                "appearance": {
                    "placeholder": "LESPNCDWKNNR...",
                },
                "rules": [
                    {
                        "regex": "^[a-zA-Z].{17,}$",
                        "message": "Does not match 16 or more amino acid single letter codes.",
                    }
                ],
            }
        ),
    ]

whitespace_regex = re.compile(r"\s", re.UNICODE)

@large_gpu_task
def mine_inference_amber(
    fasta_file: Optional[LatchFile],
    aa_sequences: Optional[List[List[AAChain]]],
    run_name: str,
    nrof_models: int,
    output_dir: Optional[LatchDir],
    nrof_recycles: int,
) -> LatchDir:

    if nrof_models < 1:
        message(
            "warning",
            {
                "title": f"Invalid Input",
                "body": "Number of models below 1. Setting to 1",
            },
        )
        nrof_models = 1
    if nrof_models > 5:
        message(
            "warning",
            {
                "title": f"Invalid Input",
                "body": "Number of models greater than 5. Setting to 5",
            },
        )
        nrof_models = 5

    if nrof_recycles < 1:
        message(
            "warning",
            {
                "title": f"Invalid Input",
                "body": "Number of recycles below 1. Setting to 1",
            },
        )
        nrof_recycles = 1
    if nrof_recycles > 50:
        message(
            "warning",
            {
                "title": f"Invalid Input",
                "body": "Number of recycles greater than 50. Setting to 50",
            },
        )
        nrof_recycles = 50

    print("Organizing data", flush=True)
    input_path = Path("/root/sequence.fasta")
    if fasta_file is not None:
        with open(Path(fasta_file), "r") as f:
            with open(input_path, "w") as out:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue

                    if line.startswith(">"):
                        out.write(f"{line}\n")
                        continue

                    cleaned_amino_acids = whitespace_regex.sub("", line)

                    allowed = set(string.ascii_lowercase)
                    chains = cleaned_amino_acids.split(":")
                    for chain in chains:
                        if not set(chain.lower()) <= allowed:
                            message(
                                "error",
                                {
                                    "header": "Invalid amino acid codes in input sequence.",
                                    "body": f"Only single letter codes are allowed. Found: {set(chain.lower()) - allowed}",
                                },
                            )
                            raise ValueError(
                                f"Invalid amino acid codes in input sequence. Found: {set(chain.lower()) - allowed}"
                            )

                        if len(chain) < 16:
                            message(
                                "error",
                                {
                                    "header": "Input chain is too short. Please provide at least 16 amino acids.",
                                    "body": f"Input chain {chain} contains {len(chain)} amino acids",
                                },
                            )
                            raise ValueError(
                                f"Input chain is too short: {len(chain)}/16 amino acids"
                            )

                    out.write(f"{cleaned_amino_acids}\n")
    else:
        if aa_sequences is None:
            raise ValueError(
                "Invariant violation: either fasta_file or aa_sequences must be provided"
            )

        if len(aa_sequences) == 0:
            raise ValueError(
                "Invariant violation: aa_sequences must not be empty when selected"
            )

        with input_path.open("w") as f:
            for protein in aa_sequences:
                names: List[str] = []
                chains: List[str] = []
                for i, chain in enumerate(protein):
                    if chain.name is None:
                        names.append(f"Chain_{i}")
                    else:
                        names.append(f"{chain.name}")

                    cleaned_amino_acids = whitespace_regex.sub("", chain.amino_acids)

                    allowed = set(string.ascii_lowercase)
                    if not set(cleaned_amino_acids.lower()) <= allowed:
                        message(
                            "error",
                            {
                                "header": "Invalid amino acid codes in input sequence.",
                                "body": f"Only single letter codes are allowed. Found: {set(cleaned_amino_acids.lower()) - allowed}",
                            },
                        )
                        raise ValueError(
                            f"Invalid amino acid codes in input sequence. Found: {set(cleaned_amino_acids.lower()) - allowed}"
                        )

                    if len(cleaned_amino_acids) < 16:
                        message(
                            "error",
                            {
                                "header": "Input chain is too short. Please provide at least 16 amino acids.",
                                "body": f"Input chain {chain} contains {len(cleaned_amino_acids)} amino acids",
                            },
                        )
                        raise ValueError(
                            f"Input chain is too short: {len(cleaned_amino_acids)}/16 amino acids"
                        )
                    chains.append(cleaned_amino_acids)

                protein_name = f">{'_'.join(names)}"
                protein_chains = ":".join(chains)
                f.write(f"{protein_name}\n")
                f.write(f"{protein_chains}\n")

    with input_path.open("r") as f:
        print(f.read(), flush=True)

    local_output = Path("/root/preds")
    local_output.mkdir(parents=True, exist_ok=True)

    data_dir = Path("/root/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "colabfold_batch",
        str(input_path),
        "/root/preds",
        "--amber",
        "--use-gpu-relax",
        "--num-models",
        str(nrof_models),
        "--num-recycle",
        str(nrof_recycles),
        "--data",
        "/root/data",
        "--host-url",
        "http://ec2-52-38-163-139.us-west-2.compute.amazonaws.com:80/api",
    ]

    command = " ".join(command)
    process = subprocess.Popen(
        command,
        shell=True,
        errors="replace",
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    while True:
        realtime_output = process.stdout.readline()

        if realtime_output == "" and process.poll() is not None:
            break
        if realtime_output:
            if "RESOURCE_EXHAUSTED" in realtime_output:
                message(
                    "error",
                    {
                        "title": f"Resource Exhausted",
                        "body": "The GPU ran out of memory. Please try again with a smaller input or use AlphaFold2.",
                    },
                )
                print(realtime_output.strip(), flush=True)
                raise RuntimeError("The GPU ran out of memory.")
            if "MMseqs2 API is giving errors" in realtime_output:
                message(
                    "error",
                    {
                        "title": f"MMseqs2 API Error",
                        "body": "The MMseqs2 API is giving errors. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
                    },
                )
                print(realtime_output.strip(), flush=True)
                raise RuntimeError("The MMseqs2 API is giving errors.")
            if "Could not get MSA/templates" in realtime_output:
                message(
                    "error",
                    {
                        "title": f"MMseqs2 Results Parsing Error",
                        "body": "Failed to parse results from MMseqs2. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
                    },
                )
                print(realtime_output.strip(), flush=True)
                raise RuntimeError("MMseqs2 Results Parsing Error")
            if "Could not generate input features" in realtime_output:
                message(
                    "error",
                    {
                        "title": f"No candidates found for sequence.",
                        "body": "No candidates found for sequence. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
                    },
                )
                print(realtime_output.strip(), flush=True)
                raise RuntimeError("Could not generate input features")

            print(realtime_output.strip(), flush=True)

    retval = process.poll()
    if retval != 0:
        raise Exception(f"colabfold_batch failed with exit code {retval}")

    cleaned_output_dir = Path("/root/cleaned_preds")
    pdb_dir = cleaned_output_dir / "pdb results"
    junk_dir = cleaned_output_dir / "other"
    cleaned_output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    for file in local_output.glob("*.pdb"):
        shutil.move(file, pdb_dir)

    shutil.move(local_output, junk_dir)

    if output_dir is not None:
        latch_out_location = _fmt_dir(output_dir.remote_source) + f"/{run_name}"
    else:
        latch_out_location = f"latch:///ColabFold Outputs/{run_name}"

    return LatchDir(
        path=str((cleaned_output_dir).resolve()),
        remote_path=latch_out_location,
    )


@workflow
def colabfold_mmseqs2_wf(
    input_sequence_fork: str = "text",
    output_location_fork: str = "default",
    fasta_file: Optional[
        Annotated[
            LatchFile,
            FlyteAnnotation(
                {
                    "rules": [
                        {
                            "regex": "(.fasta|.fa|.faa|.fas)$",
                            "message": "Only .fasta, .fa, .fas, or .faa extensions are valid",
                        }
                    ],
                }
            ),
        ]
    ] = None,
    aa_sequences: Optional[
        Annotated[
            List[
                Annotated[
                    List[AAChain],
                    FlyteAnnotation({"appearance": {"add_button_title": "Add Chain"}}),
                ]
            ],
            FlyteAnnotation(
                {
                    "appearance": {
                        "add_button_title": "Add Protein",
                    },
                }
            ),
        ]
    ] = None,
    custom_output_dir: Optional[LatchDir] = None,
    nrof_models: int = 1,
    nrof_recycles: int = 3,
    run_name: str = "run1",
) -> LatchDir:
    """The ColabFold version of AlphaFold2 is optimized for extremely fast predictions on small proteins. It uses the same basic architecture as AlphaFold2, but optimizes the sequence search procedure.

    ![header](https://github.com/deepmind/alphafold/raw/main/imgs/header.jpg)

    # ColabFold

    This implementation of ColabFold is a Latch workflow which offloads msa generation (mining for similar sequences) to a dedicated server hosted
    by LatchBio. This server loads and keeps the databases used by the search procedure into RAM for extremely fast searches. The forward pass of
    the neural network and the amber relaxation step are handled by the workflow. This approach, while slightly less accurate
    than [AlphaFold2](console.latch.bio/se/alphafold), is much faster for small proteins and can be run in batched mode,
    batching inferences for multiple similarly sized proteins together. For proteins longer than 1500 amino acids, we recommend using this
    [AlphaFold2](console.latch.bio/se/alphafold) workflow.

    This implementation uses the colabfold environmental database as well as uniref30_2103. We will consider adding more databases in the future
    and it should be noted that the additional template database (pdb70) used by colabfold offers minimal improvements and is thus not implemented
    in the Latch version.

    Acknowledgements and reference help from the [ColabFold README](https://github.com/sokrypton/ColabFold/blob/main/README.md)
    ### Acknowledgments
    - We would like to thank the [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold) and [AlphaFold](https://github.com/deepmind/alphafold) team for doing an excellent job open sourcing the software.
    - Also credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin, without whom these notebooks would be quite boring!
    - A colab by Sergey Ovchinnikov (@sokrypton), Milot Mirdita (@milot_mirdita) and Martin Steinegger (@thesteinegger).

    ### How do I reference this work?

    - Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. ColabFold: Making protein folding accessible to all. <br />
    Nature Methods (2022) doi: [10.1038/s41592-022-01488-1](https://www.nature.com/articles/s41592-022-01488-1)
    - If you’re using **AlphaFold**, please also cite: <br />
    Jumper et al. "Highly accurate protein structure prediction with AlphaFold." <br />
    Nature (2021) doi: [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
    - If you’re using **AlphaFold-multimer**, please also cite: <br />
    Evans et al. "Protein complex prediction with AlphaFold-Multimer." <br />
    biorxiv (2021) doi: [10.1101/2021.10.04.463034v1](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)
    - If you are using **RoseTTAFold**, please also cite: <br />
    Minkyung et al. "Accurate prediction of protein structures and interactions using a three-track neural network." <br />
    Science (2021) doi: [10.1126/science.abj8754](https://doi.org/10.1126/science.abj8754)

    [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.5123296.svg)](https://doi.org/10.5281/zenodo.5123296)

    __metadata__:
        display_name: ColabFold
        author:
            name: Sergey O
            email:
            github: https://github.com/sokrypton
        repository: https://github.com/sokrypton/ColabFold
        license:
            id: Apache-2.0
        batched_template_url: https://latch-public.s3.us-west-2.amazonaws.com/batched_templates/AlphaFold_BatchedTemplateUpdate4.csv
        flow:
        - section: Protein Sequence
          flow:
            - fork: input_sequence_fork
              flows:
                text:
                    display_name: Amino Acids
                    _tmp_unwrap_optionals:
                        - aa_sequences
                    flow:
                        - text: >-
                            Colabfold can be run on a single protein or multiple proteins. Each protein is broken up into a list of chains for ease of use.
                            For each protein, enter each amino acid chain in the sequence in a separate cell. If the protein contains one amino acid chain it will be folded
                            as a monomer. Otherwise, it will be folded as a multimer. The name of each chain will be reflected in the output.
                        - params:
                            - aa_sequences
                file:
                    display_name: File
                    _tmp_unwrap_optionals:
                        - fasta_file
                    flow:
                        - text: >-
                            Enter the file containing the protein(s) you wish to fold. The format is a fasta file, where each protein is a new entry
                            in the file, and each amino acid chain of the protein complex is on the same line, separated by colons. The header lines in the fasta
                            file are the names of the proteins. The format is extremely strict and the workflow will fail unless it is adhered to. Examples can be found
                            [here](https://github.com/latch-verified/colabfold).
                        - params:
                            - fasta_file

        - section: Tuning Parameters
          flow:
            - params:
                - nrof_models
                - nrof_recycles

        - section: Output Settings
          flow:
          - params:
              - run_name
          - fork: output_location_fork
            flows:
                default:
                    display_name: Default
                    flow:
                    - text:
                        Output will be at default location in the data
                        viewer - ColabFold Outputs/"Run Name"
                custom:
                    display_name: Specify Custom Path
                    _tmp_unwrap_optionals:
                        - custom_output_dir
                    flow:
                    - params:
                        - custom_output_dir
    Args:
        fasta_file:
            Path to a FASTA file containing amino acid sequence. DNA and RNA are not supported.

            __metadata__:
                display_name: FASTA File

        aa_sequences:
            Amino acid sequence.

            __metadata__:
                display_name: Protein(s)

        input_sequence_fork:

        output_location_fork:

            __metadata__:
                display_name: Output Location

        custom_output_dir:
            Where you want your output directory to go eg "/colabfold_output"

            __metadata__:
                display_name: Output Directory
                output: true

        run_name:
            Name for the results directory to be created eg "run21" with an output dir of "/colabfold_output" will create the output "/colabfold_output/run21/(colabfold output files)"

            __metadata__:
                display_name: Run Name

        nrof_models:
            Number of models to use (default: 1). Each uses weights trained using a different random seed. The maximum number of models is 5.

            __metadata__:
                display_name: Number of Models

        nrof_recycles:
            Number of prediction cycles. Increasing recycles can improve the quality but slows down the prediction.

            __metadata__:
                display_name: Number of Prediction Cycles
    """

    return mine_inference_amber(
        fasta_file=fasta_file,
        aa_sequences=aa_sequences,
        nrof_models=nrof_models,
        run_name=run_name,
        output_dir=custom_output_dir,
        nrof_recycles=nrof_recycles,
    )


LaunchPlan(
    colabfold_mmseqs2_wf,
    "Monomer",
    {
        "aa_sequences": [
            [
                AAChain(
                    name="Chain_1",
                    amino_acids="MTANHLESPNCDWKNNRMAIVHMVNVTPLRMMEEPRAAVEAAFEGIMEPAVVGDMVEYWNKMISTCCNYYQMGSSRSHLEEKAQMVDRFWFCPCIYYASGKWRNMFLNILHVWGHHHYPRNDLKPCSYLSCKLPDLRIFFNHMQTCCHFVTLLFLTEWPTYMIYNSVDLCPMTIPRRNTCRTMTEVSSWCEPAIPEWWQATVKGGWMSTHTKFCWYPVLDPHHEYAESKMDTYGQCKKGGMVRCYKHKQQVWGNNHNESKAPCDDQPTYLCPPGEVYKGDHISKREAENMTNAWLGEDTHNFMEIMHCTAKMASTHFGSTTIYWAWGGHVRPAATWRVYPMIQEGSHCQC",
                )
            ]
        ],
        "run_name": "test_monomer",
    },
)

LaunchPlan(
    colabfold_mmseqs2_wf,
    "Multimer",
    {
        "aa_sequences": [
            [
                AAChain(
                    name="Chain_1",
                    amino_acids="MAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGD",
                ),
                AAChain(
                    name="Chain_2",
                    amino_acids="MAVKISGVLKDGTGKPVQNVVYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLNAENTVKNIAFNYIVRLA",
                ),
                AAChain(
                    name="Chain_3",
                    amino_acids="MAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQRLA",
                ),
            ]
        ],
        "run_name": "test_multimer",
    },
)
