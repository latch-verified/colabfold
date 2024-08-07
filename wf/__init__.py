import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from flytekit.core.annotation import FlyteAnnotation
from latch import large_gpu_task, message, workflow
from latch.resources.launch_plan import LaunchPlan
from latch.types import LatchDir, LatchFile


def _fmt_dir(bucket_path: str) -> str:
    if bucket_path[-1] == "/":
        return bucket_path[:-1]
    return bucket_path


def handle_error(out: str):
    if "RESOURCE_EXHAUSTED" in out:
        message(
            "error",
            {
                "title": "Resource Exhausted",
                "body": "The GPU ran out of memory. Please try again with a smaller input or use AlphaFold2.",
            },
        )
    elif "MMseqs2 API is giving errors" in out:
        message(
            "error",
            {
                "title": "MMseqs2 API Error",
                "body": "The MMseqs2 API is giving errors. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
            },
        )
    elif "Could not get MSA/templates" in out:
        message(
            "error",
            {
                "title": "MMseqs2 Results Parsing Error",
                "body": "Failed to parse results from MMseqs2. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
            },
        )
    elif "Could not generate input features" in out:
        message(
            "error",
            {
                "title": "No candidates found for sequence.",
                "body": "No candidates found for sequence. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
            },
        )
    else:
        return

    raise RuntimeError(f"colabfold_batch failed with error {out}")

@large_gpu_task
def mine_inference_amber(
    fasta_file: Optional[LatchFile],
    aa_sequence: Optional[str],
    run_name: str,
    nrof_models: int,
    output_dir: Optional[LatchDir],
    nrof_recycles: int,
    template_dir: Optional[LatchDir],
) -> LatchDir:

    if nrof_models < 1:
        message(
            "warning",
            {
                "title": "Invalid Input",
                "body": "Number of models below 1. Setting to 1",
            },
        )
        nrof_models = 1
    if nrof_models > 5:
        message(
            "warning",
            {
                "title": "Invalid Input",
                "body": "Number of models greater than 5. Setting to 5",
            },
        )
        nrof_models = 5

    if nrof_recycles < 1:
        message(
            "warning",
            {
                "title": "Invalid Input",
                "body": "Number of recycles below 1. Setting to 1",
            },
        )
        nrof_recycles = 1
    if nrof_recycles > 50:
        message(
            "warning",
            {
                "title": "Invalid Input",
                "body": "Number of recycles greater than 50. Setting to 50",
            },
        )
        nrof_recycles = 50

    print("Organizing data", flush=True)
    input_path = Path("/sequence.fasta")
    if fasta_file is not None:
        with open(Path(fasta_file), "r") as f:
            with open(input_path, "w") as out:
                for line in f:
                    if line.strip() != "":
                        out.write(f"{line.strip()}\n")
    else:
        if aa_sequence is None:
            raise ValueError(
                "Invariant violation: either fasta_file or aa_sequence must be provided"
            )

        with input_path.open("w") as f:
            broken = aa_sequence.split("\n")
            broken = [x for x in broken if x.strip() != ""]
            for l in broken:
                if " " in l and not l.startswith(">"):
                    message(
                        "error",
                        "Spaces in input sequence are not allowed. Please format multimers using colon separation.",
                    )
                    raise ValueError("Spaces in fasta file are not allowed.")
            if broken[0].startswith(">"):
                f.write("\n".join(broken))
            else:
                for i, line in enumerate(broken):
                    f.write(f">sequence_{i}\n{line}\n")

    with input_path.open("r") as f:
        nrof_lines = sum(1 for _ in f)
        if nrof_lines == 0:
            message(
                "error",
                {
                    "title": "Empty Input",
                    "body": "No sequences were found in the input.",
                },
            )
            raise RuntimeError("No sequences were found in the input.")
        if nrof_lines % 2 != 0:
            message(
                "error",
                {
                    "title": "Invalid Input",
                    "body": "Input contains an odd number of lines indicating an unpaired line",
                },
            )
            raise RuntimeError(
                "Input contains an odd number of lines indicating an unpaired line"
            )

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

    if template_dir is not None:
        local_template_dir = Path(template_dir)
        command.extend(
            ["--templates", "--custom-template-path", str(local_template_dir)]
        )
        print(f"Path to templates: {str(local_template_dir)}", flush=True)

    command = " ".join(command)
    process = subprocess.Popen(
        command,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    while True:
        if process.stdout is None:
            break

        realtime_output = process.stdout.readline().decode("utf-8").strip()

        if realtime_output == "" and process.poll() is not None:
            break

        if realtime_output:
            handle_error(realtime_output)
            print(realtime_output, flush=True)

    raw_out, raw_err = process.communicate()

    if raw_out is not None:
        raw_out = raw_out.decode("utf-8")
        handle_error(raw_out)

    if raw_err is not None:
        raw_err = raw_err.decode("utf-8")
        handle_error(raw_err)

    retval = process.poll()
    if retval != 0:
        raise RuntimeError(f"colabfold_batch failed with error {raw_err if raw_err is not None else 'unknown'}")

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
    aa_sequence: Optional[
        Annotated[
            str,
            FlyteAnnotation(
                {
                    "appearance": {
                        "type": "paragraph",
                        "placeholder": ">SequenceOne\nLESPNCDWKNNR:RLENKNNCSPDW:CDWKNNENPDEA",
                    },
                    "rules": [
                        {
                            "regex": "^((>[^\n]+\n[A-Z]+(:[A-Z]+)*)(\n>.+\n[A-Z]+(:[A-Z]+)*)*|([A-Z]+(:[A-Z]+)*)(\n[A-Z]+(:[A-Z]+)*)*)$",
                            "message": "Error: provide a list of named amino acid sequences, each formatted over two lines. If names are not inputted, the names sequence_1, sequence_2, etc will be provided. Ensure that there are no spaces or newlines in a single sequence. The name line must start with `>` and the sequence line can only contain capital letters. For folding multimers, separate sequences using a colon.",
                        }
                    ],
                }
            ),
        ]
    ] = None,
    custom_output_dir: Optional[LatchDir] = None,
    nrof_models: int = 1,
    nrof_recycles: int = 3,
    template_dir: Optional[LatchDir] = None,
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
        - section: Amino Acid Sequence
          flow:
            - text: >-
                Enter the file containing the amino acid sequence (or sequences) you wish to fold. Alternatively,
                you can enter the amino acid sequence directly. Each line of the input fasta represents a single
                protein to be folded. The header line must start with `>` and the sequence line can only contain
                capital letters. Or, the header line can be entirely omitted in which case the entires will be
                labelled by index. For folding multimers, separate sequences under a single header with colon.
                For example, `MTA...ANH:CDW...RMA:ESP...CDW` would represent a multimer with three chains.
            - fork: input_sequence_fork
              flows:
                text:
                    display_name: Text
                    _tmp_unwrap_optionals:
                        - aa_sequence
                    flow:
                        - params:
                            - aa_sequence
                file:
                    display_name: File
                    _tmp_unwrap_optionals:
                        - fasta_file
                    flow:
                        - params:
                            - fasta_file

        - section: Tuning Parameters
          flow:
            - params:
                - nrof_models
                - nrof_recycles
                - template_dir

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

        aa_sequence:
            Amino acid sequence.

            __metadata__:
                display_name: Amino Acid Sequence(s)

        input_sequence_fork:

            __metadata__:
                display_name: Input Sequence

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

        template_dir:
            Directory with custom pdb files to be used as templates for the inference.

            __metadata__:
                display_name: Template Directory
    """

    return mine_inference_amber(
        fasta_file=fasta_file,
        aa_sequence=aa_sequence,
        nrof_models=nrof_models,
        run_name=run_name,
        output_dir=custom_output_dir,
        nrof_recycles=nrof_recycles,
        template_dir=template_dir,
    )


LaunchPlan(
    colabfold_mmseqs2_wf,
    "Monomer",
    {
        "aa_sequence": ">TEST_SEQUENCE\nMTANHLESPNCDWKNNRMAIVHMVNVTPLRMMEEPRAAVEAAFEGIMEPAVVGDMVEYWNKMISTCCNYYQMGSSRSHLEEKAQMVDRFWFCPCIYYASGKWRNMFLNILHVWGHHHYPRNDLKPCSYLSCKLPDLRIFFNHMQTCCHFVTLLFLTEWPTYMIYNSVDLCPMTIPRRNTCRTMTEVSSWCEPAIPEWWQATVKGGWMSTHTKFCWYPVLDPHHEYAESKMDTYGQCKKGGMVRCYKHKQQVWGNNHNESKAPCDDQPTYLCPPGEVYKGDHISKREAENMTNAWLGEDTHNFMEIMHCTAKMASTHFGSTTIYWAWGGHVRPAATWRVYPMIQEGSHCQC",
        "run_name": "test_monomer",
    },
)

LaunchPlan(
    colabfold_mmseqs2_wf,
    "Multimer",
    {
        "aa_sequence": ">Two_Chain\nMAVKISGVLKDGTGKPVQNCATIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGDASASAAQVAALVTDATDSARAASTSAGQAASSAQEASSGAEAASAKATEAEKSAAAAESSKNAAATSAGAAKTSETNAAASQQSAATSASTAATKASEAATSARDAVASKEAAKSSETNASSSAGRAASSATAAENSARAAKTSETNARSSETAAERSASAAADAKTAAAGSASTASTKATEAAGSAVSASQSKSAAEAAAIRAKNSAKRAEDIASAVALEDADTTRKGIVQLSSATNSTSETLAATPKAVKVVMDETNRKAPLDSPALTGTPTAPTALRGTNNTQIANTAFVLAAIADVIDASPDALNTLNELAAALGNDPDFATTMTNALAGKQPKNATLTALAGLSTAKNKLPYFAENDAASLTELTQVGRDILAKNSVADVLEYLGAGENSAFPAGAPIPWPSDIVPSGYVLMQGQAFDKSAYPKLAVAYPSGVLPDMRGWTIKGKPASGRAVLSQEQDGIKSHTHSASASGTDLGTKTTSSFDYGTKTTGSFDYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLSKTDSQGSHSHSLSGTAVSAGAHAHTVGIGAHQHPVVIGAHAHSFSIGSHGHTITVNAAGNAENTVKNIAFNYIVRLA:MAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGDASASAAQVAALVTDATDSARAASTSAGQAASSAQEASSGAEAASAKATEAEKSAAAAESSKNAAATSAGAAKTSETNAAASQQSAATSASTAATKASEAATSARDAVASKEAAKSSETNASSSAGRAASSATAAENSARAAKTSETNARSSETAAERSASAAADAKTAAAGSASTASTKATEAAGSAVSASQSKSAAEAAAIRAKNSAKRAEDIASAVALEDADTTRKGIVQLSSATNSTSETLAATPKAVKVVMDETNRKAPLDSPALTGTPTAPTALRGTNNTQIANTAFVLAAIADVIDASPDALNTLNELAAALGNDPDFATTMTNALAGKQPKNATLTALAGLSTAKNKLPYFAENDAASLTELTQVGRDILAKNSVADVLEYLGAGENSAFPAGAPIPWPSDIVPSGYVLMQGQAFDKSAYPKLAVAYPSGVLPDMRGWTIKGKPASGRAVLSQEQDGIKSHTHSASASGTDLGTKTTSSFDYGTKTTGSFDYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLSKTDSQGSHSHSLSGTAVSAGAHAHTVGIGAHQHPVVIGAHAHSFSIGSHGHTITVNAAGNAENTVKNIAFNYIVRLA:MAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGDASASAAQVAALVTDATDSARAASTSAGQAASSAQEASSGAEAASAKATEAEKSAAAAESSKNAAATSAGAAKTSETNAAASQQSAATSASTAATKASEAATSARDAVASKEAAKSSETNASSSAGRAASSATAAENSARAAKTSETNARSSETAAERSASAAADAKTAAAGSASTASTKATEAAGSAVSASQSKSAAEAAAIRAKNSAKRAEDIASAVALEDADTTRKGIVQLSSATNSTSETLAATPKAVKVVMDETNRKAPLDSPALTGTPTAPTALRGTNNTQIANTAFVLAAIADVIDASPDALNTLNELAAALGNDPDFATTMTNALAGKQPKNATLTALAGLSTAKNKLPYFAENDAASLTELTQVGRDILAKNSVADVLEYLGAGENSAFPAGAPIPWPSDIVPSGYVLMQGQAFDKSAYPKLAVAYPSGVLPDMRGWTIKGKPASGRAVLSQEQDGIKSHTHSASASGTDLGTKTTSSFDYGTKTTGSFDYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLSKTDSQGSHSHSLSGTAVSAGAHAHTVGIGAHQHPVVIGAHAHSFSIGSHGHTITVNAAGNAENTVKNIAFNYIVRLA",
        "run_name": "test_multimer",
    },
)
