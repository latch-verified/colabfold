import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Annotated
from latch import large_gpu_task, workflow, message
from latch.resources.launch_plan import LaunchPlan
from latch.types import LatchFile, LatchDir
from flytekit.core.annotation import FlyteAnnotation


def _fmt_dir(bucket_path: str) -> str:
    if bucket_path[-1] == "/":
        return bucket_path[:-1]
    return bucket_path


class NROFModels(Enum):
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"


@large_gpu_task
def mine_inference_amber(
    fasta_file: Optional[LatchFile],
    aa_sequence: Optional[str],
    run_name: str,
    nrof_models: NROFModels,
    output_dir: Optional[LatchDir],
) -> LatchDir:

    print("Organizing data", flush=True)
    input_path = Path("/sequence.fasta")
    if fasta_file is not None:
        with open(Path(fasta_file), "r") as f:
            with open(input_path, "w") as out:
                for line in f:
                    if line.strip() != "":
                        out.write(f"{line.strip()}\n")
    else:
        with input_path.open("w") as f:
            f.write(aa_sequence)

    with input_path.open("r") as f:
        nrof_lines = sum(1 for _ in f)
        if nrof_lines == 0:
            message(
                "error",
                {
                    "title": f"Empty Input",
                    "body": "No sequences were found in the input.",
                },
            )
            raise RuntimeError("No sequences were found in the input.")
        if nrof_lines % 2 != 0:
            message(
                "error",
                {
                    "title": f"Invalid Input",
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
        str(nrof_models.value),
        "--data",
        "/root/data",
        "--host-url",
        "http://ec2-35-167-25-158.us-west-2.compute.amazonaws.com:80/api",
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
                raise RuntimeError("The GPU ran out of memory.")
            if "MMseqs2 API is giving errors" in realtime_output:
                message(
                    "error",
                    {
                        "title": f"MMseqs2 API Error",
                        "body": "The MMseqs2 API is giving errors. Please contact support@latch.bio or message us via the chat in the bottom of the left sidebar.",
                    },
                )
                raise RuntimeError("The MMseqs2 API is giving errors.")
            print(realtime_output.strip(), flush=True)

    retval = process.poll()
    if retval != 0:
        raise Exception(f"colabfold_batch failed with exit code {retval}")

    if output_dir is not None:
        latch_out_location = _fmt_dir(output_dir.remote_source) + f"/{run_name}"
    else:
        latch_out_location = f"latch:///ColabFold Outputs/{run_name}"

    return LatchDir(
        str((local_output).resolve()),
        remote_directory=latch_out_location,
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
                            "regex": "^(>[^\n]+\n[A-Z]+(:[A-Z]+)*)(\n>.+\n[A-Z]+(:[A-Z]+)*)*$",
                            "message": "Must be a list of named amino acid sequences, each formatted over two lines. The name line must start with `>` and the sequence line can only contain capital letters. For folding multimers, separate sequences under a single header with colons",
                        }
                    ],
                }
            ),
        ]
    ] = None,
    custom_output_dir: Optional[LatchDir] = None,
    nrof_models: NROFModels = NROFModels.one,
    run_name: str = "run1"
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
            Number of models to use (default: 1). Each uses weights trained using a different random seed.

            __metadata__:
                display_name: Number of Models

    """

    return mine_inference_amber(
        fasta_file=fasta_file,
        aa_sequence=aa_sequence,
        nrof_models=nrof_models,
        run_name=run_name,
        output_dir=custom_output_dir,
    )

LaunchPlan(
    colabfold_mmseqs2_wf,
    "Monomer",
    {
        "aa_sequence": ">TEST_SEQUENCE\nMTANHLESPNCDWKNNRMAIVHMVNVTPLRMMEEPRAAVEAAFEGIMEPAVVGDMVEYWNKMISTCCNYYQMGSSRSHLEEKAQMVDRFWFCPCIYYASGKWRNMFLNILHVWGHHHYPRNDLKPCSYLSCKLPDLRIFFNHMQTCCHFVTLLFLTEWPTYMIYNSVDLCPMTIPRRNTCRTMTEVSSWCEPAIPEWWQATVKGGWMSTHTKFCWYPVLDPHHEYAESKMDTYGQCKKGGMVRCYKHKQQVWGNNHNESKAPCDDQPTYLCPPGEVYKGDHISKREAENMTNAWLGEDTHNFMEIMHCTAKMASTHFGSTTIYWAWGGHVRPAATWRVYPMIQEGSHCQC",
        "run_name": "test_monomer",
    }
)

LaunchPlan(
    colabfold_mmseqs2_wf,
    "Multimer",
    {
        "aa_sequence": ">Lambda_1_Lambda_1\nMAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGDASASAAQVAALVTDATDSARAASTSAGQAASSAQEASSGAEAASAKATEAEKSAAAAESSKNAAATSAGAAKTSETNAAASQQSAATSASTAATKASEAATSARDAVASKEAAKSSETNASSSAGRAASSATAAENSARAAKTSETNARSSETAAERSASAAADAKTAAAGSASTASTKATEAAGSAVSASQSKSAAEAAAIRAKNSAKRAEDIASAVALEDADTTRKGIVQLSSATNSTSETLAATPKAVKVVMDETNRKAPLDSPALTGTPTAPTALRGTNNTQIANTAFVLAAIADVIDASPDALNTLNELAAALGNDPDFATTMTNALAGKQPKNATLTALAGLSTAKNKLPYFAENDAASLTELTQVGRDILAKNSVADVLEYLGAGENSAFPAGAPIPWPSDIVPSGYVLMQGQAFDKSAYPKLAVAYPSGVLPDMRGWTIKGKPASGRAVLSQEQDGIKSHTHSASASGTDLGTKTTSSFDYGTKTTGSFDYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLSKTDSQGSHSHSLSGTAVSAGAHAHTVGIGAHQHPVVIGAHAHSFSIGSHGHTITVNAAGNAENTVKNIAFNYIVRLA:MAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGDASASAAQVAALVTDATDSARAASTSAGQAASSAQEASSGAEAASAKATEAEKSAAAAESSKNAAATSAGAAKTSETNAAASQQSAATSASTAATKASEAATSARDAVASKEAAKSSETNASSSAGRAASSATAAENSARAAKTSETNARSSETAAERSASAAADAKTAAAGSASTASTKATEAAGSAVSASQSKSAAEAAAIRAKNSAKRAEDIASAVALEDADTTRKGIVQLSSATNSTSETLAATPKAVKVVMDETNRKAPLDSPALTGTPTAPTALRGTNNTQIANTAFVLAAIADVIDASPDALNTLNELAAALGNDPDFATTMTNALAGKQPKNATLTALAGLSTAKNKLPYFAENDAASLTELTQVGRDILAKNSVADVLEYLGAGENSAFPAGAPIPWPSDIVPSGYVLMQGQAFDKSAYPKLAVAYPSGVLPDMRGWTIKGKPASGRAVLSQEQDGIKSHTHSASASGTDLGTKTTSSFDYGTKTTGSFDYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLSKTDSQGSHSHSLSGTAVSAGAHAHTVGIGAHQHPVVIGAHAHSFSIGSHGHTITVNAAGNAENTVKNIAFNYIVRLA:MAVKISGVLKDGTGKPVQNCTIQLKARRNSTTVVVNTVGSENPDEAGRYSMDVEYGQYSVILQVDGFPPSHAGTITVYEDSQPGTLNDFLCAMTEDDARPEVLRRLELMVEEVARNASVVAQSTADAKKSAGDASASAAQVAALVTDATDSARAASTSAGQAASSAQEASSGAEAASAKATEAEKSAAAAESSKNAAATSAGAAKTSETNAAASQQSAATSASTAATKASEAATSARDAVASKEAAKSSETNASSSAGRAASSATAAENSARAAKTSETNARSSETAAERSASAAADAKTAAAGSASTASTKATEAAGSAVSASQSKSAAEAAAIRAKNSAKRAEDIASAVALEDADTTRKGIVQLSSATNSTSETLAATPKAVKVVMDETNRKAPLDSPALTGTPTAPTALRGTNNTQIANTAFVLAAIADVIDASPDALNTLNELAAALGNDPDFATTMTNALAGKQPKNATLTALAGLSTAKNKLPYFAENDAASLTELTQVGRDILAKNSVADVLEYLGAGENSAFPAGAPIPWPSDIVPSGYVLMQGQAFDKSAYPKLAVAYPSGVLPDMRGWTIKGKPASGRAVLSQEQDGIKSHTHSASASGTDLGTKTTSSFDYGTKTTGSFDYGTKSTNNTGAHAHSLSGSTGAAGAHAHTSGLRMNSSGWSQYGTATITGSLSTVKGTSTQGIAYLSKTDSQGSHSHSLSGTAVSAGAHAHTVGIGAHQHPVVIGAHAHSFSIGSHGHTITVNAAGNAENTVKNIAFNYIVRLA",
        "run_name": "test_multimer",
    }
)
