
from . import modules
from pathlib import Path
from . import scheduler
from .interface import Interface
from .modules.transformer import VampNet


__version__ = "0.0.1"

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models" / "vampnet"

from huggingface_hub import hf_hub_download, HfFileSystem
DEFAULT_HF_MODEL_REPO_DIR = ROOT / "DEFAULT_HF_MODEL_REPO"
try:
    DEFAULT_HF_MODEL_REPO = DEFAULT_HF_MODEL_REPO_DIR.read_text().strip()
except FileNotFoundError:
    DEFAULT_HF_MODEL_REPO = "default"  # or whatever default you prefer

FS = HfFileSystem()


def download_codec():
    # from dac.model.dac import DAC
    from lac.model.lac import LAC as DAC
    filename = "codec.pth"

    # First check if codec.pth exists locally in common locations
    possible_paths = [
        Path.cwd() / "models" / "vampnet" / filename,
        MODELS_DIR / filename,
        Path(f"{MODELS_DIR}/{filename}"),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Found local codec at: {path}")
            return str(path)

    # If not found locally, try to download from HuggingFace
    repo_id = DEFAULT_HF_MODEL_REPO
    if repo_id == "default":
        raise ValueError(
            "Cannot download from HuggingFace: DEFAULT_HF_MODEL_REPO is not set to a valid repo. Please place codec.pth in models/vampnet/")

    codec_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=None,
        local_dir=MODELS_DIR
    )
    return codec_path


def download_default():
    filenames = ["coarse.pth", "c2f.pth", "wavebeat.pth"]
    paths = []

    for filename in filenames:
        # Check multiple possible locations
        possible_locations = [
            Path.cwd() / "models" / "vampnet" / filename,
            Path(f"{MODELS_DIR}/{filename}"),
            MODELS_DIR / filename,
        ]

        found = False
        for location in possible_locations:
            if location.exists():
                print(f"Found local {filename} at: {location}")
                paths.append(str(location))
                found = True
                break

        if not found:
            # Try to download if not found locally
            repo_id = DEFAULT_HF_MODEL_REPO
            if repo_id == "default":
                raise ValueError(
                    f"Cannot find {filename} locally and cannot download from HuggingFace: DEFAULT_HF_MODEL_REPO is not set to a valid repo.")

            path = f"{MODELS_DIR}/{filename}"
            print(f"{path} does not exist, downloading")
            FS.download(f"{repo_id}/{filename}", path)
            paths.append(path)

    # load the models - return coarse and c2f paths
    return paths[0], paths[1]


def download_finetuned(name, repo_id=DEFAULT_HF_MODEL_REPO):
    filenames = ["coarse.pth", "c2f.pth"]
    paths = []
    for filename in filenames:
        path = f"{MODELS_DIR}/loras/{name}/{filename}"
        if not Path(path).exists():
            print(f"{path} does not exist, downloading")
            FS.download(f"{repo_id}/loras/{name}/{filename}", path)
        paths.append(path)

    # load the models
    return paths[0], paths[1]


def list_finetuned(repo_id=DEFAULT_HF_MODEL_REPO):
    # If using default/invalid repo, return empty list
    if repo_id == "default":
        print("Skipping HuggingFace model listing (no valid repo configured)")
        return []

    try:
        diritems = FS.listdir(f"{repo_id}/loras")
        # iterate through all the names
        valid_diritems = []
        for item in diritems:
            model_file_items = FS.listdir(item["name"])
            item_names = [item["name"].split("/")[-1] for item in model_file_items]
            # check that theres a "c2f.pth" and "coarse.pth" in the items
            c2f_exists = "c2f.pth" in item_names
            coarse_exists = "coarse.pth" in item_names
            if c2f_exists and coarse_exists:
                valid_diritems.append(item)

        # get the names of the valid items
        names = [item["name"].split("/")[-1] for item in valid_diritems]
        return names
    except Exception as e:
        print(f"Could not list finetuned models from HuggingFace: {e}")
        return []


