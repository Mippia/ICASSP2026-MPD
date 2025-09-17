from ml_models.omnizart.music import app as mapp
from ml_models.omnizart.chord import app as capp
from ml_models.omnizart.drum import app as dapp
from ml_models.omnizart.vocal import app as vapp
from ml_models.omnizart.vocal_contour import app as vcapp
from ml_models.omnizart.beat import app as bapp
import os
import random
import numpy as np
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 줄이기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEED = 0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)


def transcript(mode,wav, model=""):
    """
    Need to Transcript each separated data.
    Note that original omnizart module has own Separation method, but we use separated source, so we modified original code a little.
    """

    if mode.startswith("music"):
        mode_list = mode.split("-")
        mode = mode_list[0]
        model = "-".join(mode_list[1:])
        
    app = {
    "music": mapp,
    "chord": capp,
    "drum": dapp,
    "vocal": vapp,
    "vocal-contour": vcapp,
    "beat": bapp
    }[mode]
    model_path = {
        "piano": "Piano",
        "piano_v2": "PianoV2",
        "assemble": "Stream",
        "pop": "Pop",
        "": None
    }[model]

    filename = os.path.splitext(wav)[0]
    os.makedirs("models/transcription/"+mode+"_transcript/"+filename, exist_ok=True)
    midi = app.transcribe(wav, model_path="ml_models/omnizart/checkpoints/chord/chord_v1",output="models/transcription/"+mode+"_transcript/"+filename)
    return midi


if __name__=='__main__':
    main()