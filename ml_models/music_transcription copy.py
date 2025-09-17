from ml_models.omnizart.music import app as mapp
from ml_models.omnizart.chord import app as capp
from ml_models.omnizart.drum import app as dapp
from ml_models.omnizart.vocal import app as vapp
from ml_models.omnizart.vocal_contour import app as vcapp
from ml_models.omnizart.beat import app as bapp
import os

def transcript(mode,wav):
    """
    Need to Transcript each separated data.
    Note that original omnizart module has own Separation method, but we use separated source, so we modified original code a little.
    """

    model = ""
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
    midi = app.transcribe(wav, model_path=model_path,output="models/transcription/"+mode+"_transcript/"+filename)
    return midi


if __name__=='__main__':
    main()