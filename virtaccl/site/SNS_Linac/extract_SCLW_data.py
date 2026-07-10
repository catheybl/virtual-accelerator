import json
import numpy as np
from pathlib import Path
import os

json_out = Path("va_rf_config.json")
bpm_path = Path("sclwbpmoutput.dat")
cav_path = Path("sclwcavoutput.dat")

if os.path.exists(json_out):
    print(f"File {json_out} already exists. overwrite? (y,n)")
    answer = input()
    if not answer.lower() == "y":
        print("Exiting...")
        exit()

bpm, offset_deg = np.genfromtxt(
    bpm_path,
    comments="#",
    usecols=(1, 3),
    dtype=None,
    unpack=True,
)

cav,init_amp, init_phase,offset_phase = np.genfromtxt(
    cav_path,
    comments="#",
    usecols=(1, 5, 6, 8),
    dtype=None,
    unpack=True,
)
with open(json_out, "w") as f:
    bpm_dict = dict(zip(bpm, offset_deg))
    cav_dict = {f"{cav[0:3]}{cav[6:]}": {"init_amp": ia,
                     "init_phase": ip,
                     "offset_phase": op}
                for cav,ia,ip,op in zip(cav,init_amp,init_phase,offset_phase)}
    out = {
        "BPM": bpm_dict,
        "RF_Cavity": cav_dict,
    }
    json.dump(out, f,indent = 2)
print("Done!")
