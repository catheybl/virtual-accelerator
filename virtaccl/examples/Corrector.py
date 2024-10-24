# This example needs SCL virtual accelerator running
# It requires pyepics installed
# In a separate terminal launch VA:
# cd ../EPICS/
# python virtual_accelerator.py --debug

from epics import caget, caput
from time import sleep
corrector = "SCL_Mag:DCH00:B"
corrector_set = "SCL_Mag:PS_DCH00:B_Set"
bpm = "SCL_Diag:BPM04:xAvg"

original_val = caget(corrector_set)

for i in range(5):
    caput(corrector_set, i/50)
    sleep(1.5)
    print(f'Corrector value: {caget(corrector)}')
    print(f'BPM value: {caget(bpm)}')

caput(corrector_set, original_val)
