# This example needs SCL virtual accelerator running
# It requires pyepics installed
# In a separate terminal launch VA for MEBT:
# cd ../EPICS/
#  python virtual_accelerator.py --debug --refresh_rate 0.5 --bunch MEBT_in.dat --sequences MEBT

# Reworked (Thomas Bailey) for overhaul of wire scanner in virac.

from epics import caget, caput
from time import sleep

ws = 'MEBT_Diag:WS14'
position = f'{ws}:Position'
command = f'{ws}:Command'
x = f'{ws}:Hor_Cont'
y = f'{ws}:Ver_Cont'
rep_rate = caget(f'{ws}:BeamRepRate')

caput(command, 21)

print(f'{"Position":^12s}  {"x":^8s}  {"y":^8s}')
while caget(command) == 21:
    p = caget(position)
    charge_x = caget(x)
    charge_y = caget(y)
    sleep(rep_rate)
    print(f'{p:12.2f}  {charge_x:8.3f}  {charge_y:8.3f}')
p = caget(position)
charge_x = caget(x)
charge_y = caget(y)
print(f'{p:12.2f}  {charge_x:8.3f}  {charge_y:8.3f}')
print('Done')