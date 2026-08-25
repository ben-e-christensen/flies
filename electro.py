import os, time
from datetime import datetime
from pathlib import Path
import numpy as np
import serial
import matplotlib.pyplot as plt
# ============ CONFIG ============
ROOT_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'
ROOT_FOLDER.mkdir(exist_ok=True)
DELAY_MS    = 5
DURATION_S  = 15       # None = run until Ctrl+C
PREFACTOR   = 1e12      # C -> pC
SERIAL_PORT = 'COM19'
BAUDRATE    = 9600
# ================================
SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG 20e-9; "
             b":SENS:CHAR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")
stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filepath = os.path.join(ROOT_FOLDER, f'electrometer_{stamp}.csv')
t_vec, q_vec = [], []
with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=5) as em, open(filepath, 'w') as f:
    f.write('time,charge\n')
    em.write(SETUP_CMD)
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if DURATION_S is not None and t >= DURATION_S:
                break
            em.write(b'READ?\r')
            raw = em.readline()
            print(repr(raw))          # <-- debug, remove later
            try:
                q = float(raw)
            except ValueError:
                q = np.nan
            t_vec.append(t)
            q_vec.append(q)
            f.write(f'{t},{q}\n')
            f.flush()
            time.sleep(DELAY_MS / 1000)
    except KeyboardInterrupt:
        pass
print(f'Recording stopped. Saved to {filepath}')
plt.plot(t_vec, np.array(q_vec) * PREFACTOR, 'ro-')
plt.xlabel('time [s]')
plt.ylabel('charge [pC]')
plt.show()