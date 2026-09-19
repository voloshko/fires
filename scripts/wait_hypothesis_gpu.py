"""Wait between independent GPU jobs, without interrupting anyone else's process."""
import subprocess
import time
while True:
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(output.splitlines()[0])>=22000: break
    time.sleep(30)
