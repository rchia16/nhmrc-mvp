# nhmrc-mvp



## Requirements

If running on the raspberry pi, must setup with C-lang SWIG code generator and use
lgpio
See [this link](https://abyz.me.uk/lg/download.html)

Setup up miniconda3 and use the rpi.yml.
```
conda env create -f path/to/rpi.yml
conda activate rpi
```

ssh to raspberry pi with password `raspberry`:
```
ssh nhmrc@172.19.123.253
```

## Authors and Acknowledgement

Adapted from
[vrano714/max30102-tutorial-raspberry-pi](https://github.com/vrano714/max30102-tutorial-raspberrypi/blob/master/max30102.py)
