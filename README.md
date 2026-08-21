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

## Raspberry Pi LSL Sensors

The current Raspberry Pi sensor streamer uses separate buses:

- BNO085 IMU over SPI
- MAX30102/PPG over I2C

Enable SPI on the Raspberry Pi before running the streamer:

```bash
sudo raspi-config
# Interface Options -> SPI -> Enable, then reboot if prompted
```

### BNO085 SPI Wiring

The BNO085 must be strapped into SPI mode before boot/reset:

```text
BNO085 P0  -> 3.3V
BNO085 P1  -> 3.3V
```

Wire the IMU SPI bus as:

```text
BNO085 VIN -> Pi 3.3V, physical pin 1
BNO085 GND -> Pi GND, physical pin 6
BNO085 SCL -> Pi SCLK, physical pin 23
BNO085 SDA -> Pi MISO, physical pin 21
BNO085 DI  -> Pi MOSI, physical pin 19
BNO085 CS  -> Pi CE0, physical pin 24
BNO085 INT -> Pi BCM23 / D23, physical pin 16
BNO085 RST -> Pi BCM24 / D24, physical pin 18
```

The matching `streaming_config.yaml` settings are:

```yaml
bno085:
  transport: spi
  spi:
    cs_pin: CE0
    int_pin: D23
    baudrate: 1000000
  reset_pin: D24
```

### PPG I2C Wiring

Wire the PPG sensor on the Raspberry Pi I2C bus:

```text
PPG SDA -> Pi SDA1 / BCM2, physical pin 3
PPG SCL -> Pi SCL1 / BCM3, physical pin 5
PPG INT -> Pi BCM25 / D25, physical pin 22
```

The current streamer polls the MAX30102 FIFO over I2C and does not use the PPG interrupt pin. It can remain connected.

Check that the PPG is visible on I2C:

```bash
i2cdetect -y 1
```

Expected PPG address:

```text
0x57
```

### Running

Test the BNO085 over SPI by itself:

```bash
python bno085_lsl_streamer.py --transport spi --spi-cs-pin CE0 --spi-int-pin D23 --reset-pin D24 --reports accelerometer --rate-hz 250
```

Run the combined BNO085 IMU + PPG LSL streamer:

```bash
python rpi_lsl_imu_ppg.py --config ./streaming_config.yaml --rate-print --diagnostics
```

The streamer prints the configured rate for each enabled IMU report and for the PPG sensor during startup.
## Authors and Acknowledgement

Adapted from
[vrano714/max30102-tutorial-raspberry-pi](https://github.com/vrano714/max30102-tutorial-raspberrypi/blob/master/max30102.py)
