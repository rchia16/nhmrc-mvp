# -*-coding:utf-8-*-

# this code is currently for python 2.7
from __future__ import print_function
import time
from time import sleep

from RPi import GPIO
from smbus2 import SMBus

# i2c address-es
# not required?
I2C_WRITE_ADDR = 0xAE
I2C_READ_ADDR = 0xAF

INT_PIN = 7     # GPIO 7 for MAX30102 INT pin
# INT_PIN = 11     # GPIO 0 for MAX30102 INT pin

# register address-es
REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01

REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03

REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08

REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C

REG_LED2_PA = 0x0D
REG_PILOT_PA = 0x10
REG_MULTI_LED_CTRL1 = 0x11
REG_MULTI_LED_CTRL2 = 0x12

REG_TEMP_INTR = 0x1F
REG_TEMP_FRAC = 0x20
REG_TEMP_CONFIG = 0x21
REG_PROX_INT_THRESH = 0x30
REG_REV_ID = 0xFE
REG_PART_ID = 0xFF

# currently not used
MAX_BRIGHTNESS = 255


class MAX30102():
    # by default, this assumes that physical pin 7 (GPIO 4) is used as interrupt
    # by default, this assumes that the device is at 0x57 on channel 1
    def __init__(self, channel=1, address=0x57, gpio_pin=INT_PIN, led_mode=0x03):
        print("Channel: {0}, address: 0x{1:x}".format(channel, address))
        self.address = address
        self.channel = channel
        self.bus = SMBus(self.channel)
        self.interrupt = gpio_pin

        # set gpio mode
        GPIO.setmode(GPIO.BOARD)
        # GPIO.setup(INT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.interrupt, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.reset()

        sleep(1)  # wait 1 sec

        # read & clear interrupt register (read 1 byte)
        reg_data = self.bus.read_i2c_block_data(self.address, REG_INTR_STATUS_1, 1)
        # print("[SETUP] reset complete with interrupt register0: {0}".format(reg_data))
        self.setup(led_mode=led_mode)
        # print("[SETUP] setup complete")

    def shutdown(self):
        """
        Shutdown the device.
        """
        self.bus.write_i2c_block_data(self.address, REG_MODE_CONFIG, [0x80])

    def reset(self):
        """
        Reset the device, this will clear all settings,
        so after running this, run setup() again.
        """
        self.bus.write_i2c_block_data(self.address, REG_MODE_CONFIG, [0x40])

    def setup(
        self,
        led_mode=0x03,
        sample_rate=100,
        pulse_width=411,
        adc_range=4096,
        fifo_average=4,
        fifo_rollover=False,
        fifo_a_full=17,
        led1_pa=0x24,
        led2_pa=0x24,
        pilot_pa=0x7f,
    ):
        """
        Configure MAX30102.

        led_mode:
            0x02 = HR only (Red)
            0x03 = SpO2 (Red + IR)
            0x07 = Multi-LED mode

        sample_rate (Hz): 50, 100, 200, 400, 800, 1000, 1600, 3200
        pulse_width (us): 69, 118, 215, 411
        adc_range (nA):   2048, 4096, 8192, 16384
        fifo_average:     1,2,4,8,16,32
        fifo_a_full:      0..31  
            (interrupt when FIFO has this many empty spaces left;
            datasheet wording varies)
        """

        # --------------------
        # Interrupt enables
        # --------------------
        # 0x80: PPG_RDY_EN only (your current behavior)
        # 0xC0: A_FULL + PPG_RDY 
        self.bus.write_i2c_block_data(self.address, REG_INTR_ENABLE_1, [0xC0])
        # self.bus.write_i2c_block_data(self.address, REG_INTR_ENABLE_1, [0x80])
        self.bus.write_i2c_block_data(self.address, REG_INTR_ENABLE_2, [0x00])

        # --------------------
        # FIFO pointers reset
        # --------------------
        self.bus.write_i2c_block_data(self.address, REG_FIFO_WR_PTR, [0x00])
        self.bus.write_i2c_block_data(self.address, REG_OVF_COUNTER, [0x00])
        self.bus.write_i2c_block_data(self.address, REG_FIFO_RD_PTR, [0x00])

        # --------------------
        # FIFO config register (0x08)
        # [7:5] SMP_AVE, [4] FIFO_ROLLOVER_EN, [3:0] FIFO_A_FULL (actually 5 bits per datasheet,
        # but many libs use 4 LSBs. Here we keep behavior compatible with your original 0x4F.)
        # Your original: 0x4F => avg=4, rollover=0, a_full=15
        # --------------------
        ave_map = {1: 0b000, 2: 0b001, 4: 0b010, 8: 0b011, 16: 0b100, 32: 0b101}
        if fifo_average not in ave_map:
            raise ValueError("fifo_average must be one of: 1,2,4,8,16,32")
        smp_ave = ave_map[fifo_average] << 5

        rollover = (1 << 4) if fifo_rollover else 0

        # Keep in 0..15 to match your prior style (0x4F used 0x0F).
        # If you want full 0..31 support, we can rework this to use 5 bits.
        a_full = int(fifo_a_full) & 0x0F

        fifo_cfg = smp_ave | rollover | a_full
        self.bus.write_i2c_block_data(self.address, REG_FIFO_CONFIG, [fifo_cfg])

        # --------------------
        # Mode config (0x09)
        # --------------------
        self.bus.write_i2c_block_data(self.address, REG_MODE_CONFIG, [led_mode])

        # --------------------
        # SPO2 config register (0x0A)
        # [6:5] ADC_RGE, [4:2] SR, [1:0] PW
        # Your original 0x27 => range=4096, SR=100, PW=411us
        # --------------------
        adc_map = {2048: 0b00, 4096: 0b01, 8192: 0b10, 16384: 0b11}
        sr_map = {50: 0b000, 100: 0b001, 200: 0b010, 400: 0b011, 800: 0b100, 1000: 0b101, 1600: 0b110, 3200: 0b111}
        pw_map = {69: 0b00, 118: 0b01, 215: 0b10, 411: 0b11}

        if adc_range not in adc_map:
            raise ValueError("adc_range must be one of: 2048,4096,8192,16384")
        if sample_rate not in sr_map:
            raise ValueError("sample_rate must be one of: 50,100,200,400,800,1000,1600,3200")
        if pulse_width not in pw_map:
            raise ValueError("pulse_width must be one of: 69,118,215,411")

        spo2_cfg = (adc_map[adc_range] << 5) | (sr_map[sample_rate] << 2) | pw_map[pulse_width]
        self.bus.write_i2c_block_data(self.address, REG_SPO2_CONFIG, [spo2_cfg])

        # --------------------
        # LED currents
        # --------------------
        self.bus.write_i2c_block_data(self.address, REG_LED1_PA, [int(led1_pa) & 0xFF])
        self.bus.write_i2c_block_data(self.address, REG_LED2_PA, [int(led2_pa) & 0xFF])
        self.bus.write_i2c_block_data(self.address, REG_PILOT_PA, [int(pilot_pa) & 0xFF])


    # this won't validate the arguments!
    # use when changing the values from default
    def set_config(self, reg, value):
        self.bus.write_i2c_block_data(self.address, reg, value)

    def get_data_present(self):
        read_ptr = self.bus.read_byte_data(self.address, REG_FIFO_RD_PTR)
        write_ptr = self.bus.read_byte_data(self.address, REG_FIFO_WR_PTR)
        if read_ptr == write_ptr:
            return 0
        else:
            num_samples = write_ptr - read_ptr
            # account for pointer wrap around
            if num_samples < 0:
                num_samples += 32
            return num_samples

    def read_fifo(self):
        """
        This function will read the data register.
        """
        red_led = None
        ir_led = None

        # # read 1 byte from registers (values are discarded)
        # reg_INTR1 = self.bus.read_i2c_block_data(self.address, REG_INTR_STATUS_1, 1)
        # reg_INTR2 = self.bus.read_i2c_block_data(self.address, REG_INTR_STATUS_2, 1)

        # read 6-byte data from the device
        d = self.bus.read_i2c_block_data(self.address, REG_FIFO_DATA, 6)

        # mask MSB [23:18]
        red_led = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
        ir_led = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF

        return red_led, ir_led

    def i2c_thread_func(self, max_batch=32, require_ppg_rdy=False):
        """
        Drain FIFO and return a batch of samples: [(ts, red, ir), ...]
        Returns None if nothing is available.

        max_batch: cap number of samples per call to keep callback quick
        require_ppg_rdy: if True, only read when PPG_RDY bit is set
        """
        # Read & clear interrupt status (read clears latched bits)
        status = self.bus.read_i2c_block_data(self.address, REG_INTR_STATUS_1, 1)[0]

        if require_ppg_rdy and not (status & 0x80):
            return None

        n = self.get_data_present()
        if n <= 0:
            return None

        if max_batch is not None:
            n = min(int(n), int(max_batch))

        t0 = time.time()
        out = []
        for _ in range(n):
            red, ir = self.read_fifo()
            out.append((t0, red, ir))

        return out


    def read_sequential(self, amount=None, buffer_add=None, stop_event=None):
        """
        Stream red/IR samples from the sensor.

        Parameters
        ----------
        amount : int or None
            Number of samples to read. If None, read indefinitely until stop_event is set.
        buffer_add : callable or None
            If provided, called as buffer_add((timestamp, ir, red)) for each sample.
            If None, the function behaves like the old implementation and returns
            (red_buf, ir_buf) after reading 'amount' samples.
        stop_event : threading.Event or None
            If provided and amount is None, reading will stop when stop_event.is_set() is True.
        """
        red_buf = []
        ir_buf = []

        # If amount is None, treat as "infinite" until stop_event triggers.
        infinite = amount is None
        count = amount if amount is not None else 0

        while infinite or count > 0:
            # Allow clean exit via stop_event when in infinite mode
            if infinite and stop_event is not None and stop_event.is_set():
                break

            num_samples = self.get_data_present()
            if num_samples == 0:
                # Nothing ready yet; small sleep to avoid busy-waiting
                time.sleep(0.001)
                continue

            while num_samples > 0 and (infinite or count > 0):
                # Timestamp + one sample
                timestamp = time.time()
                red, ir = self.read_fifo()

                if buffer_add is not None:
                    # Write straight into external buffer
                    buffer_add((timestamp, ir, red))
                else:
                    # Legacy mode: accumulate locally
                    red_buf.append(red)
                    ir_buf.append(ir)

                num_samples -= 1
                if not infinite:
                    count -= 1

                # Check stop_event frequently in infinite mode
                if infinite and stop_event is not None and stop_event.is_set():
                    break

        if buffer_add is None:
            # For backward compatibility
            return red_buf, ir_buf

    
    def read_reg(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def dump_regs(self):
        regs = {
            "INTR_EN1": REG_INTR_ENABLE_1,
            "INTR_EN2": REG_INTR_ENABLE_2,
            "FIFO_CFG": REG_FIFO_CONFIG,
            "MODE_CFG": REG_MODE_CONFIG,
            "SPO2_CFG": REG_SPO2_CONFIG,
            "LED1_PA": REG_LED1_PA,
            "LED2_PA": REG_LED2_PA,
            "PART_ID": REG_PART_ID,
            "REV_ID":  REG_REV_ID,
            "FIFO_WR": REG_FIFO_WR_PTR,
            "FIFO_RD": REG_FIFO_RD_PTR,
            "OVF":     REG_OVF_COUNTER,
        }
        out = {k: self.read_reg(v) for k, v in regs.items()}
        return out

