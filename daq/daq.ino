/*
  bno085_usb_stream.ino

  Arduino Mega 2560 + BNO085
  --------------------------
  Streams 6-axis IMU data over USB serial for Raspberry Pi / LSL ingestion.

  BNO085:
    I2C address: 0x4A
    SDA: Mega pin 20
    SCL: Mega pin 21
    INT: Mega pin 4
    P0:  GND
    P1:  GND

  Sampling:
    Accelerometer: 200 Hz
    Gyroscope:     200 Hz

  I2C:
    400 kHz
    (This was stable for the BNO085 when tested on its own.)

  USB serial:
    1,000,000 baud

  Output format:
    I,t_us,accel_seq,gyro_seq,ax,ay,az,gx,gy,gz

  Example:
    I,123456789,42,42,0.02148,-0.01855,9.79688,0.001953,0.000000,-0.001953

  Diagnostic/status lines begin with '#'.

  Compatible with the previously created rpi_arduino_lsl.py parser.

  Library:
    Adafruit BNO08x
*/

#include <Wire.h>
#include <Adafruit_BNO08x.h>


// ============================================================================
// Configuration
// ============================================================================

#define BNO_ADDR        0x4A
#define BNO_INT_PIN     4

#define I2C_RATE_HZ     400000UL

#define IMU_RATE_HZ     200UL
#define IMU_PERIOD_US   (1000000UL / IMU_RATE_HZ)  // 5000 us

#define SERIAL_BAUD     1000000UL


// ============================================================================
// BNO085
// ============================================================================

// -1 means the library does not control a hardware reset pin.
Adafruit_BNO08x bno(-1);
sh2_SensorValue_t sensorValue;


// ============================================================================
// Latest samples
// ============================================================================

float ax = 0.0f;
float ay = 0.0f;
float az = 0.0f;

float gx = 0.0f;
float gy = 0.0f;
float gz = 0.0f;

uint8_t accelSeq = 0;
uint8_t gyroSeq = 0;

bool haveAccel = false;
bool haveGyro = false;

bool newAccel = false;
bool newGyro = false;


// ============================================================================
// Diagnostics
// ============================================================================

uint32_t accelEventsTotal = 0;
uint32_t gyroEventsTotal = 0;
uint32_t imuRecordsTotal = 0;

uint32_t accelDropsTotal = 0;
uint32_t gyroDropsTotal = 0;

uint32_t wireTimeoutTotal = 0;

uint32_t lastAccelEvents = 0;
uint32_t lastGyroEvents = 0;
uint32_t lastImuRecords = 0;

uint32_t lastStatsMs = 0;

bool accelSeqValid = false;
bool gyroSeqValid = false;

uint8_t previousAccelSeq = 0;
uint8_t previousGyroSeq = 0;


// ============================================================================
// 64-bit Arduino clock
//
// micros() on AVR is 32-bit and wraps after ~71.6 minutes.
// This extends it into a monotonically increasing uint64_t clock.
// ============================================================================

uint64_t micros64()
{
  static uint32_t previous = 0;
  static uint64_t upper = 0;

  uint32_t now = micros();

  if (now < previous)
  {
    upper += (1ULL << 32);
  }

  previous = now;

  return upper | (uint64_t)now;
}


// AVR Print has no uint64_t overload.
void printUint64(uint64_t value)
{
  char buffer[21];
  char *p = &buffer[20];

  *p = '\0';

  do
  {
    *--p = char('0' + (value % 10));
    value /= 10;
  }
  while (value != 0);

  Serial.print(p);
}


// ============================================================================
// I2C helpers
// ============================================================================

uint8_t probeAddress(uint8_t address)
{
  Wire.beginTransmission(address);
  return Wire.endTransmission();
}


void checkWireTimeout()
{
  if (Wire.getWireTimeoutFlag())
  {
    wireTimeoutTotal++;
    Wire.clearWireTimeoutFlag();
  }
}


// ============================================================================
// Sequence-gap tracking
// ============================================================================

void updateAccelSequence(uint8_t seq)
{
  if (accelSeqValid)
  {
    uint8_t expected = (uint8_t)(previousAccelSeq + 1);
    uint8_t gap = (uint8_t)(seq - expected);

    // A small forward gap indicates dropped reports.
    // Ignore very large values as likely reset/restart discontinuities.
    if (gap > 0 && gap < 128)
    {
      accelDropsTotal += gap;
    }
  }

  previousAccelSeq = seq;
  accelSeqValid = true;
}


void updateGyroSequence(uint8_t seq)
{
  if (gyroSeqValid)
  {
    uint8_t expected = (uint8_t)(previousGyroSeq + 1);
    uint8_t gap = (uint8_t)(seq - expected);

    if (gap > 0 && gap < 128)
    {
      gyroDropsTotal += gap;
    }
  }

  previousGyroSeq = seq;
  gyroSeqValid = true;
}


// ============================================================================
// Stream one paired 6-axis record
// ============================================================================

void sendIMURecord()
{
  uint64_t tUs = micros64();

  Serial.print(F("I,"));
  printUint64(tUs);

  Serial.print(',');
  Serial.print(accelSeq);

  Serial.print(',');
  Serial.print(gyroSeq);

  Serial.print(',');
  Serial.print(ax, 5);

  Serial.print(',');
  Serial.print(ay, 5);

  Serial.print(',');
  Serial.print(az, 5);

  Serial.print(',');
  Serial.print(gx, 6);

  Serial.print(',');
  Serial.print(gy, 6);

  Serial.print(',');
  Serial.println(gz, 6);

  imuRecordsTotal++;
}


// ============================================================================
// Configure BNO reports
// ============================================================================

bool configureBNO()
{
  Serial.println(F("# Configuring BNO085 reports"));

  bool accelOK = bno.enableReport(
    SH2_ACCELEROMETER,
    IMU_PERIOD_US
  );

  Serial.print(F("# ACCEL 200Hz="));
  Serial.println(accelOK ? F("OK") : F("FAIL"));

  checkWireTimeout();

  delay(10);

  bool gyroOK = bno.enableReport(
    SH2_GYROSCOPE_CALIBRATED,
    IMU_PERIOD_US
  );

  Serial.print(F("# GYRO 200Hz="));
  Serial.println(gyroOK ? F("OK") : F("FAIL"));

  checkWireTimeout();

  return accelOK && gyroOK;
}


// ============================================================================
// Service BNO085
// ============================================================================

void serviceBNO()
{
  // BNO085 INT is active LOW.
  if (digitalRead(BNO_INT_PIN) != LOW)
  {
    return;
  }

  // Drain a bounded number of pending SH2 packets each loop.
  // This prevents a pathological stuck-low INT from blocking everything.
  for (uint8_t n = 0; n < 16; n++)
  {
    if (digitalRead(BNO_INT_PIN) == HIGH)
    {
      break;
    }

    bool gotEvent = bno.getSensorEvent(&sensorValue);

    checkWireTimeout();

    if (!gotEvent)
    {
      // SH2 may have consumed a control/timestamp packet rather than
      // returning a sensor sample.
      continue;
    }

    switch (sensorValue.sensorId)
    {
      case SH2_ACCELEROMETER:
      {
        ax = sensorValue.un.accelerometer.x;
        ay = sensorValue.un.accelerometer.y;
        az = sensorValue.un.accelerometer.z;

        accelSeq = sensorValue.sequence;

        updateAccelSequence(accelSeq);

        haveAccel = true;
        newAccel = true;

        accelEventsTotal++;

        break;
      }


      case SH2_GYROSCOPE_CALIBRATED:
      {
        gx = sensorValue.un.gyroscope.x;
        gy = sensorValue.un.gyroscope.y;
        gz = sensorValue.un.gyroscope.z;

        gyroSeq = sensorValue.sequence;

        updateGyroSequence(gyroSeq);

        haveGyro = true;
        newGyro = true;

        gyroEventsTotal++;

        break;
      }


      default:
        break;
    }

    // At equal requested rates, emit one record after receiving
    // a fresh sample from each physical sensor stream.
    if (haveAccel && haveGyro && newAccel && newGyro)
    {
      sendIMURecord();

      newAccel = false;
      newGyro = false;
    }
  }
}


// ============================================================================
// Once-per-second status
// ============================================================================

void printStats()
{
  uint32_t nowMs = millis();

  if ((uint32_t)(nowMs - lastStatsMs) < 1000UL)
  {
    return;
  }

  lastStatsMs = nowMs;

  uint32_t accelHz = accelEventsTotal - lastAccelEvents;
  uint32_t gyroHz = gyroEventsTotal - lastGyroEvents;
  uint32_t imuHz = imuRecordsTotal - lastImuRecords;

  lastAccelEvents = accelEventsTotal;
  lastGyroEvents = gyroEventsTotal;
  lastImuRecords = imuRecordsTotal;

  Serial.print(F("# STAT A="));
  Serial.print(accelHz);

  Serial.print(F(" G="));
  Serial.print(gyroHz);

  Serial.print(F(" I="));
  Serial.print(imuHz);

  Serial.print(F(" dropA="));
  Serial.print(accelDropsTotal);

  Serial.print(F(" dropG="));
  Serial.print(gyroDropsTotal);

  Serial.print(F(" timeout="));
  Serial.println(wireTimeoutTotal);
}


// ============================================================================
// Setup
// ============================================================================

void setup()
{
  Serial.begin(SERIAL_BAUD);

  delay(1000);

  Serial.println();
  Serial.println(F("# ====================================="));
  Serial.println(F("# BNO085 USB STREAM"));
  Serial.println(F("# ====================================="));
  Serial.println(F("# I2C=400000"));
  Serial.println(F("# ACCEL=200Hz"));
  Serial.println(F("# GYRO=200Hz"));
  Serial.println(F("# SERIAL=1000000"));

  pinMode(BNO_INT_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(I2C_RATE_HZ);

  // Avoid a permanent AVR Wire lockup.
  Wire.setWireTimeout(500000UL, true);
  Wire.clearWireTimeoutFlag();

  delay(100);

  // --------------------------------------------------------------------------
  // Address probe
  // --------------------------------------------------------------------------

  uint8_t ack = probeAddress(BNO_ADDR);

  Serial.print(F("# BNO 0x4A ACK="));
  Serial.println(ack);

  if (ack != 0)
  {
    Serial.println(F("# FATAL: BNO085 did not ACK"));
    while (1)
    {
      delay(1000);
    }
  }


  // --------------------------------------------------------------------------
  // BNO initialization
  // --------------------------------------------------------------------------

  Serial.println(F("# Starting BNO085"));

  if (!bno.begin_I2C(BNO_ADDR, &Wire))
  {
    Serial.println(F("# FATAL: begin_I2C failed"));
    while (1)
    {
      delay(1000);
    }
  }

  Serial.println(F("# BNO085 OK"));

  // Consume the expected startup/reset indication generated during init.
  (void)bno.wasReset();

  // Restore known-good bus configuration after library initialization.
  Wire.setClock(I2C_RATE_HZ);
  Wire.setWireTimeout(500000UL, true);
  Wire.clearWireTimeoutFlag();

  delay(100);

  if (!configureBNO())
  {
    Serial.println(F("# FATAL: report configuration failed"));
    while (1)
    {
      delay(1000);
    }
  }

  lastStatsMs = millis();

  Serial.println(F("# STREAMING"));
}


// ============================================================================
// Main loop
// ============================================================================

void loop()
{
  serviceBNO();

  checkWireTimeout();

  printStats();
}