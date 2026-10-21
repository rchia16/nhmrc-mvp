#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <MAX30105.h>
#include "hardware/timer.h"

// ============================================================
// Raspberry Pi Pico / Pico 2
// Combined BNO085 + MAX30102 acquisition
//
// Arduino-Pico core (Earle Philhower)
// BNO085:   I2C0 / Wire  -> GP4 SDA, GP5 SCL
// MAX30102: I2C1 / Wire1 -> GP6 SDA, GP7 SCL
//
// Optional wired pins:
// BNO085 RST -> GP8
// MAX30102 INT -> GP9  (reserved, not used in this polling build)
// BNO085 INT -> GP13   (reserved, Adafruit I2C API does not use it)
//
// USB Serial -> Raspberry Pi
//
// Protocol:
//   Same fixed 42-byte Protocol V2 frame as the Mega build.
//   Adds packet_type 3 = PPG, packet_type 4 = PPG stats.
// ============================================================


// ============================================================
// PIN ASSIGNMENT
// ============================================================

constexpr uint8_t BNO_SDA_PIN   = 4;
constexpr uint8_t BNO_SCL_PIN   = 5;
constexpr uint8_t PPG_SDA_PIN   = 6;
constexpr uint8_t PPG_SCL_PIN   = 7;
constexpr uint8_t BNO_RESET_PIN = 1; // previously 8
constexpr uint8_t PPG_INT_PIN   = 9;   // reserved for later
constexpr uint8_t BNO_INT_PIN   = 0;  // reserved for later, previously 13

constexpr uint8_t BNO_I2C_ADDRESS = 0x4A;
constexpr uint8_t PPG_I2C_ADDRESS = 0x57;


// ============================================================
// SENSOR OBJECTS
// ============================================================

Adafruit_BNO08x bno(BNO_RESET_PIN);
sh2_SensorValue_t sensorValue;

MAX30105 ppg;


// ============================================================
// SENSOR CONFIG
// ============================================================

// BNO085 requested report intervals
constexpr uint32_t ACCEL_INTERVAL_US = 5000;   // request 200 Hz
constexpr uint32_t GYRO_INTERVAL_US  = 5000;   // request 200 Hz
constexpr uint32_t QUAT_INTERVAL_US  = 10000;  // request 100 Hz

// MAX30102
constexpr uint32_t PPG_SAMPLE_RATE_HZ = 200;
constexpr uint64_t PPG_INTERVAL_US =
    1000000ULL / PPG_SAMPLE_RATE_HZ;           // 5000 us

constexpr uint8_t PPG_LED_BRIGHTNESS = 0x24;  // match prior Pi setup
constexpr uint8_t PPG_SAMPLE_AVERAGE = 1;
constexpr uint8_t PPG_LED_MODE = 2;            // Red + IR
constexpr uint16_t PPG_PULSE_WIDTH_US = 118;
constexpr uint16_t PPG_ADC_RANGE = 4096;

// Poll hardware FIFO more often than the 5 ms sample interval.
constexpr uint32_t PPG_POLL_INTERVAL_US = 1000;

// If the PPG sample clock is separated from the last sample by more
// than this, treat it as a real interruption and re-anchor.
constexpr uint64_t PPG_REANCHOR_US = 250000ULL;

// Gentle phase correction applied to the nominal 5 ms PPG clock.
// This compensates for sensor oscillator error without allowing
// FIFO polling jitter to dominate timestamps.
constexpr int64_t PPG_PHASE_DIVISOR = 50;      // 2% correction
constexpr int64_t PPG_MAX_CORRECTION_US = 20;


// ============================================================
// PROTOCOL V2
// ============================================================

constexpr uint8_t MAGIC0 = 0xA5;
constexpr uint8_t MAGIC1 = 0x5A;
constexpr uint8_t PROTOCOL_VERSION = 2;

enum PacketType : uint8_t {
  PKT_IMU       = 1,
  PKT_STATS     = 2,
  PKT_PPG       = 3,
  PKT_PPG_STATS = 4
};

enum ImuType : uint8_t {
  IMU_ACCEL = 1,
  IMU_GYRO  = 2,
  IMU_QUAT  = 3
};

enum PpgType : uint8_t {
  PPG_RED_IR = 1
};


// ============================================================
// FIXED 42-BYTE SERIAL FRAME
// ============================================================
//
// Offset  Size
//   0      1    magic0
//   1      1    magic1
//   2      1    version
//   3      1    packet_type
//   4      4    packet_sequence
//   8      8    sample_timestamp_us  [Pico common 64-bit clock]
//  16      4    timestamp_low32_us   [backward-compatible low 32 bits]
//  20      1    sensor_sequence
//  21      1    sensor_type
//  22      1    sensor_status
//  23      1    n_values
//  24     16    payload
//  40      2    CRC16-CCITT
//
// IMPORTANT:
// In this Pico build the 64-bit timestamp field is intentionally a
// common Pico clock timestamp for BOTH IMU and PPG. We no longer use
// the unreliable SH2 timestamp as the synchronization clock.
// ============================================================

struct __attribute__((packed)) SerialPacketV2 {
  uint8_t magic0;
  uint8_t magic1;
  uint8_t version;
  uint8_t packet_type;

  uint32_t packet_sequence;

  uint64_t sensor_timestamp_us;
  uint32_t host_timestamp_us;

  uint8_t sensor_sequence;
  uint8_t sensor_type;
  uint8_t sensor_status;
  uint8_t n_values;

  union {
    float f[4];
    uint32_t u32[4];
  } payload;

  uint16_t crc;
};

static_assert(sizeof(SerialPacketV2) == 42,
              "SerialPacketV2 must remain exactly 42 bytes");


// ============================================================
// INTERNAL EVENTS
// ============================================================

struct __attribute__((packed)) ImuEvent {
  uint64_t sample_timestamp_us;
  uint8_t sensor_sequence;
  uint8_t sensor_type;
  uint8_t sensor_status;
  uint8_t n_values;
  float v[4];
};

struct __attribute__((packed)) PpgEvent {
  uint64_t sample_timestamp_us;
  uint32_t red;
  uint32_t ir;
  uint32_t sample_index;
};


// ============================================================
// EXPLICIT PROTOTYPES
// ============================================================
//
// Arduino IDE normally auto-generates prototypes for .ino files.
// Explicit declarations here keep custom ImuEvent / PpgEvent types
// visible and avoid prototype-generation issues on some IDE/core versions.
// ============================================================

uint16_t imuQueueDepth();
bool imuBufferEmpty();
bool imuBufferPush(const ImuEvent &event);
bool imuBufferPeek(ImuEvent &event);
void imuBufferRemove();

uint16_t ppgQueueDepth();
bool ppgBufferEmpty();
bool ppgBufferPush(const PpgEvent &event);
bool ppgBufferPeek(PpgEvent &event);
void ppgBufferRemove();

uint16_t crc16_ccitt(const uint8_t *data, size_t length);

bool setBnoReports();
bool beginBnoWithRetry(uint8_t attempts, uint32_t retryDelayMs);
bool setBnoReportsWithRetry(uint8_t attempts, uint32_t retryDelayMs);
void serviceBNO();
bool readPpgFifoSample(uint32_t &red, uint32_t &ir);
void servicePPG();

void makeImuPacket(const ImuEvent &event, SerialPacketV2 &packet);
void makePpgPacket(const PpgEvent &event, SerialPacketV2 &packet);
void serviceSerialTx();

void makeImuStatsPacket(SerialPacketV2 &packet);
void makePpgStatsPacket(SerialPacketV2 &packet);
void serviceStats();
void serviceHeartbeat();
void serviceHostCommands();
void startAcquisition();
void stopAcquisition();
void discardBufferedSamples();



// ============================================================
// RING BUFFERS
// ============================================================
//
// Pico has substantially more RAM than the Mega, so keep useful
// transient headroom while still making all operations O(1).
// ============================================================

constexpr uint16_t IMU_BUFFER_SIZE = 128;
constexpr uint16_t IMU_BUFFER_MASK = IMU_BUFFER_SIZE - 1;

constexpr uint16_t PPG_BUFFER_SIZE = 256;
constexpr uint16_t PPG_BUFFER_MASK = PPG_BUFFER_SIZE - 1;

static_assert((IMU_BUFFER_SIZE & (IMU_BUFFER_SIZE - 1)) == 0,
              "IMU buffer size must be power of two");

static_assert((PPG_BUFFER_SIZE & (PPG_BUFFER_SIZE - 1)) == 0,
              "PPG buffer size must be power of two");

ImuEvent imuBuffer[IMU_BUFFER_SIZE];
PpgEvent ppgBuffer[PPG_BUFFER_SIZE];

uint16_t imuHead = 0;
uint16_t imuTail = 0;
uint16_t ppgHead = 0;
uint16_t ppgTail = 0;


// ============================================================
// DIAGNOSTICS
// ============================================================

uint32_t packetSequence = 0;

uint32_t imuReceived = 0;
uint32_t imuTransmitted = 0;
uint32_t imuOverflow = 0;
uint32_t unknownSensorEvents = 0;
uint32_t bnoResetCount = 0;
uint32_t bnoReportConfigFailures = 0;
bool bnoReportsNeedConfig = false;
uint32_t lastBnoReportRetryMs = 0;
uint16_t maxImuQueueDepth = 0;

uint32_t ppgReceived = 0;
uint32_t ppgTransmitted = 0;
uint32_t ppgOverflow = 0;
uint32_t ppgSampleIndex = 0;
uint32_t ppgReanchorCount = 0;
uint32_t ppgReadErrors = 0;
uint16_t maxPpgQueueDepth = 0;

uint64_t lastPpgSampleUs = 0;
uint64_t lastPpgPollUs = 0;

bool acquisitionActive = false;


// ============================================================
// RING HELPERS
// ============================================================

uint16_t imuQueueDepth() {
  return (imuHead - imuTail) & IMU_BUFFER_MASK;
}

bool imuBufferEmpty() {
  return imuHead == imuTail;
}

bool imuBufferPush(const ImuEvent &event) {
  const uint16_t next = (imuHead + 1) & IMU_BUFFER_MASK;

  if (next == imuTail) {
    imuOverflow++;
    return false; // drop newest
  }

  imuBuffer[imuHead] = event;
  imuHead = next;

  const uint16_t depth = imuQueueDepth();
  if (depth > maxImuQueueDepth) {
    maxImuQueueDepth = depth;
  }

  return true;
}

bool imuBufferPeek(ImuEvent &event) {
  if (imuBufferEmpty()) {
    return false;
  }

  event = imuBuffer[imuTail];
  return true;
}

void imuBufferRemove() {
  if (!imuBufferEmpty()) {
    imuTail = (imuTail + 1) & IMU_BUFFER_MASK;
  }
}


uint16_t ppgQueueDepth() {
  return (ppgHead - ppgTail) & PPG_BUFFER_MASK;
}

bool ppgBufferEmpty() {
  return ppgHead == ppgTail;
}

bool ppgBufferPush(const PpgEvent &event) {
  const uint16_t next = (ppgHead + 1) & PPG_BUFFER_MASK;

  if (next == ppgTail) {
    ppgOverflow++;
    return false; // drop newest
  }

  ppgBuffer[ppgHead] = event;
  ppgHead = next;

  const uint16_t depth = ppgQueueDepth();
  if (depth > maxPpgQueueDepth) {
    maxPpgQueueDepth = depth;
  }

  return true;
}

bool ppgBufferPeek(PpgEvent &event) {
  if (ppgBufferEmpty()) {
    return false;
  }

  event = ppgBuffer[ppgTail];
  return true;
}

void ppgBufferRemove() {
  if (!ppgBufferEmpty()) {
    ppgTail = (ppgTail + 1) & PPG_BUFFER_MASK;
  }
}


// ============================================================
// CRC16-CCITT
// ============================================================

uint16_t crc16_ccitt(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFF;

  while (length--) {
    crc ^= ((uint16_t)*data++) << 8;

    for (uint8_t i = 0; i < 8; i++) {
      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc <<= 1;
      }
    }
  }

  return crc;
}


// ============================================================
// FATAL ERROR LED
// ============================================================
//
// 2 flashes = BNO085 begin_I2C failure after retries
// 3 flashes = MAX30102 init failure
//
// BNO report-enable failures are NOT fatal. They are retried in loop()
// so PPG/USB streaming can remain alive while the BNO recovers.
// ============================================================

void fatalBlink(uint8_t flashes) {
  pinMode(LED_BUILTIN, OUTPUT);

  while (true) {
    for (uint8_t i = 0; i < flashes; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(150);
      digitalWrite(LED_BUILTIN, LOW);
      delay(150);
    }
    delay(900);
  }
}


// ============================================================
// BNO085 REPORT CONFIG / RETRY
// ============================================================

bool setBnoReports() {
  // Configure reports individually with a short settling gap.
  // Re-enabling an already-enabled report is safe and makes reset
  // recovery much less brittle than treating one transient failure
  // as a fatal acquisition error.

  const bool accOk =
      bno.enableReport(SH2_ACCELEROMETER, ACCEL_INTERVAL_US);

  delay(5);

  const bool gyroOk =
      bno.enableReport(
          SH2_GYROSCOPE_CALIBRATED,
          GYRO_INTERVAL_US);

  delay(5);

  const bool quatOk =
      bno.enableReport(
          SH2_ROTATION_VECTOR,
          QUAT_INTERVAL_US);

  return accOk && gyroOk && quatOk;
}


bool beginBnoWithRetry(uint8_t attempts,
                       uint32_t retryDelayMs) {
  for (uint8_t attempt = 0; attempt < attempts; ++attempt) {

    if (bno.begin_I2C(BNO_I2C_ADDRESS, &Wire)) {
      return true;
    }

    delay(retryDelayMs);
  }

  return false;
}


bool setBnoReportsWithRetry(uint8_t attempts,
                            uint32_t retryDelayMs) {
  for (uint8_t attempt = 0; attempt < attempts; ++attempt) {

    if (setBnoReports()) {
      return true;
    }

    bnoReportConfigFailures++;

    if (retryDelayMs > 0) {
      delay(retryDelayMs);
    }
  }

  return false;
}


// ============================================================
// BNO085 SERVICE
// ============================================================

void serviceBNO() {
  while (bno.getSensorEvent(&sensorValue)) {
    imuReceived++;

    ImuEvent event{};

    // Common acquisition clock shared with PPG.
    event.sample_timestamp_us = time_us_64();

    event.sensor_sequence = sensorValue.sequence;
    event.sensor_status = sensorValue.status;

    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        event.sensor_type = IMU_ACCEL;
        event.n_values = 3;
        event.v[0] = sensorValue.un.accelerometer.x;
        event.v[1] = sensorValue.un.accelerometer.y;
        event.v[2] = sensorValue.un.accelerometer.z;
        event.v[3] = 0.0f;
        imuBufferPush(event);
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        event.sensor_type = IMU_GYRO;
        event.n_values = 3;
        event.v[0] = sensorValue.un.gyroscope.x;
        event.v[1] = sensorValue.un.gyroscope.y;
        event.v[2] = sensorValue.un.gyroscope.z;
        event.v[3] = 0.0f;
        imuBufferPush(event);
        break;

      case SH2_ROTATION_VECTOR:
        event.sensor_type = IMU_QUAT;
        event.n_values = 4;
        event.v[0] = sensorValue.un.rotationVector.real;
        event.v[1] = sensorValue.un.rotationVector.i;
        event.v[2] = sensorValue.un.rotationVector.j;
        event.v[3] = sensorValue.un.rotationVector.k;
        imuBufferPush(event);
        break;

      default:
        unknownSensorEvents++;
        break;
    }
  }
}


// ============================================================
// MAX30102 LOW-LEVEL FIFO READ
// ============================================================
//
// SparkFun library is used to configure the MAX30102, but the FIFO is
// drained directly here. This avoids the library's small 4-sample
// internal software buffer and lets the Pico ring buffer own buffering.
// ============================================================

constexpr uint8_t MAX30102_REG_FIFO_DATA = 0x07;
constexpr uint8_t MAX30102_REG_OVF_COUNTER = 0x05;

bool readPpgFifoSample(uint32_t &red, uint32_t &ir) {
  Wire1.beginTransmission(PPG_I2C_ADDRESS);
  Wire1.write(MAX30102_REG_FIFO_DATA);

  if (Wire1.endTransmission(false) != 0) {
    return false;
  }

  const uint8_t requested = 6;
  const uint8_t got =
      Wire1.requestFrom(PPG_I2C_ADDRESS, requested);

  if (got != requested || Wire1.available() < requested) {
    while (Wire1.available()) {
      Wire1.read();
    }
    return false;
  }

  const uint8_t r0 = Wire1.read();
  const uint8_t r1 = Wire1.read();
  const uint8_t r2 = Wire1.read();

  const uint8_t i0 = Wire1.read();
  const uint8_t i1 = Wire1.read();
  const uint8_t i2 = Wire1.read();

  red = ((uint32_t)r0 << 16) |
        ((uint32_t)r1 << 8) |
        (uint32_t)r2;

  ir = ((uint32_t)i0 << 16) |
       ((uint32_t)i1 << 8) |
       (uint32_t)i2;

  red &= 0x03FFFF;
  ir &= 0x03FFFF;

  return true;
}

uint8_t readPpgOverflowCounter() {
  return ppg.readRegister8(PPG_I2C_ADDRESS,
                           MAX30102_REG_OVF_COUNTER);
}


// ============================================================
// MAX30102 SERVICE
// ============================================================
//
// Use the SparkFun driver's proven FIFO-drain path:
//
//   ppg.check()
//   ppg.getFIFORed()
//   ppg.getFIFOIR()
//   ppg.nextSample()
//
// Do NOT use getRed()/getIR() here: those call safeCheck() and can block,
// which previously throttled both the PPG and BNO085.
//
// Samples are still timestamped on the common Pico time_us_64() clock.
// Because the MAX30102 is configured at 200 Hz, buffered samples are
// reconstructed at nominal 5000 us spacing with a slow phase correction.
// ============================================================

void servicePPG() {
  const uint64_t nowUs = time_us_64();

  if ((nowUs - lastPpgPollUs) < PPG_POLL_INTERVAL_US) {
    return;
  }

  lastPpgPollUs = nowUs;

  // Pull any hardware-FIFO samples into the SparkFun software FIFO.
  // The successful standalone sensor test used this exact non-blocking path.
  ppg.check();

  int availableSamples = ppg.available();

  if (availableSamples <= 0) {
    return;
  }

  // The SparkFun software FIFO is small. Under normal operation this will
  // usually be one sample because servicePPG() is called far faster than 200 Hz.
  const uint8_t n = (uint8_t)availableSamples;

  // Timestamp immediately after the FIFO check. All samples currently exposed
  // by available() already existed at or before this point.
  const uint64_t pollUs = time_us_64();

  uint64_t firstSampleUs;

  if (lastPpgSampleUs == 0) {
    firstSampleUs =
        pollUs - ((uint64_t)(n - 1) * PPG_INTERVAL_US);
  } else {
    const uint64_t expectedFirstUs =
        lastPpgSampleUs + PPG_INTERVAL_US;

    const uint64_t expectedLatestUs =
        expectedFirstUs + ((uint64_t)(n - 1) * PPG_INTERVAL_US);

    const int64_t phaseErrorUs =
        (int64_t)pollUs - (int64_t)expectedLatestUs;

    if (phaseErrorUs > (int64_t)PPG_REANCHOR_US ||
        phaseErrorUs < -(int64_t)PPG_REANCHOR_US) {

      // Genuine long interruption: re-anchor the newest buffered sample
      // close to the current Pico clock.
      firstSampleUs =
          pollUs - ((uint64_t)(n - 1) * PPG_INTERVAL_US);

      ppgReanchorCount++;

    } else {
      // Slowly correct sensor/Pico oscillator mismatch while preventing
      // FIFO polling jitter from directly becoming timestamp jitter.
      int64_t correctionUs =
          phaseErrorUs / PPG_PHASE_DIVISOR;

      if (correctionUs > PPG_MAX_CORRECTION_US) {
        correctionUs = PPG_MAX_CORRECTION_US;
      }

      if (correctionUs < -PPG_MAX_CORRECTION_US) {
        correctionUs = -PPG_MAX_CORRECTION_US;
      }

      int64_t correctedFirst =
          (int64_t)expectedFirstUs + correctionUs;

      // Explicit monotonicity guard.
      if (correctedFirst <= (int64_t)lastPpgSampleUs) {
        correctedFirst = (int64_t)lastPpgSampleUs + 1;
      }

      firstSampleUs = (uint64_t)correctedFirst;
    }
  }

  for (uint8_t i = 0; i < n; ++i) {
    if (ppg.available() <= 0) {
      break;
    }

    PpgEvent event{};

    // IMPORTANT: FIFO getters only read the already-buffered sample.
    // They do not call safeCheck(), so they do not block waiting for data.
    event.red = ppg.getFIFORed();
    event.ir  = ppg.getFIFOIR();

    event.sample_timestamp_us =
        firstSampleUs + ((uint64_t)i * PPG_INTERVAL_US);

    event.sample_index = ppgSampleIndex++;

    ppgReceived++;

    ppgBufferPush(event);

    lastPpgSampleUs = event.sample_timestamp_us;

    // Advance SparkFun's software FIFO to the next already-buffered sample.
    ppg.nextSample();
  }
}


// ============================================================
// PACKET BUILDERS
// ============================================================

void initPacket(SerialPacketV2 &packet,
                uint8_t packetType,
                uint64_t sampleTimestampUs) {

  memset(&packet, 0, sizeof(packet));

  packet.magic0 = MAGIC0;
  packet.magic1 = MAGIC1;
  packet.version = PROTOCOL_VERSION;
  packet.packet_type = packetType;

  packet.packet_sequence = packetSequence++;

  // Same Pico hardware clock for every sensor.
  packet.sensor_timestamp_us = sampleTimestampUs;

  // Preserve the existing V2 32-bit timestamp field for receivers
  // that already unwrap it.
  packet.host_timestamp_us =
      (uint32_t)(sampleTimestampUs & 0xFFFFFFFFULL);
}


void finalizePacket(SerialPacketV2 &packet) {
  packet.crc = crc16_ccitt(
      reinterpret_cast<const uint8_t *>(&packet),
      sizeof(SerialPacketV2) - sizeof(packet.crc));
}


void makeImuPacket(const ImuEvent &event,
                   SerialPacketV2 &packet) {

  initPacket(packet,
             PKT_IMU,
             event.sample_timestamp_us);

  packet.sensor_sequence = event.sensor_sequence;
  packet.sensor_type = event.sensor_type;
  packet.sensor_status = event.sensor_status;
  packet.n_values = event.n_values;

  for (uint8_t i = 0; i < 4; i++) {
    packet.payload.f[i] = event.v[i];
  }

  finalizePacket(packet);
}


void makePpgPacket(const PpgEvent &event,
                   SerialPacketV2 &packet) {

  initPacket(packet,
             PKT_PPG,
             event.sample_timestamp_us);

  packet.sensor_sequence =
      (uint8_t)(event.sample_index & 0xFF);

  packet.sensor_type = PPG_RED_IR;
  packet.sensor_status = 0;
  packet.n_values = 2;

  packet.payload.u32[0] = event.red;
  packet.payload.u32[1] = event.ir;
  packet.payload.u32[2] = event.sample_index;
  packet.payload.u32[3] = 0;

  finalizePacket(packet);
}


// ============================================================
// FAIR, TIME-ORDERED USB SERIAL TX
// ============================================================
//
// Do not empty one sensor queue completely before servicing the other.
// The oldest event on the common Pico clock is sent first.
//
// Limit each TX burst so the loop returns frequently to both sensors.
// ============================================================

constexpr uint8_t MAX_TX_PACKETS_PER_SERVICE = 8;

void serviceSerialTx() {
  uint8_t sentThisCall = 0;

  while (sentThisCall < MAX_TX_PACKETS_PER_SERVICE) {

    // Do not gate transmission on DTR or availableForWrite().
    // Serial.ignoreFlowControl(true) lets Serial.write() operate without
    // depending on the host's virtual modem-control state.

    ImuEvent imuEvent{};
    PpgEvent ppgEvent{};

    const bool haveImu = imuBufferPeek(imuEvent);
    const bool havePpg = ppgBufferPeek(ppgEvent);

    if (!haveImu && !havePpg) {
      break;
    }

    SerialPacketV2 packet{};

    const bool sendImu =
        haveImu &&
        (!havePpg ||
         imuEvent.sample_timestamp_us <=
             ppgEvent.sample_timestamp_us);

    size_t written = 0;

    if (sendImu) {
      makeImuPacket(imuEvent, packet);

      written = Serial.write(
          reinterpret_cast<const uint8_t *>(&packet),
          sizeof(packet));

      if (written != sizeof(packet)) {
        break;
      }

      imuBufferRemove();
      imuTransmitted++;

    } else {
      makePpgPacket(ppgEvent, packet);

      written = Serial.write(
          reinterpret_cast<const uint8_t *>(&packet),
          sizeof(packet));

      if (written != sizeof(packet)) {
        break;
      }

      ppgBufferRemove();
      ppgTransmitted++;
    }

    sentThisCall++;
  }
}


// ============================================================
// STATS PACKETS
// ============================================================
//
// PKT_STATS is intentionally left compatible with the Mega V2:
//
//   payload[0] = IMU ring overflow
//   payload[1] = IMU max queue depth
//   payload[2] = IMU current queue depth
//   payload[3] = BNO reset count
//
// New PKT_PPG_STATS:
//
//   payload[0] = PPG ring overflow
//   payload[1] = PPG max queue depth
//   payload[2] = PPG current queue depth
//   payload[3] = MAX30102 hardware FIFO overflow counter
//
// ============================================================

void makeImuStatsPacket(SerialPacketV2 &packet) {
  const uint64_t nowUs = time_us_64();

  initPacket(packet, PKT_STATS, nowUs);

  packet.sensor_sequence = 0;
  packet.sensor_type = 0;
  packet.sensor_status = 0;
  packet.n_values = 4;

  packet.payload.u32[0] = imuOverflow;
  packet.payload.u32[1] = maxImuQueueDepth;
  packet.payload.u32[2] = imuQueueDepth();
  packet.payload.u32[3] = bnoResetCount;

  finalizePacket(packet);
}


void makePpgStatsPacket(SerialPacketV2 &packet) {
  const uint64_t nowUs = time_us_64();

  initPacket(packet, PKT_PPG_STATS, nowUs);

  packet.sensor_sequence = 0;
  packet.sensor_type = PPG_RED_IR;
  packet.sensor_status = 0;
  packet.n_values = 4;

  packet.payload.u32[0] = ppgOverflow;
  packet.payload.u32[1] = maxPpgQueueDepth;
  packet.payload.u32[2] = ppgQueueDepth();
  packet.payload.u32[3] = readPpgOverflowCounter();

  finalizePacket(packet);
}


void serviceStats() {
  static uint32_t lastStatsMs = 0;
  static bool sendPpgStatsNext = false;

  const uint32_t nowMs = millis();

  // Alternate every 500 ms:
  //   IMU stats at ~1 Hz
  //   PPG stats at ~1 Hz
  //
  // This keeps every write to one 42-byte frame and does not assume
  // that the USB TX buffer can advertise >=84 free bytes at once.
  if ((uint32_t)(nowMs - lastStatsMs) < 500) {
    return;
  }

  SerialPacketV2 packet{};

  if (sendPpgStatsNext) {
    makePpgStatsPacket(packet);
  } else {
    makeImuStatsPacket(packet);
  }

  const size_t written = Serial.write(
      reinterpret_cast<const uint8_t *>(&packet),
      sizeof(packet));

  if (written != sizeof(packet)) {
    return;
  }

  lastStatsMs = nowMs;

  sendPpgStatsNext = !sendPpgStatsNext;
}


// ============================================================
// NORMAL-OPERATION LED HEARTBEAT
// ============================================================

void serviceHeartbeat() {
  static uint32_t lastHeartbeatMs = 0;

  const uint32_t nowMs = millis();

  if ((uint32_t)(nowMs - lastHeartbeatMs) >= 500) {
    lastHeartbeatMs = nowMs;
    digitalWrite(
        LED_BUILTIN,
        !digitalRead(LED_BUILTIN));
  }
}



// ============================================================
// HOST START / STOP CONTROL
// ============================================================
//
// RX commands are one byte and travel in the opposite USB direction from
// the binary sensor stream:
//
//   'S' = start a fresh acquisition session
//   'X' = stop transmission / discard queued samples
//
// No ASCII is ever written back to the host.
//
// This gives every RPi recording a clean boundary:
//   - no stale pre-connection packets
//   - packet/sample counters start at zero
//   - ring-overflow counters start at zero
//   - MAX30102 FIFO is cleared immediately before streaming
// ============================================================

void discardBufferedSamples() {
  imuTail = imuHead;
  ppgTail = ppgHead;
}


void stopAcquisition() {
  acquisitionActive = false;
  discardBufferedSamples();
}


void startAcquisition() {
  acquisitionActive = false;

  // Drop anything accumulated before the host requested a recording.
  imuHead = 0;
  imuTail = 0;
  ppgHead = 0;
  ppgTail = 0;

  // Session-local transport/drop diagnostics.
  packetSequence = 0;

  imuReceived = 0;
  imuTransmitted = 0;
  imuOverflow = 0;
  unknownSensorEvents = 0;
  bnoResetCount = 0;
  maxImuQueueDepth = 0;

  ppgReceived = 0;
  ppgTransmitted = 0;
  ppgOverflow = 0;
  ppgSampleIndex = 0;
  ppgReanchorCount = 0;
  ppgReadErrors = 0;
  maxPpgQueueDepth = 0;

  // Empty SparkFun's software FIFO first.
  while (ppg.available() > 0) {
    ppg.nextSample();
  }

  // Then reset the MAX30102 hardware FIFO pointers and overflow counter.
  ppg.clearFIFO();

  lastPpgSampleUs = 0;
  lastPpgPollUs = time_us_64();

  acquisitionActive = true;
}


void serviceHostCommands() {
  while (Serial.available() > 0) {
    const int c = Serial.read();

    if (c == 'S') {
      startAcquisition();
    } else if (c == 'X') {
      stopAcquisition();
    }
  }
}


// ============================================================
// SETUP
// ============================================================

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  pinMode(PPG_INT_PIN, INPUT_PULLUP);
  pinMode(BNO_INT_PIN, INPUT_PULLUP);

  // On Pico native USB the baud argument is ignored, but keeping
  // 500000 preserves host-side command compatibility.
  Serial.begin(500000);

  // Native USB CDC: do not require DTR/RTS from the host.
  Serial.ignoreFlowControl(true);

  // --------------------------
  // BNO085 on I2C0 / Wire
  // --------------------------

  Wire.setSDA(BNO_SDA_PIN);
  Wire.setSCL(BNO_SCL_PIN);
  Wire.begin();
  Wire.setClock(400000);

  // Give the bus and sensor supply a short settling period.
  delay(50);

  // A two-flash fatal pattern now means ONLY that begin_I2C failed
  // repeatedly. This makes the LED diagnostic unambiguous.
  if (!beginBnoWithRetry(3, 100)) {
    fatalBlink(2);
  }

  Wire.setClock(400000);

  // The BNO can transiently reject a feature command immediately after
  // reset. Try several times. If configuration still fails, do NOT kill
  // PPG/USB acquisition: keep retrying from loop().
  delay(100);

  if (!setBnoReportsWithRetry(5, 50)) {
    bnoReportsNeedConfig = true;
    lastBnoReportRetryMs = millis();
  }

  // --------------------------
  // MAX30102 on I2C1 / Wire1
  // --------------------------
  //
  // Pin assignment is made before Wire1.begin(). We explicitly start
  // I2C1 here using the same configuration verified by the sensor test.
  // --------------------------

  Wire1.setSDA(PPG_SDA_PIN);
  Wire1.setSCL(PPG_SCL_PIN);
  Wire1.begin();
  Wire1.setClock(400000);

  if (!ppg.begin(Wire1,
                 I2C_SPEED_FAST,
                 PPG_I2C_ADDRESS)) {
    fatalBlink(3);
  }

  Wire1.setClock(400000);

  ppg.setup(
      PPG_LED_BRIGHTNESS,
      PPG_SAMPLE_AVERAGE,
      PPG_LED_MODE,
      PPG_SAMPLE_RATE_HZ,
      PPG_PULSE_WIDTH_US,
      PPG_ADC_RANGE);

  // Keep FIFO rollover enabled as a fail-safe. Normal operation drains
  // the FIFO continuously, but rollover prevents a disconnected/slow host
  // from leaving the MAX30102 permanently stuck in a full FIFO state.
  ppg.enableFIFORollover();
  ppg.setFIFOAlmostFull(15);
  ppg.clearFIFO();

  // Clear any latched interrupt status from startup.
  (void)ppg.getINT1();
  (void)ppg.getINT2();

  lastPpgPollUs = time_us_64();

  // No ASCII startup text is sent. The USB stream is binary-only.
  // Transmission begins only after the Raspberry Pi sends 'S'.
  acquisitionActive = false;

  digitalWrite(LED_BUILTIN, HIGH);
}


// ============================================================
// LOOP
// ============================================================

void loop() {
  // Always service incoming host control first.
  serviceHostCommands();

  // BNO reset recovery never halts the PPG/USB side.
  if (bno.wasReset()) {
    if (acquisitionActive) {
      bnoResetCount++;
    }

    bnoReportsNeedConfig = true;
    lastBnoReportRetryMs = 0;
  }

  if (bnoReportsNeedConfig) {
    const uint32_t nowMs = millis();

    if (lastBnoReportRetryMs == 0 ||
        (uint32_t)(nowMs - lastBnoReportRetryMs) >= 50) {

      lastBnoReportRetryMs = nowMs;

      if (setBnoReports()) {
        bnoReportsNeedConfig = false;
      } else if (acquisitionActive) {
        bnoReportConfigFailures++;
      }
    }
  }

  // Keep sensors serviced even while waiting for a host. This prevents
  // hardware FIFOs from filling, but pre-start samples are discarded below.
  if (!bnoReportsNeedConfig) {
    serviceBNO();
  }

  servicePPG();

  if (acquisitionActive) {
    serviceSerialTx();

    if (!bnoReportsNeedConfig) {
      serviceBNO();
    }

    serviceStats();
  } else {
    // Never accumulate stale samples before an explicit START.
    discardBufferedSamples();
  }

  serviceHeartbeat();
}
