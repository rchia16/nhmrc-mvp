#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include "MAX30105.h"  // Works beautifully for MAX30102 / MAXREFDES117

// --- Pin Allocations ---
#define BNO08X_CS    53
#define BNO08X_INT   4
#define BNO08X_RESET 5

Adafruit_BNO08x bno(BNO08X_RESET);
MAX30105 particleSensor; 

sh2_SensorValue_t bnoValue;
bool bnoDataReady = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10); 

  // 1. Initialize BNO085 Over Hardware SPI
  if (!bno.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    Serial.println("DATA_ERR: BNO085 SPI Fail");
    while (1);
  }
  bno.enableReport(SH2_ROTATION_VECTOR, 20000); // Stream at 50Hz

  // 2. Initialize MAX30102 Over Hardware I2C (Pins 20/21)
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("DATA_ERR: MAX30102 I2C Fail");
    while (1);
  }

  // Configure MAX30102 LED power parameters for pulse detection
  particleSensor.setup(); 
}

void loop() {
  // Check if BNO085 has updated movement numbers
  if (bno.getSensorEvent(&bnoValue)) {
    if (bnoValue.sensorId == SH2_ROTATION_VECTOR) {
      bnoDataReady = true;
    }
  }

  // Read raw optical light levels from the MAX30102 sensor
  // Red light tracks blood movement, IR tracks oxygen concentration levels
  uint32_t redValue = particleSensor.getRed();
  uint32_t irValue = particleSensor.getIR();

  // Print a combined CSV line back up to the Raspberry Pi over USB Serial
  // Format: [BNO_Real],[BNO_I],[BNO_J],[BNO_K],[Pulse_Red],[Pulse_IR]
  if (bnoDataReady) {
    Serial.print(bnoValue.un.rotationVector.real, 4);   Serial.print(",");
    Serial.print(bnoValue.un.rotationVector.i, 4);      Serial.print(",");
    Serial.print(bnoValue.un.rotationVector.j, 4);      Serial.print(",");
    Serial.print(bnoValue.un.rotationVector.k, 4);      Serial.print(",");
    Serial.print(redValue);                             Serial.print(",");
    Serial.println(irValue);
    
    bnoDataReady = false; // Clear flag latch
  }
  
  delay(10); // Quick padding to keep communication lines stable
}
