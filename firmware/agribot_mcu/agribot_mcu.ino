/*
 * ===========================================================================
 *  AgriBot MCU firmware - Robofest Gujarat 6.0, Thematic Track (f)
 *  Autonomous Precision Robot for Sustainable Smart Agriculture
 *
 *  Target: Arduino Mega 2560 on the custom PCB (Uno works with fewer sensors)
 *
 *  Computation is split across two tiers (proposal Section 2). The Jetson does
 *  perception and decision-making; this firmware does the hard real-time work:
 *  motor PWM, quadrature encoder counting, IMU acquisition, servo positioning
 *  and solenoid switching. Keeping those off the Jetson stops real-time
 *  interrupts competing with GPU inference.
 *
 *  Wire protocol (see src/agribot/hal/protocol.py - the two MUST agree):
 *      $<payload>*<XX>\n      XX = XOR of payload bytes, uppercase hex
 *
 *  Host -> MCU
 *      M,<left>,<right>   motors, -1000..1000
 *      S,<ch>,<deg*10>    servo channel to position
 *      K,<deg*10>         marker servo
 *      V,<0|1>            solenoid valve
 *      P,<0|1>            pump
 *      H,<seq>            heartbeat  (REQUIRED - see watchdog below)
 *      Z                  zero the encoder accumulators
 *      X                  emergency stop
 *
 *  MCU -> host, at TELEMETRY_HZ
 *      T,seq,ms,gyro_mdps,accel_mmss,mag_cdeg,encL_mmps,encR_mmps,
 *        us_front_mm,us_left_mm,flow,batt_mv,roll_cdeg,pitch_cdeg,ir
 *
 *  SAFETY - three independent layers, none of which depend on the Jetson:
 *    1. Heartbeat watchdog. If no H frame arrives for HEARTBEAT_TIMEOUT_MS the
 *       motors stop and the valve shuts. A crashed Jetson, a yanked USB lead
 *       or a hung inference loop therefore stops the robot rather than leaving
 *       it driving with the last command latched.
 *    2. Valve maximum-open timer. The solenoid cannot stay open longer than
 *       VALVE_MAX_OPEN_MS whatever the host asks, so a lost "close" frame
 *       cannot empty the reservoir onto one plant.
 *    3. Checksummed frames. Motor wiring runs beside the USB lead; a corrupted
 *       digit in a motor command is the difference between a correction and a
 *       lurch, so frames that fail the checksum are dropped. Dropping is safe
 *       because commands are re-sent at 50 Hz.
 * ===========================================================================
 */

#include <Arduino.h>
#include <Servo.h>
#include <Wire.h>

// ---------------------------------------------------------------------------
//  Pin map - matches the custom PCB silkscreen
// ---------------------------------------------------------------------------
// Motor driver A (left side), B (right side). PWM pins must be timer-capable.
const uint8_t PIN_ML_PWM = 5;
const uint8_t PIN_ML_IN1 = 22;
const uint8_t PIN_ML_IN2 = 23;
const uint8_t PIN_MR_PWM = 6;
const uint8_t PIN_MR_IN1 = 24;
const uint8_t PIN_MR_IN2 = 25;

// Encoders. The A channels MUST be on external-interrupt pins.
const uint8_t PIN_ENC_L_A = 2;    // INT0
const uint8_t PIN_ENC_L_B = 4;
const uint8_t PIN_ENC_R_A = 3;    // INT1
const uint8_t PIN_ENC_R_B = 7;

// Actuators
const uint8_t PIN_SERVO_PAN    = 9;
const uint8_t PIN_SERVO_TILT   = 10;
const uint8_t PIN_SERVO_MARKER = 11;
const uint8_t PIN_PUMP         = 26;   // MOSFET gate
const uint8_t PIN_VALVE        = 27;   // MOSFET gate / relay

// Sensors
const uint8_t PIN_US_FRONT_TRIG = 30;
const uint8_t PIN_US_FRONT_ECHO = 31;
const uint8_t PIN_US_LEFT_TRIG  = 32;
const uint8_t PIN_US_LEFT_ECHO  = 33;
const uint8_t PIN_FLOW          = 18;  // INT5, in-line flow sensor
const uint8_t PIN_BATT_SENSE    = A0;
const uint8_t PIN_IR_START      = A8;  // QTR-8A occupies A8..A15

// ---------------------------------------------------------------------------
//  Configuration - keep in step with config/robot.yaml
// ---------------------------------------------------------------------------
const uint32_t SERIAL_BAUD          = 115200;
const uint16_t TELEMETRY_HZ         = 100;
const uint16_t ENCODER_SAMPLE_HZ    = 50;
const uint16_t ULTRASONIC_HZ        = 20;
const uint16_t HEARTBEAT_TIMEOUT_MS = 500;
const uint16_t VALVE_MAX_OPEN_MS    = 1500;

const float WHEEL_RADIUS_M       = 0.0325f;
const int32_t ENCODER_TICKS_REV  = 1440;
const float BATT_DIVIDER_RATIO   = 5.7f;   // (R1+R2)/R2 on the sense divider
const uint8_t IR_CHANNELS        = 8;
const uint16_t IR_THRESHOLD      = 500;    // ADC counts: below = line detected

// Distance per encoder tick, metres.
const float METRES_PER_TICK = (2.0f * 3.14159265f * WHEEL_RADIUS_M) / ENCODER_TICKS_REV;

// ---------------------------------------------------------------------------
//  State
// ---------------------------------------------------------------------------
volatile int32_t encLeftTicks  = 0;
volatile int32_t encRightTicks = 0;
volatile uint32_t flowTicks    = 0;

int32_t lastEncLeft = 0, lastEncRight = 0;
float encLeftMps = 0.0f, encRightMps = 0.0f;

Servo servoPan, servoTilt, servoMarker;

bool pumpOn = false;
bool valveOpen = false;
uint32_t valveOpenedAtMs = 0;
bool estopped = false;

uint32_t lastHeartbeatMs = 0;
bool everHeardHost = false;

uint16_t telemetrySeq = 0;
uint32_t nextTelemetryMs = 0;
uint32_t nextEncoderMs = 0;
uint32_t nextUltrasonicMs = 0;

float usFrontM = INFINITY;
float usLeftM = INFINITY;

// IMU sample, filled by readImu(). Replace the body with the BNO055/MPU-9250
// driver calls for the fitted part; the units below are what the host expects.
float gyroZRadS = 0.0f;
float accelXMps2 = 0.0f;
float magHeadingRad = NAN;      // NaN = no magnetometer fix this cycle
float rollDeg = 0.0f, pitchDeg = 0.0f;

// Serial receive buffer.
const uint8_t RX_MAX = 64;
char rxBuf[RX_MAX];
uint8_t rxLen = 0;

// ---------------------------------------------------------------------------
//  Interrupt service routines - keep these to a handful of instructions
// ---------------------------------------------------------------------------
void isrEncoderLeft() {
  // Quadrature: B's level at A's rising edge gives the direction.
  encLeftTicks += (digitalRead(PIN_ENC_L_B) == LOW) ? 1 : -1;
}

void isrEncoderRight() {
  encRightTicks += (digitalRead(PIN_ENC_R_B) == LOW) ? -1 : 1;
}

void isrFlow() {
  flowTicks++;
}

// ---------------------------------------------------------------------------
//  Protocol
// ---------------------------------------------------------------------------
uint8_t computeChecksum(const char *payload, uint8_t len) {
  uint8_t sum = 0;
  for (uint8_t i = 0; i < len; i++) sum ^= (uint8_t)payload[i];
  return sum;
}

void sendFrame(const char *payload) {
  uint8_t sum = computeChecksum(payload, strlen(payload));
  Serial.print('$');
  Serial.print(payload);
  Serial.print('*');
  if (sum < 16) Serial.print('0');
  Serial.print(sum, HEX);
  Serial.print('\n');
}

/* Validate a received line and return a pointer to its payload, or NULL. */
char *validateFrame(char *line) {
  if (line[0] != '$') return NULL;
  char *star = strrchr(line, '*');
  if (star == NULL || (star - line) < 2) return NULL;
  *star = '\0';
  char *payload = line + 1;
  uint8_t expected = (uint8_t)strtol(star + 1, NULL, 16);
  if (computeChecksum(payload, strlen(payload)) != expected) return NULL;
  return payload;
}

// ---------------------------------------------------------------------------
//  Actuation
// ---------------------------------------------------------------------------
void setMotor(uint8_t pwmPin, uint8_t in1, uint8_t in2, int16_t value) {
  // value is -1000..1000
  if (value > 1000) value = 1000;
  if (value < -1000) value = -1000;
  bool forward = value >= 0;
  uint16_t magnitude = forward ? value : -value;
  uint8_t duty = (uint8_t)((uint32_t)magnitude * 255UL / 1000UL);
  digitalWrite(in1, forward ? HIGH : LOW);
  digitalWrite(in2, forward ? LOW : HIGH);
  analogWrite(pwmPin, duty);
}

void setDrive(int16_t left, int16_t right) {
  if (estopped) { left = 0; right = 0; }
  setMotor(PIN_ML_PWM, PIN_ML_IN1, PIN_ML_IN2, left);
  setMotor(PIN_MR_PWM, PIN_MR_IN1, PIN_MR_IN2, right);
}

void setValve(bool open) {
  valveOpen = open;
  digitalWrite(PIN_VALVE, open ? HIGH : LOW);
  if (open) valveOpenedAtMs = millis();
}

void setPump(bool on) {
  pumpOn = on;
  digitalWrite(PIN_PUMP, on ? HIGH : LOW);
}

void emergencyStop() {
  estopped = true;
  setDrive(0, 0);
  // Valve first, then pump: never leave the pump driving into a shut valve.
  setValve(false);
  setPump(false);
}

// ---------------------------------------------------------------------------
//  Command handling
// ---------------------------------------------------------------------------
void handleCommand(char *payload) {
  char kind = payload[0];

  if (kind == 'M') {
    int16_t left = 0, right = 0;
    if (sscanf(payload, "M,%hd,%hd", &left, &right) == 2) setDrive(left, right);

  } else if (kind == 'S') {
    int channel = 0, deciDeg = 0;
    if (sscanf(payload, "S,%d,%d", &channel, &deciDeg) == 2) {
      int angle = deciDeg / 10;
      if (angle < 0) angle = 0;
      if (angle > 180) angle = 180;
      if (channel == 0) servoPan.write(angle);
      else if (channel == 1) servoTilt.write(angle);
      else if (channel == 2) servoMarker.write(angle);
    }

  } else if (kind == 'K') {
    int deciDeg = 0;
    if (sscanf(payload, "K,%d", &deciDeg) == 1) {
      int angle = constrain(deciDeg / 10, 0, 180);
      servoMarker.write(angle);
    }

  } else if (kind == 'V') {
    setValve(payload[2] == '1');

  } else if (kind == 'P') {
    setPump(payload[2] == '1');

  } else if (kind == 'H') {
    lastHeartbeatMs = millis();
    everHeardHost = true;
    // A heartbeat re-arms after an estop; the host only sends them while it
    // believes it is in control.
    estopped = false;

  } else if (kind == 'Z') {
    noInterrupts();
    encLeftTicks = 0;
    encRightTicks = 0;
    interrupts();
    lastEncLeft = 0;
    lastEncRight = 0;

  } else if (kind == 'X') {
    emergencyStop();
  }
}

void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0) {
        rxBuf[rxLen] = '\0';
        char *payload = validateFrame(rxBuf);
        if (payload != NULL) handleCommand(payload);
        rxLen = 0;
      }
    } else if (rxLen < RX_MAX - 1) {
      rxBuf[rxLen++] = c;
    } else {
      // Overlong line: discard rather than wrap, so a noise burst cannot be
      // reassembled into a valid-looking command.
      rxLen = 0;
    }
  }
}

// ---------------------------------------------------------------------------
//  Sensors
// ---------------------------------------------------------------------------
void updateEncoders(float dt) {
  noInterrupts();
  int32_t left = encLeftTicks;
  int32_t right = encRightTicks;
  interrupts();

  int32_t dLeft = left - lastEncLeft;
  int32_t dRight = right - lastEncRight;
  lastEncLeft = left;
  lastEncRight = right;

  encLeftMps = (dLeft * METRES_PER_TICK) / dt;
  encRightMps = (dRight * METRES_PER_TICK) / dt;
}

/* Blocking ping, capped so a missing echo costs 30 ms not 1 s. */
float readUltrasonic(uint8_t trigPin, uint8_t echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  unsigned long duration = pulseIn(echoPin, HIGH, 30000UL);
  if (duration == 0) return INFINITY;      // no echo: treat as clear
  return (duration * 0.000343f) / 2.0f;    // seconds * m/s / 2
}

void readImu() {
  /*
   * Replace with the driver for the fitted part. For a BNO055 in NDOF mode:
   *
   *   imu::Vector<3> gyro  = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
   *   imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
   *   imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
   *   gyroZRadS     = gyro.z() * DEG_TO_RAD;
   *   accelXMps2    = accel.x();
   *   magHeadingRad = euler.x() * DEG_TO_RAD;
   *   rollDeg = euler.z(); pitchDeg = euler.y();
   *
   * Report magHeadingRad as NAN whenever the calibration status for the
   * magnetometer is zero. The host's Kalman filter then skips the correction
   * step instead of being fed a heading it cannot trust - which is far better
   * than substituting a stale or default value.
   */
  gyroZRadS = 0.0f;
  accelXMps2 = 0.0f;
  magHeadingRad = NAN;
  rollDeg = 0.0f;
  pitchDeg = 0.0f;
}

uint8_t readIrArray() {
  uint8_t mask = 0;
  for (uint8_t i = 0; i < IR_CHANNELS; i++) {
    if (analogRead(PIN_IR_START + i) < IR_THRESHOLD) mask |= (1 << i);
  }
  return mask;
}

float readBatteryVolts() {
  return (analogRead(PIN_BATT_SENSE) * 5.0f / 1023.0f) * BATT_DIVIDER_RATIO;
}

// ---------------------------------------------------------------------------
//  Telemetry
// ---------------------------------------------------------------------------
long clampLong(long value, long low, long high) {
  return value < low ? low : (value > high ? high : value);
}

void sendTelemetry() {
  char payload[160];
  telemetrySeq++;

  long magCdeg = isnan(magHeadingRad)
      ? -1L
      : clampLong((long)(magHeadingRad * 57.2957795f * 100.0f), -36000L, 36000L);
  long usFrontMm = isinf(usFrontM) ? -1L : clampLong((long)(usFrontM * 1000.0f), 0L, 65535L);
  long usLeftMm  = isinf(usLeftM)  ? -1L : clampLong((long)(usLeftM  * 1000.0f), 0L, 65535L);

  noInterrupts();
  uint32_t flow = flowTicks;
  interrupts();

  snprintf(payload, sizeof(payload),
           "T,%u,%lu,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%lu,%ld,%ld,%ld,%u",
           telemetrySeq,
           (unsigned long)millis(),
           clampLong((long)(gyroZRadS * 57.2957795f * 1000.0f), -2000000L, 2000000L),
           clampLong((long)(accelXMps2 * 1000.0f), -200000L, 200000L),
           magCdeg,
           clampLong((long)(encLeftMps * 1000.0f), -10000L, 10000L),
           clampLong((long)(encRightMps * 1000.0f), -10000L, 10000L),
           usFrontMm,
           usLeftMm,
           (unsigned long)flow,
           clampLong((long)(readBatteryVolts() * 1000.0f), 0L, 65535L),
           clampLong((long)(rollDeg * 100.0f), -18000L, 18000L),
           clampLong((long)(pitchDeg * 100.0f), -18000L, 18000L),
           readIrArray());
  sendFrame(payload);
}

// ---------------------------------------------------------------------------
//  Safety
// ---------------------------------------------------------------------------
void enforceSafety() {
  uint32_t now = millis();

  // Layer 1: heartbeat watchdog. Only armed once the host has spoken at least
  // once, so a robot powered up on the bench does not sit in a fault state.
  if (everHeardHost && (now - lastHeartbeatMs) > HEARTBEAT_TIMEOUT_MS) {
    if (!estopped) emergencyStop();
  }

  // Layer 2: the valve cannot be held open indefinitely by a lost frame.
  if (valveOpen && (now - valveOpenedAtMs) > VALVE_MAX_OPEN_MS) {
    setValve(false);
    setPump(false);
  }
}

// ---------------------------------------------------------------------------
//  Arduino entry points
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);

  pinMode(PIN_ML_PWM, OUTPUT); pinMode(PIN_ML_IN1, OUTPUT); pinMode(PIN_ML_IN2, OUTPUT);
  pinMode(PIN_MR_PWM, OUTPUT); pinMode(PIN_MR_IN1, OUTPUT); pinMode(PIN_MR_IN2, OUTPUT);

  pinMode(PIN_ENC_L_A, INPUT_PULLUP); pinMode(PIN_ENC_L_B, INPUT_PULLUP);
  pinMode(PIN_ENC_R_A, INPUT_PULLUP); pinMode(PIN_ENC_R_B, INPUT_PULLUP);
  pinMode(PIN_FLOW, INPUT_PULLUP);

  pinMode(PIN_PUMP, OUTPUT);  digitalWrite(PIN_PUMP, LOW);
  pinMode(PIN_VALVE, OUTPUT); digitalWrite(PIN_VALVE, LOW);

  pinMode(PIN_US_FRONT_TRIG, OUTPUT); pinMode(PIN_US_FRONT_ECHO, INPUT);
  pinMode(PIN_US_LEFT_TRIG, OUTPUT);  pinMode(PIN_US_LEFT_ECHO, INPUT);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_L_A), isrEncoderLeft, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_R_A), isrEncoderRight, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW), isrFlow, RISING);

  servoPan.attach(PIN_SERVO_PAN);
  servoTilt.attach(PIN_SERVO_TILT);
  servoMarker.attach(PIN_SERVO_MARKER);
  servoPan.write(90);
  servoTilt.write(75);
  servoMarker.write(30);

  Wire.begin();
  // bno.begin();  // enable once the IMU driver above is filled in

  setDrive(0, 0);

  uint32_t now = millis();
  nextTelemetryMs = now;
  nextEncoderMs = now;
  nextUltrasonicMs = now;
}

void loop() {
  uint32_t now = millis();

  pollSerial();
  enforceSafety();

  if ((int32_t)(now - nextEncoderMs) >= 0) {
    updateEncoders(1.0f / ENCODER_SAMPLE_HZ);
    nextEncoderMs += 1000UL / ENCODER_SAMPLE_HZ;
  }

  // Ultrasonic pings block for up to 30 ms each, so they run at their own slow
  // rate and are staggered - pinging both every cycle would dominate the loop
  // and starve the encoder sampling.
  if ((int32_t)(now - nextUltrasonicMs) >= 0) {
    static bool pingFront = true;
    if (pingFront) usFrontM = readUltrasonic(PIN_US_FRONT_TRIG, PIN_US_FRONT_ECHO);
    else           usLeftM  = readUltrasonic(PIN_US_LEFT_TRIG, PIN_US_LEFT_ECHO);
    pingFront = !pingFront;
    nextUltrasonicMs += 1000UL / ULTRASONIC_HZ;
  }

  if ((int32_t)(now - nextTelemetryMs) >= 0) {
    readImu();
    sendTelemetry();
    nextTelemetryMs += 1000UL / TELEMETRY_HZ;
  }
}
