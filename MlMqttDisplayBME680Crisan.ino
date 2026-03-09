#include <ArduTFLite.h>

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <time.h>
#include <TimeLib.h>

#include <PubSubClient.h>
#include <ArduinoJson.h>

#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ===== TinyML =====
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tempPredictor.h"

// ================= OLED =================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

const char* ssid = "Ossian Residence Dorm Room 2.4G";
const char* password = "1234543210";
//const char* mqtt_server = "79.119.133.189";
const char* mqtt_server = "cbuni.go.ro";

StaticJsonDocument<256> docTemperature;
StaticJsonDocument<256> docHumidity;
StaticJsonDocument<128> docPrediction;

volatile bool running = true;

WiFiClient espClient;
PubSubClient client(espClient);
long lastMsg = 0;
char msg[50];
int value = 0;
float temp, pred, gas, hum, pres;

// ================= TinyML =================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
static tflite::AllOpsResolver resolver;

// Normalization params
const float X_min = 18.64935072147522;
const float X_max = 25.362445689939285;
const float y_min = 18.64935072147522;
const float y_max = 25.362445689939285;


Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ================= BME680 =================
#define BME680_ADDRESS 0x77   // change to 0x76 if needed
Adafruit_BME680 bme;

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA, SCL for ESP32

  // ML model
  model = tflite::GetModel(tempPredictor);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
  } else {
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("Failed to allocate tensors!");
      interpreter = nullptr;
    } else {
      input = interpreter->input(0);
      output = interpreter->output(0);
      if (!input || !output) Serial.println("Failed to get input/output tensors!");
      else Serial.println("TinyML model loaded.");
    }
  }

  // ---- OLED INIT ----
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("SSD1306 allocation failed");
    while (1);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Starting...");
  display.display();

  // ---- BME680 INIT ----
  if (!bme.begin(BME680_ADDRESS)) {
    Serial.println("Could not find BME680");
    display.println("BME680 error!");
    display.display();
    while (1);
  }

  // BME680 configuration
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150); // 320°C for 150 ms

  //mqtt config
  setup_wifi();
  client.setServer(mqtt_server, 8883);
  client.setCallback(callback);

  configTime(0, 0, "pool.ntp.org", "time.nist.gov");

  Serial.print("Waiting for NTP time sync");
  time_t now = time(nullptr);
  while (now < 100000) {
    delay(500);
    Serial.print(".");
    now = time(nullptr);
  }
  Serial.println("\nTime synced!");




  display.println("BME680 OK");
  display.display();
  delay(2000);


}

String getISOTimestamp() {
  struct tm timeinfo;
  getLocalTime(&timeinfo);

  // Seconds part
  char dateTime[25];
  strftime(dateTime, sizeof(dateTime), "%Y-%m-%dT%H:%M:%S", &timeinfo);

  // Microseconds (from uptime)
  int64_t us = esp_timer_get_time();
  int micro = us % 1000000;

  char timestamp[32];
  snprintf(timestamp, sizeof(timestamp),
           "%s.%06d",
           dateTime, micro);

  return String(timestamp);
}

void setup_wifi() {
  delay(10);
  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}


void callback(char* topic, byte* payload, unsigned int length) {
  String topicStr = String(topic);
  String message;

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("MQTT [");
  Serial.print(topicStr);
  Serial.print("]: ");
  Serial.println(message);

  if (topicStr == "options") {
    if (message == "off") {
      running = false;
      Serial.println("⏸ System paused");
    } 
    else if (message == "on") {
      running = true;
      Serial.println("▶ System resumed");
    }
  }
}


float normalizeInput(float temp) {
  return (temp - X_min) / (X_max - X_min);
}

float denormalizeOutput(float norm) {
  return norm * (y_max - y_min) + y_min;
}

float predictTemperature(float currentTemp) {
  if (!interpreter) return NAN;
  input->data.f[0] = normalizeInput(currentTemp);
  if (interpreter->Invoke() != kTfLiteOk) return NAN;
  return denormalizeOutput(output->data.f[0]);
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect("ESP32Client")) {
      Serial.println("connected");
      client.subscribe("options");   // only what you need
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

// ================= LOOP =================
void loop() {

  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  if (!running) {
    display.clearDisplay();
    display.setCursor(0,0);
    display.println("SYSTEM PAUSED");
    display.println("Waiting for ON");
    display.display();
    delay(100);   // small delay to avoid watchdog reset
    return;       // wait for "on"
  }
  

  if (!bme.performReading()) {
    Serial.println("Failed to perform reading");
    return;
  }

  temp = bme.temperature;
  hum  = bme.humidity;
  pres = bme.pressure / 100.0; // hPa
  gas  = bme.gas_resistance / 1000.0; // kOhms
  pred = predictTemperature(temp);

  // Serial output
  Serial.print("Temp: "); Serial.print(temp); Serial.println(" °C");
  Serial.print("Hum : "); Serial.print(hum);  Serial.println(" %");
  Serial.print("Pres: "); Serial.print(pres); Serial.println(" hPa");
  Serial.print("Gas : "); Serial.print(gas);  Serial.println(" kOhm");
  Serial.println("--------------------");

  // OLED display
  display.clearDisplay();
  display.setCursor(0, 0);

  display.println("BME680 Sensor");
  display.println("----------------");
  display.print("Temp: "); display.print(temp); display.println(" C");
  display.print("Pred: "); display.print(pred); display.println(" C");
  display.print("Hum : "); display.print(hum);  display.println(" %");
  display.print("Pres: "); display.print(pres); display.println(" hPa");
  display.print("Gas : "); display.print(gas);  display.println(" kOhm");

  display.display();




  long now = millis();
  if (now - lastMsg > 5000) {
    lastMsg = now;

    // Temperature in Celsius
    float temperature = temp;  
    // Uncomment the next line to set temperature in Fahrenheit 
    // (and comment the previous temperature line)
    //temperature = 1.8 * bme.readTemperature() + 32; // Temperature in Fahrenheit
    
    // Convert the value to a char array
    char tempString[8];
    dtostrf(temperature, 1, 2, tempString);
    Serial.print("Temperature: ");
    Serial.println(tempString);
    
    docTemperature["sensor_id"] = "1";
    docTemperature["sensor_type"] = "temperature";
    docTemperature["value"] = tempString;
    docTemperature["time"] = getISOTimestamp();

    char jsonBuffer[256],jsonBuffer2[256];
    serializeJson(docTemperature, jsonBuffer);
    Serial.print("Publishing JSON: ");
    Serial.println(jsonBuffer);

    client.publish("sensors/temperature", jsonBuffer);

    float humidity    = hum;
    
    // ===== Prediction JSON =====
    docPrediction.clear();
    docPrediction["sensor_type"]   = "temperature";
    docPrediction["real"]   = temp;
    docPrediction["prediction"] = pred;
  
    char predBuffer[128];
    serializeJson(docPrediction, predBuffer);
    client.publish("ml", predBuffer);

    Serial.print("Publishing prediction: ");
    Serial.println(predBuffer);

    // Convert the value to a char array
    char humString[8];
    dtostrf(humidity, 1, 2, humString);
    Serial.print("Humidity: ");
    Serial.println(humString);

    docHumidity["sensor_id"] = "1";
    docHumidity["sensor_type"] = "humidity";
    docHumidity["value"] = humString;
    docHumidity["time"] = getISOTimestamp();

    serializeJson(docHumidity, jsonBuffer2);
    Serial.print("Publishing JSON: ");
    Serial.println(jsonBuffer2);

    client.publish("sensors/humidity", jsonBuffer2);
  }

  delay(2000);
}
