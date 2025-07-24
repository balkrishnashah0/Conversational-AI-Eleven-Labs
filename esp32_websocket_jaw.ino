#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Adafruit_PWMServoDriver.h>

const char* ssid = "print3d_2";
const char* password = "INNOVATE3D";
const char* websocket_host = "192.168.1.77";  // Laptop IP
const int websocket_port = 8765;

WebSocketsClient webSocket;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const int servo_channel = 0;
int servo_min = 110, servo_max = 510;
int lastAngle = -1;  // Track last angle to detect changes
unsigned long messageCount = 0;

void moveServo(int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = map(angle, 0, 180, servo_min, servo_max);
  pwm.setPWM(servo_channel, 0, pulse);
  
  Serial.print("🦾 Moving servo to angle: ");
  Serial.print(angle);
  Serial.print("° → pulse: ");
  Serial.print(pulse);
  
  if (angle != lastAngle) {
    Serial.print(" (CHANGED from ");
    Serial.print(lastAngle);
    Serial.print("°)");
    lastAngle = angle;
  } else {
    Serial.print(" (same)");
  }
  Serial.println();
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("❌ WebSocket disconnected");
      break;

    case WStype_CONNECTED:
      Serial.println("✅ WebSocket connected - Ready to receive angles!");
      messageCount = 0;
      break;

    case WStype_TEXT: {
      messageCount++;
      String message = String((char*)payload);
      int angle = message.toInt();
      
      // Enhanced debug output
      Serial.print("📨 [");
      Serial.print(messageCount);
      Serial.print("] Received: '");
      Serial.print(message);
      Serial.print("' → angle: ");
      Serial.print(angle);
      Serial.print("° (len: ");
      Serial.print(length);
      Serial.print(", time: ");
      Serial.print(millis());
      Serial.println("ms)");
      
      moveServo(angle);
      break;
    }

    case WStype_ERROR:
      Serial.println("⚠️ WebSocket error");
      break;

    case WStype_BIN:
      Serial.println("📦 Binary data received (unexpected)");
      break;

    default:
      Serial.print("🔍 Unknown WebSocket event: ");
      Serial.println(type);
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);  // Give time for Serial to initialize
  
  Serial.println("\n🚀 ESP32 Jaw Animation Debug");
  Serial.println("============================");
  
  Serial.println("🔧 Initializing PCA9685...");
  pwm.begin();
  pwm.setPWMFreq(60);
  delay(500);

  Serial.print("📶 Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Wi-Fi connected!");
  Serial.print("📡 ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  Serial.println("🌐 Connecting to WebSocket...");
  webSocket.begin(websocket_host, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  
  // Move to initial position
  Serial.println("🏠 Moving to initial position (90°)");
  moveServo(90);
  
  Serial.println("🎯 Ready! Waiting for angle commands...");
}

void loop() {
  webSocket.loop();
  
  // Optional: Print periodic status
  static unsigned long lastStatusPrint = 0;
  if (millis() - lastStatusPrint > 10000) {  // Every 10 seconds
    Serial.print("💓 Status: ");
    Serial.print(messageCount);
    Serial.print(" messages received, WebSocket ");
    Serial.println(webSocket.isConnected() ? "connected" : "disconnected");
    lastStatusPrint = millis();
  }
}