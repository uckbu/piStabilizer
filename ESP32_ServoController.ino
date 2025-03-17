#include <WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// WiFi credentials
const char* ssid = "Verizon_X37CV4";
const char* password = "mild-hut3-mixed";

// MQTT broker settings
const char* mqtt_server = "192.168.1.10"; // Replace with your broker's IP
const int mqtt_port = 1883;
const char* mqtt_topic = "servo/angles";

// Initialize WiFi and MQTT clients
WiFiClient espClient;
PubSubClient client(espClient);

// Create servo objects for each channel
Servo servoPitch;
Servo servoYaw;
Servo servoRoll;

// MQTT callback: called when a new message arrives on subscribed topics
void callback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  message.trim(); // Remove whitespace/newlines
  Serial.print("Received: ");
  Serial.println(message);

  // Expected format: "pitch,yaw,roll"
  int firstComma = message.indexOf(',');
  int secondComma = message.indexOf(',', firstComma + 1);
  if (firstComma == -1 || secondComma == -1) {
    Serial.println("Parsing error");
    return;
  }
  
  float pitch = message.substring(0, firstComma).toFloat();
  float yaw   = message.substring(firstComma + 1, secondComma).toFloat();
  float roll  = message.substring(secondComma + 1).toFloat();

  // Update servos with the new angles
  servoPitch.write(pitch);
  servoYaw.write(yaw);
  servoRoll.write(roll);

  Serial.print("Set servos to: Pitch=");
  Serial.print(pitch);
  Serial.print(", Yaw=");
  Serial.print(yaw);
  Serial.print(", Roll=");
  Serial.println(roll);
}

// Reconnect to MQTT broker if the connection is lost
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32ServoSubscriber")) {
      Serial.println("connected");
      client.subscribe(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // Set up MQTT client
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // Attach servos to the desired GPIO pins (adjust as needed)
  servoPitch.attach(17);  // Example: connect servo for pitch to GPIO 17
  servoYaw.attach(27);    // Example: connect servo for yaw to GPIO 27
  servoRoll.attach(22);   // Example: connect servo for roll to GPIO 22
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
