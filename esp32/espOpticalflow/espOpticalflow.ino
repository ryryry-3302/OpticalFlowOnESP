#include <helper.h>


uint8_t frame1[IMG_SIZE][IMG_SIZE];
uint8_t frame2[IMG_SIZE][IMG_SIZE];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println("Serial monitor started");

}


void loop() {
  // put your main code here, to run repeatedly:
  //Serial.println("Serial monitor started");
  receive_frame(frame1);
  Serial.println("Frame1 values");
  for (int i = 0; i < IMG_SIZE; i++) {
      for (int j = 0; j < IMG_SIZE; j++) {
          Serial.print(frame1[i][j]);
          Serial.print(" ");
      }
      Serial.println();
  }
}
