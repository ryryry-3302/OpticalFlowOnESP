#include <Arduino.h>

#define ARRAY_SIZE 256  // 16x16 = 256 bytes
uint8_t receivedData1[16][16];  // First frame
uint8_t receivedData2[16][16];  // Second frame
int dataCount = 0;  // Track the position in the array
bool isFirstFrame = true;  // Toggle between the two frames

const int SCALE_FACTOR = 100;



void computeOpticalFlow(int x, int y) {
  // Gradients (float for precision)
  float I_x = 0.0f, I_y = 0.0f, I_t = 0.0f;

  // Local 3x3 window (matrix G, vector b)
  float G[2][2] = {0.0f};  // Gradient matrix
  float b[2] = {0.0f};     // RHS vector

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int xCoord = x + i;
      int yCoord = y + j;

      if (xCoord >= 0 && xCoord < 16 && yCoord >= 0 && yCoord < 16) {
        // Use int16_t to safely handle potential overflow when subtracting
        int16_t I_x_int = (int16_t)receivedData1[yCoord][xCoord + 1] - (int16_t)receivedData1[yCoord][xCoord - 1];
        int16_t I_y_int = (int16_t)receivedData1[yCoord + 1][xCoord] - (int16_t)receivedData1[yCoord - 1][xCoord];
        int16_t I_t_int = (int16_t)receivedData2[yCoord][xCoord] - (int16_t)receivedData1[yCoord][xCoord];

        // Convert to float for further calculations
        I_x = (float)I_x_int;
        I_y = (float)I_y_int;
        I_t = (float)I_t_int;

        // Update matrix G
        G[0][0] += I_x * I_x;
        G[0][1] += I_x * I_y;
        G[1][0] += I_x * I_y;
        G[1][1] += I_y * I_y;

        // Update vector b
        b[0] += I_x * I_t;
        b[1] += I_y * I_t;
      }
    }
  }
  
  // Determinant of G
  float det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
  float u;
  float v;

  if (det == 0) {
    u = 0;
    v = 0;
  }
  
  else {
  // Solve for (u, v) using Cramer's rule
  u = -(b[0] * G[1][1] - b[1] * G[0][1]) / det;
  v = -(b[1] * G[0][0] - b[0] * G[1][0]) / det;
  }

  // Scale the flow vectors by the SCALE_FACTOR and convert to integers
  int scaled_u = (int)(u * SCALE_FACTOR);
  int scaled_v = (int)(v * SCALE_FACTOR);

  // Send the scaled optical flow vector back
  Serial.write((uint8_t)(scaled_u >> 8));  // High byte of u
  Serial.write((uint8_t)(scaled_u & 0xFF));  // Low byte of u
  Serial.write((uint8_t)(scaled_v >> 8));  // High byte of v
  Serial.write((uint8_t)(scaled_v & 0xFF));  // Low byte of v
}



void setup() {
  Serial.begin(500000);  // Initialize serial communication
  Serial.println("Ready to receive data...");
}


void loop() {
  if (Serial.available() > 0) {
    uint8_t byteReceived = Serial.read();

    // Store the received byte in the appropriate array
    int row = dataCount / 16;  // Determine the row
    int col = dataCount % 16;  // Determine the column

    if (isFirstFrame) {
      receivedData1[row][col] = byteReceived;
    } else {
      receivedData2[row][col] = byteReceived;
    }

    dataCount++;

    // Once a full frame is received, process it
    if (dataCount >= ARRAY_SIZE) {
      if (!isFirstFrame) {
        // Compute optical flow for coordinate (8, 8)
        computeOpticalFlow(8, 8);
        memcpy(receivedData1, receivedData2, sizeof(receivedData1));
      }

      // Toggle to the next frame and reset counter
      isFirstFrame = false;
      dataCount = 0;
    }
  }
}