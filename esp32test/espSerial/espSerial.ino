#define ARRAY_SIZE 256  // 16x16 = 256 bytes
uint8_t receivedData1[16][16];  // First frame
uint8_t receivedData2[16][16];  // Second frame
int dataCount = 0;  // Track the position in the array
bool isFirstFrame = true;  // Toggle between the two frames

void setup() {
  Serial.begin(115200);  // Initialize serial communication
  Serial.println("Ready to receive data...");
}

void computeOpticalFlow(int x, int y) {
  // Gradients
  int I_x = 0, I_y = 0, I_t = 0;

  // Local 3x3 window
  int G[2][2] = {0};  // Gradient matrix
  int b[2] = {0};     // RHS vector

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int xCoord = x + i;
      int yCoord = y + j;

      if (xCoord >= 0 && xCoord < 16 && yCoord >= 0 && yCoord < 16) {
        I_x = receivedData1[yCoord][xCoord + 1] - receivedData1[yCoord][xCoord - 1];
        I_y = receivedData1[yCoord + 1][xCoord] - receivedData1[yCoord - 1][xCoord];
        I_t = receivedData2[yCoord][xCoord] - receivedData1[yCoord][xCoord];

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
  int det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
  if (det == 0) {
    Serial.println("No flow detected (singular matrix).");
    return;
  }

  // Solve for (u, v) using Cramer's rule
  int u = -(b[0] * G[1][1] - b[1] * G[0][1]) / det;
  int v = -(b[1] * G[0][0] - b[0] * G[1][0]) / det;

  // Send the optical flow vector back
  Serial.write((uint8_t)u);
  Serial.write((uint8_t)v);
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
      }

      // Toggle to the next frame and reset counter
      isFirstFrame = !isFirstFrame;
      dataCount = 0;
    }
  }
}
