#include <motion.h>

#define ARRAY_SIZE 256  // 16x16 = 256 bytes
uint8_t receivedData1[16][16];  // First frame
uint8_t receivedData2[16][16];  // Second frame
int dataCount = 0;  // Track the position in the array
bool isFirstFrame = true;  // Toggle between the two frames

// Scaling factor for the optical flow
const int SCALE_FACTOR = 100;
static MotionEstContext* me_ctx = new MotionEstContext();



// Gradient kernels


// Convolution helper
#include <Arduino.h>

// Gradient kernels
const float IxKernel[2][2] = {{-0.25, 0.25}, {-0.25, 0.25}};
const float IyKernel[2][2] = {{-0.25, -0.25}, {0.25, 0.25}};
const float ItKernel1[2][2] = {{0.25, 0.25}, {0.25, 0.25}};  // img1
const float ItKernel2[2][2] = {{-0.25, -0.25}, {-0.25, -0.25}};  // img2

// Convolution helper for 2x2 kernel
float convolve2D(const uint8_t img[5][5], const float kernel[2][2], int x, int y) {
    float sum = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sum += img[x + i][y + j] * kernel[i][j];
        }
    }
    return sum;
}

// Optical flow calculation
void calculateOpticalFlow( int x, int y) {
    float Ix[5][5], Iy[5][5], It[5][5];
    float A[2][2] = {0}, b[2] = {0};
    const float SCALE_FACTOR = 100.0;

    // Ensure x, y are valid for a 5x5 neighborhood
    if (x < 2 || y < 2 || x > 13 || y > 13) {
        Serial.println("Error: Coordinates out of bounds for a 5x5 neighborhood.");
        return;
    }

    // Extract 5x5 window around (x, y)
    uint8_t window1[5][5], window2[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            window1[i][j] = receivedData1[x - 2 + i][y - 2 + j];
            window2[i][j] = receivedData2[x - 2 + i][y - 2 + j];
        }
    }

    // Calculate gradients
    for (int i = 0; i < 4; i++) {  // Gradient kernels are 2x2, so iterate only within valid range
        for (int j = 0; j < 4; j++) {
            Ix[i][j] = convolve2D(window1, IxKernel, i, j) + convolve2D(window2, IxKernel, i, j);
            Iy[i][j] = convolve2D(window1, IyKernel, i, j) + convolve2D(window2, IyKernel, i, j);
            It[i][j] = convolve2D(window2, ItKernel2, i, j) + convolve2D(window1, ItKernel1, i, j);
        }
    }

    // Populate A and b
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            A[0][0] += Ix[i][j] * Ix[i][j];
            A[0][1] += Ix[i][j] * Iy[i][j];
            A[1][1] += Iy[i][j] * Iy[i][j];
            b[0] += Ix[i][j] * It[i][j];
            b[1] += Iy[i][j] * It[i][j];
        }
    }
    A[1][0] = A[0][1];  // Symmetric matrix

    // Solve for u, v
    float u = 0.0, v = 0.0;
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (det != 0) {
        u = (A[1][1] * -b[0] - A[0][1] * -b[1]) / det;
        v = (A[0][0] * -b[1] - A[0][1] * -b[0]) / det;
    }

    // Scale and send results over Serial
    int scaled_u = (int)(u * SCALE_FACTOR);
    int scaled_v = (int)(v * SCALE_FACTOR);

    Serial.write((uint8_t)(scaled_u >> 8));  // High byte of u
    Serial.write((uint8_t)(scaled_u & 0xFF));  // Low byte of u
    Serial.write((uint8_t)(scaled_v >> 8));  // High byte of v
    Serial.write((uint8_t)(scaled_v & 0xFF));  // Low byte of v
}

void computeOpticalFlowLK(int x, int y){
  // Set the current and previous images for motion estimation
  uint8_t* img_prev = &receivedData1[0][0];  // Previous frame
  uint8_t* img_cur = &receivedData2[0][0];   // Current frame

  // Call the motion_estimation function
  int sum = 0;
  bool success = motion_estimation(me_ctx, img_prev, img_cur);
  MotionVector16_t *mv = me_ctx->mv_table[0] + 136;
  if (success) {
    for(int i = 0; i < 16*16; i++, mv++) 
        sum += (int) sqrtf(mv->mag2);
    // Scale the motion vector components (u and v) by the scaling factor
    int scaled_u = sum / (16*16);
    int scaled_v = (int)mv->vy * SCALE_FACTOR;
    // Send the scaled optical flow vector
    Serial.write((uint8_t)(scaled_u >> 8));  // High byte of u
    Serial.write((uint8_t)(scaled_u & 0xFF));  // Low byte of u
    Serial.write((uint8_t)(scaled_v >> 8));  // High byte of v
    Serial.write((uint8_t)(scaled_v & 0xFF));  // Low byte of v
  } else {
    // Handle motion estimation failure, if needed
    Serial.println("Motion estimation failed!");
  }
}

void computeOpticalFlowSimple(int x, int y){
  float I_x = 0.0f, I_y = 0.0f, I_t = 0.0f;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int xCoord = x + i;
      int yCoord = y + j;
      if (xCoord >= 0 && xCoord < 16 && yCoord >= 0 && yCoord < 16) {
        // Use int16_t to safely handle potential overflow when subtracting
        I_x += (float)((int16_t)receivedData1[yCoord][xCoord + 1] - (int16_t)receivedData1[yCoord][xCoord - 1]);
        I_y += (float)((int16_t)receivedData1[yCoord + 1][xCoord] - (int16_t)receivedData1[yCoord - 1][xCoord]);
        I_t += (float)((int16_t)receivedData2[yCoord][xCoord] - (int16_t)receivedData1[yCoord][xCoord]);


      }
    }
  }
  float u = I_x * I_t / (pow(I_x, 2) + pow(I_y, 2) + 1e-6);
  float v = I_y * I_t / (pow(I_x, 2) + pow(I_y, 2) + 1e-6);
  int scaled_u = (int)(u * SCALE_FACTOR);
  int scaled_v = (int)(v * SCALE_FACTOR);

  Serial.write((uint8_t)(scaled_u >> 8));  // High byte of u
  Serial.write((uint8_t)(scaled_u & 0xFF));  // Low byte of u
  Serial.write((uint8_t)(scaled_v >> 8));  // High byte of v
  Serial.write((uint8_t)(scaled_v & 0xFF));  // Low byte of v
}

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
  init_context(me_ctx);
  me_ctx->method = LK_OPTICAL_FLOW;            // algo used. LK_OPTICAL_FLOW; <- good too
  me_ctx->width = 16;                         // image width
  me_ctx->height = 16; 
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