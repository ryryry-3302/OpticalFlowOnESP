#include "HardwareSerial.h"
#define IMG_SIZE 16



void receive_frame(uint8_t frame[IMG_SIZE][IMG_SIZE]) {
    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            while (!Serial.available());
            frame[i][j] = Serial.read();
        }
    }
}

int8_t compute_gradient_x(uint8_t current_frame[16][16], int x, int y) {
    return current_frame[y][x + 1] - current_frame[y][x - 1];
}

int8_t compute_gradient_y(uint8_t current_frame[16][16], int x, int y) {
    return current_frame[y + 1][x] - current_frame[y - 1][x];
}

int8_t compute_gradient_t(uint8_t current_frame[16][16], uint8_t previous_frame[16][16], int x, int y) {
    return current_frame[y][x] - previous_frame[y][x];
}

void compute_optical_flow(uint8_t current_frame[16][16], uint8_t previous_frame[16][16], int target_x, int target_y, float *u, float *v) {
    int kernel_size = 3;
    int half_k = kernel_size / 2;

    float A[2][2] = {{0, 0}, {0, 0}};
    float B[2] = {0, 0};

    for (int dy = -half_k; dy <= half_k; dy++) {
        for (int dx = -half_k; dx <= half_k; dx++) {
            int x = target_x + dx;
            int y = target_y + dy;

            int8_t Ix = compute_gradient_x(current_frame, x, y);
            int8_t Iy = compute_gradient_y(current_frame, x, y);
            int8_t It = compute_gradient_t(current_frame, previous_frame, x, y);

            A[0][0] += Ix * Ix;
            A[0][1] += Ix * Iy;
            A[1][0] += Ix * Iy;
            A[1][1] += Iy * Iy;

            B[0] += -Ix * It;
            B[1] += -Iy * It;
        }
    }

    // Invert A (2x2 matrix inversion)
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (det != 0) {
        float A_inv[2][2] = {
            {A[1][1] / det, -A[0][1] / det},
            {-A[1][0] / det, A[0][0] / det}
        };

        // Solve for u and v
        *u = A_inv[0][0] * B[0] + A_inv[0][1] * B[1];
        *v = A_inv[1][0] * B[0] + A_inv[1][1] * B[1];
    } else {
        // If A is singular, set flow to zero
        *u = 0;
        *v = 0;
    }
}

