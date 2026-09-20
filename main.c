#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 20
#define HEIGHT 20
#define PPM_SCALAR 30
#define RECT_TRAIN_SAMPLE_SIZE 91
#define CIRCLE_TRAIN_SAMPLE_SIZE 78
#define BIAS 20
#define TRAIN_PASS_COUNT 40

typedef float Layer[HEIGHT][WIDTH];

static inline int clampi(int value, int min, int max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

void layer_fill_rect(Layer layer, int x, int y, int w, int h, float value) {
  assert(w > 0);
  assert(h > 0);

  int x0 = clampi(x, 0, WIDTH - 1);
  int y0 = clampi(y, 0, HEIGHT - 1);
  int x1 = clampi(x0 + w - 1, 0, WIDTH - 1);
  int y1 = clampi(y0 + h - 1, 0, HEIGHT - 1);

  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      layer[y][x] = value;
    }
  }
}

void layer_fill_circle(Layer layer, int cx, int cy, int radius, float value) {
  assert(radius > 0);

  int x0 = clampi(cx - radius, 0, WIDTH - 1);
  int y0 = clampi(cy - radius, 0, HEIGHT - 1);
  int x1 = clampi(cx + radius, 0, WIDTH - 1);
  int y1 = clampi(cy + radius, 0, HEIGHT - 1);

  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      int dx = x - cx;
      int dy = y - cy;
      if (dx * dx + dy * dy <= radius * radius) {
        layer[y][x] = value;
      }
    }
  }
}

void layer_save_as_ppm(Layer layer, const char *file_path) {
  float min = layer[0][0];
  float max = layer[0][0];

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      float s = layer[y][x];
      if (s < min)
        min = s;
      if (s > max)
        max = s;
    }
  }

  FILE *f = fopen(file_path, "wb");

  if (f == NULL) {
    fprintf(stderr, "Error: Could not open file %s: %s\n", file_path,
            strerror(errno));
    exit(1);
  }

  fprintf(f, "P6\n%d %d\n255\n", WIDTH * PPM_SCALAR, HEIGHT * PPM_SCALAR);

  char pixel[3] = {0.0f, 0.0f, 0.0f};

  for (int y = 0; y < HEIGHT * PPM_SCALAR; ++y) {
    for (int x = 0; x < WIDTH * PPM_SCALAR; ++x) {
      float s = layer[y / PPM_SCALAR][x / PPM_SCALAR];

      float r = 0.0f;
      float b = 0.0f;
      if (s >= 0) {
        r = s / max;
      } else {
        b = s / min;
      }
      pixel[0] = (char)(r * 255.0f);
      pixel[2] = (char)(b * 255.0f);

      fwrite(pixel, sizeof(pixel), 1, f);
    }
  }

  fclose(f);
}

void layer_save_as_bin(Layer layer, const char *file_path) {
  FILE *f = fopen(file_path, "wb");
  if (f == NULL) {
    fprintf(stderr, "Error: Could not open file %s: %s\n", file_path,
            strerror(errno));
    exit(1);
  }

  fwrite(layer, sizeof(Layer), 1, f);

  fclose(f);
}

void layser_load_as_bin(Layer layer, const char *file_path) {
  FILE *f = fopen(file_path, "rb");
  if (f == NULL) {
    fprintf(stderr, "Error: Could not open file %s: %s\n", file_path,
            strerror(errno));
    exit(1);
  }

  fread(layer, sizeof(Layer), 1, f);

  fclose(f);
}

float feed_forward(Layer inputs, Layer weights) {
  float output = 0.0f;

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      output += inputs[y][x] * weights[y][x];
    }
  }

  return output;
}

void add_inputs_to_weights(Layer inputs, Layer weights) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      weights[y][x] += inputs[y][x];
    }
  }
}

void subtract_inputs_from_weights(Layer inputs, Layer weights) {
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      weights[y][x] -= inputs[y][x];
    }
  }
}

int rand_range(int min, int max) {
  assert(min <= max);

  if (min == max) {
    return min;
  }
  return min + rand() % (max - min);
}

void layer_random_rect(Layer layer) {
  layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
  int x = rand_range(0, WIDTH);
  int y = rand_range(0, HEIGHT);
  int w = rand_range(1, WIDTH - x + 1);
  int h = rand_range(1, HEIGHT - y + 1);
  layer_fill_rect(layer, x, y, w, h, 1.0f);
}

void layer_random_circle(Layer layer) {
  layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
  int cx = rand_range(0, WIDTH);
  int cy = rand_range(0, HEIGHT);
  int max_radius_cand1 = fmin(cx, cy) + 1;
  int max_radius_cand2 = fmin(WIDTH - cx, HEIGHT - cy) + 1;
  int max_radius = fmin(max_radius_cand1, max_radius_cand2);
  int radius = rand_range(1, max_radius);
  layer_fill_circle(layer, cx, cy, radius, 1.0f);
}

static Layer inputs;
static Layer weights;

void gen_layer_set(int rect_size, int circle_size, const char *output_dir) {
  char rect_bin[256];
  char rect_ppm[256];
  char circle_bin[256];
  char circle_ppm[256];

  snprintf(rect_bin, sizeof(rect_bin), "%s/rect/bin/", output_dir);
  snprintf(rect_ppm, sizeof(rect_ppm), "%s/rect/ppm/", output_dir);
  snprintf(circle_bin, sizeof(circle_bin), "%s/circle/bin/", output_dir);
  snprintf(circle_ppm, sizeof(circle_ppm), "%s/circle/ppm/", output_dir);

  char file_path[512];

  for (int i = 0; i < rect_size; ++i) {
    printf("[INFO] Generating rect %d\n", i);
    layer_random_rect(inputs);
    snprintf(file_path, sizeof(file_path), "%srect-%02d.bin", rect_bin, i);
    layer_save_as_bin(inputs, file_path);
    snprintf(file_path, sizeof(file_path), "%srect-%02d.ppm", rect_ppm, i);
    layer_save_as_ppm(inputs, file_path);
  }

  for (int i = 0; i < circle_size; ++i) {
    printf("[INFO] Generating circle %d\n", i);
    layer_random_circle(inputs);
    snprintf(file_path, sizeof(file_path), "%scircle-%02d.bin", circle_bin, i);
    layer_save_as_bin(inputs, file_path);
    snprintf(file_path, sizeof(file_path), "%scircle-%02d.ppm", circle_ppm, i);
    layer_save_as_ppm(inputs, file_path);
  }
}

int train_pass() {
  int rect_count = 0;
  int circle_count = 0;

  int choice = 0;
  int real = 0;
  char file_path[256];

  int cnt = 0;

  for (int i = 0; i < RECT_TRAIN_SAMPLE_SIZE + CIRCLE_TRAIN_SAMPLE_SIZE; ++i) {
    choice = rand() % 2;

    if (rect_count >= RECT_TRAIN_SAMPLE_SIZE) {
      choice = 1;
    } else if (circle_count >= CIRCLE_TRAIN_SAMPLE_SIZE) {
      choice = 0;
    }

    if (choice == 0) {
      snprintf(file_path, sizeof(file_path), "train/rect/bin/rect-%02d.bin",
               rect_count);
      layser_load_as_bin(inputs, file_path);
      rect_count++;
      real = 0;
    } else {
      snprintf(file_path, sizeof(file_path), "train/circle/bin/circle-%02d.bin",
               circle_count);
      layser_load_as_bin(inputs, file_path);
      circle_count++;
      real = 1;
    }

    printf("[TRAIN] Train on %s\n", file_path);

    float res = feed_forward(inputs, weights);

    if (res > BIAS && real == 0) {
      subtract_inputs_from_weights(inputs, weights);
      printf("[TRAIN] Subtract from weights on %s\n", file_path);
      cnt++;
    } else if (res < BIAS && real == 1) {
      add_inputs_to_weights(inputs, weights);
      printf("[TRAIN] Add to weights on %s\n", file_path);
      cnt++;
    }
  }

  layer_save_as_ppm(weights, "weights.ppm");
  return cnt;
}

int main(void) {
  srand(69);

  gen_layer_set(RECT_TRAIN_SAMPLE_SIZE, CIRCLE_TRAIN_SAMPLE_SIZE, "train");

  for (int i = 0; i < TRAIN_PASS_COUNT; ++i) {
    int cnt = train_pass();
    printf("[PASSES] %d\n", cnt);
  }

  return 0;
}
