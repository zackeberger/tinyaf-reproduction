#include "tinyaf.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s ADC_FILE.txt\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    uint16_t raw[2560];
    int value;
    for (int i = 0; i < 2560; i++) {
        if (fscanf(f, "%d", &value) != 1 || value < 0 || value > 4095) {
            fclose(f);
            return 2;
        }
        raw[i] = (uint16_t)value;
    }
    if (fscanf(f, "%d", &value) != EOF) {
        fclose(f);
        return 2;
    }
    fclose(f);
    float probability;
    int status = predict_c(raw, 2560, TINYAF_USE_DEFAULT_THRESHOLD, &probability);
    printf("status=%d probability=%.9g\n", status, probability);
    return status ? 1 : 0;
}
