#ifndef TINYAF_H
#define TINYAF_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef TINYAF_OPTIMIZED
#define TINYAF_OPTIMIZED 1
#endif
#ifndef TINYAF_ENABLE_NATIVE
#define TINYAF_ENABLE_NATIVE 1
#endif
#ifndef TINYAF_MAX_INPUT_SAMPLES
#define TINYAF_MAX_INPUT_SAMPLES 2560
#endif
#if TINYAF_OPTIMIZED
typedef float TinyAFReal;
#else
typedef double TinyAFReal;
#endif
#define TINYAF_QUANTA_SAMPLING_RATE_HZ 256
#define TINYAF_ECG_DURATION_SECONDS 10
#define TINYAF_QUANTA_SAMPLE_COUNT 2560
#define TINYAF_QUANTA_ADC_MAX 4095U
#define TINYAF_USE_DEFAULT_THRESHOLD (-1.0f)
#define TINYAF_PEAK_CAPACITY 128
/* Zero on success. No prediction is encoded as a negative status and NaNs. */
enum {
    TINYAF_OK = 0,
    TINYAF_E_ARGUMENT = -1,
    TINYAF_E_LENGTH = -2,
    TINYAF_E_ADC = -3,
    TINYAF_E_NO_BEATS = -4,
    TINYAF_E_THRESHOLD = -5,
    TINYAF_E_NONFINITE = -6,
    TINYAF_E_RATE = -7,
    TINYAF_E_CAPACITY = -8
};
typedef struct {
    int ok, raw_len, sampling_rate, n_beats;
    float logit, probability, threshold;
    int prediction;
} AfibPredictionResult;
/* Caller-owned reusable storage. Layout must use the same compile definitions
 * as tinyaf.c. No inference-time allocation; all overlapping stages share memory.
 * 128 corrected peaks are retained BEFORE truncating to 20 intervals. */
typedef struct {
    float trace[1000];
    int peaks[TINYAF_PEAK_CAPACITY];
    union {
        TinyAFReal filter[TINYAF_MAX_INPUT_SAMPLES + 18];
        struct {
            TinyAFReal a[1000], b[1000];
            uint16_t beg[500], end[500];
        } detection;
        struct {
            TinyAFReal rr[128], drr[128], th1[128], s12[128], s22[128], med[128], mrr[128],
                th2[128], temp[128];
            int artifact[4][128], count[4], corrected[128];
        } correction;
        struct {
            float conv1[136], conv2[144], embedding[340], sequence[320], pooled[128], hidden[32],
                beat[100];
        } model;
    } scratch;
} TinyAFWorkspace;
/* This entrypoint uses one static workspace. Calls must be serialized.
 * predict_c retains Quanta's existing signature. No inference-time allocation. */
int predict_c(const uint16_t *, int, float, float *);
size_t tinyaf_workspace_size(void);
float tinyaf_default_threshold(void);
int tinyaf_predict_adc(const uint16_t *, int, float, AfibPredictionResult *, TinyAFWorkspace *);
#if TINYAF_ENABLE_NATIVE
/* Evaluation-only physical input. Rate must match a compiled filter table entry. */
int tinyaf_predict_native(const double *, int, int, float, AfibPredictionResult *,
                          TinyAFWorkspace *);
#endif
#ifdef TINYAF_DIAGNOSTICS
int tinyaf_debug_native(const double *, int, int, float *, float *, float *, int *,
                        TinyAFWorkspace *);
int tinyaf_debug_correct(int *, int, TinyAFWorkspace *);
#endif
#ifdef __cplusplus
}
#endif
#endif
