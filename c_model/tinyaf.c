#include "tinyaf.h"
#include <math.h>
#include <string.h>
#ifndef MODEL_WEIGHTS_HEADER
#define MODEL_WEIGHTS_HEADER "TINYAF_S_weights.h"
#endif
#include MODEL_WEIGHTS_HEADER
#if TINYAF_OPTIMIZED
#define R(x) x##f
#define SQRT sqrtf
#define ABS fabsf
#define EXP expf
#else
#define R(x) x
#define SQRT sqrt
#define ABS fabs
#define EXP exp
#endif
#include "filter_coefficients.h"
#define N 1000
#define P TINYAF_PEAK_CAPACITY
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
static TinyAFWorkspace default_workspace;
size_t tinyaf_workspace_size(void) {
    return sizeof(TinyAFWorkspace);
}
float tinyaf_default_threshold(void) {
    return MODEL_THRESHOLD;
}

/* NumPy contiguous pairwise summation, including its eight-lane base case. */
static TinyAFReal pair_sum(const TinyAFReal *x, int n) {
    if (n < 8) {
        TinyAFReal s = -R(0.0);
        for (int i = 0; i < n; i++)
            s += x[i];
        return s;
    }
    if (n <= 128) {
        TinyAFReal a[8];
        for (int k = 0; k < 8; k++)
            a[k] = x[k];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++)
                a[k] += x[i + k];
        TinyAFReal s = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        for (; i < n; i++)
            s += x[i];
        return s;
    }
    int mid = n / 2;
    mid -= mid % 8;
    return pair_sum(x, mid) + pair_sum(x + mid, n - mid);
}

static const FilterCoefficients *coefficients(int fs) {
    for (size_t i = 0; i < sizeof(FILTERS) / sizeof(FILTERS[0]); i++)
        if (FILTERS[i].fs == fs)
            return &FILTERS[i];
    return NULL;
}

/* SciPy's sosfiltfilt default odd extension: 3*(2*1+1)=9 samples.
 * Exact steady-state zi replaces the old 2048-sample warm-up. */
static void filtfilt(TinyAFReal *x, int n, const FilterCoefficients *c) {
    TinyAFReal z0 = c->z0 * x[0], z1 = c->z1 * x[0];
    for (int i = 0; i < n; i++) {
        TinyAFReal v = x[i], y = c->b0 * v + z0;
        z0 = c->b1 * v - c->a1 * y + z1;
        z1 = c->b2 * v - c->a2 * y;
        x[i] = y;
    }
    z0 = c->z0 * x[n - 1];
    z1 = c->z1 * x[n - 1];
    for (int i = n - 1; i >= 0; i--) {
        TinyAFReal v = x[i], y = c->b0 * v + z0;
        z0 = c->b1 * v - c->a1 * y + z1;
        z1 = c->b2 * v - c->a2 * y;
        x[i] = y;
    }
}

static TinyAFReal interp(const TinyAFReal *x, int n, int i, int m, TinyAFReal span) {
    if (i == m - 1)
        return x[n - 1];
    TinyAFReal step = span / (TinyAFReal)(n - 1), t = (TinyAFReal)i * (span / (TinyAFReal)(m - 1));
    int j = (int)(t / step);
    if (j >= n - 1)
        return x[n - 1];
    while (j > 0 && (TinyAFReal)j * step > t)
        j--;
    while (j < n - 2 && (TinyAFReal)(j + 1) * step <= t)
        j++;
    TinyAFReal t0 = (TinyAFReal)j * step;
    if (t == t0)
        return x[j];
    TinyAFReal t1 = (j + 1 == n - 1) ? span : (TinyAFReal)(j + 1) * step;
    TinyAFReal slope = (x[j + 1] - x[j]) / (t1 - t0);
    return slope * (t - t0) + x[j];
}

static int condition_loaded(int n, int fs, TinyAFWorkspace *w) {
    const FilterCoefficients *c = coefficients(fs);
    if (!c)
        return TINYAF_E_RATE;
    TinyAFReal *x = w->scratch.filter;
    for (int i = 0; i < 9; i++) {
        x[i] = R(2.0) * x[9] - x[18 - i];
        x[9 + n + i] = R(2.0) * x[9 + n - 1] - x[9 + n - 2 - i];
    }
    filtfilt(x, n + 18, c);
    for (int i = 0; i < N; i++)
        x[i] = interp(x + 9, n, i, N, R(10.0));
    TinyAFReal prev = x[0];
    x[0] = R(0.0);
    for (int i = 1; i < N; i++) {
        TinyAFReal v = x[i];
        x[i] = v - prev + R(0.9686) * x[i - 1];
        prev = v;
    }
    TinyAFReal mean = pair_sum(x, N) / (TinyAFReal)N;
    /* Keep DC-blocked values until std is known. Second scratch pass is not
     * needed: square deviations in the free tail of the same filter buffer. */
    TinyAFReal sum;
    if (TINYAF_MAX_INPUT_SAMPLES + 18 >= 2 * N) {
        for (int i = 0; i < N; i++) {
            TinyAFReal d = x[i] - mean;
            x[N + i] = d * d;
        }
        sum = pair_sum(x + N, N);
    } else {
        sum = 0;
        for (int i = 0; i < N; i++) {
            TinyAFReal d = x[i] - mean;
            sum += d * d;
        }
    }
    TinyAFReal den = SQRT(sum / (TinyAFReal)N) + R(1e-8);
    for (int i = 0; i < N; i++)
        w->trace[i] = (float)((x[i] - mean) / den);
    return 0;
}

static void boxcar(const TinyAFReal *x, TinyAFReal *y, int size) {
    int left = size / 2;
    TinyAFReal sum = 0;
    for (int k = 0; k < size; k++)
        sum += x[MAX(0, k - left)];
    y[0] = sum / (TinyAFReal)size;
    for (int i = 1; i < N; i++) {
        sum += x[MIN(N - 1, i + size - left - 1)] - x[MAX(0, i - left - 1)];
        y[i] = sum / (TinyAFReal)size;
    }
}
static TinyAFReal prominence(const float *x, int n, int p) {
    TinyAFReal v = x[p], l = v, r = v;
    for (int i = p - 1; i >= 0; i--) {
        if (x[i] > v)
            break;
        if (x[i] < l)
            l = x[i];
    }
    for (int i = p + 1; i < n; i++) {
        if (x[i] > v)
            break;
        if (x[i] < r)
            r = x[i];
    }
    return v - MAX(l, r);
}
static int best_peak(const float *x, int n) {
    int best = -1;
    TinyAFReal score = -R(1.0);
    for (int i = 1; i < n - 1;) {
        if (x[i - 1] < x[i]) {
            int j = i + 1;
            while (j < n && x[j] == x[i])
                j++;
            if (j < n && x[j] < x[i]) {
                int p = (i + j - 1) / 2;
                TinyAFReal s = prominence(x, n, p);
                if (s > score) {
                    best = p;
                    score = s;
                }
            }
            i = j;
        } else
            i++;
    }
    return best;
}
static int detect(TinyAFWorkspace *w) {
    TinyAFReal *a = w->scratch.detection.a, *b = w->scratch.detection.b,
               *c = w->scratch.detection.a;
    const float *x = w->trace;
    a[0] = ABS((TinyAFReal)x[1] - (TinyAFReal)x[0]);
    for (int i = 1; i < N - 1; i++)
        a[i] = ABS(((TinyAFReal)x[i + 1] - (TinyAFReal)x[i - 1]) * R(0.5));
    a[N - 1] = ABS((TinyAFReal)x[N - 1] - (TinyAFReal)x[N - 2]);
    boxcar(a, b, 10);
    boxcar(b, c, 75);
    uint16_t *beg = w->scratch.detection.beg, *end = w->scratch.detection.end;
    int nb = 0, ne = 0, last = b[0] > R(1.5) * c[0];
    for (int i = 1; i < N; i++) {
        int q = b[i] > R(1.5) * c[i];
        if (!last && q)
            beg[nb++] = (uint16_t)(i - 1);
        if (last && !q)
            end[ne++] = (uint16_t)(i - 1);
        last = q;
    }
    if (!nb)
        return 0;
    int skip = 0;
    while (skip < ne && end[skip] <= beg[0])
        skip++;
    int count = MIN(nb, ne - skip);
    if (!count)
        return 0;
    TinyAFReal sum = 0;
    for (int i = 0; i < count; i++)
        sum += (TinyAFReal)(end[skip + i] - beg[i]);
    TinyAFReal minlen = (sum / (TinyAFReal)count) * R(0.4);
    int n = 0, prev = 0;
    for (int i = 0; i < count; i++) {
        int len = end[skip + i] - beg[i];
        if (len < minlen)
            continue;
        int p = best_peak(x + beg[i], len);
        if (p >= 0 && (p += beg[i]) - prev > 30) {
            if (n >= P)
                return TINYAF_E_CAPACITY;
            w->peaks[n++] = p;
            prev = p;
        }
    }
    return n;
}

/* Kubios/Lipponen artifact correction. Fixed capacity and sorted small windows.
 * Numeric and index conventions follow installed NeuroKit 0.2.12. */
static void sort_real(TinyAFReal *x, int n) {
    for (int i = 1; i < n; i++) {
        TinyAFReal v = x[i];
        int j = i;
        while (j > 0 && x[j - 1] > v) {
            x[j] = x[j - 1];
            j--;
        }
        x[j] = v;
    }
}
static void sort_int(int *x, int n) {
    for (int i = 1; i < n; i++) {
        int v = x[i], j = i;
        while (j > 0 && x[j - 1] > v) {
            x[j] = x[j - 1];
            j--;
        }
        x[j] = v;
    }
}
static TinyAFReal quantile(const TinyAFReal *x, int n, TinyAFReal q) {
    TinyAFReal p = (TinyAFReal)(n - 1) * q;
    int lo = (int)p, hi = MIN(lo + 1, n - 1);
    return x[lo] + (x[hi] - x[lo]) * (p - (TinyAFReal)lo);
}
static void rolling(const TinyAFReal *x, int n, int width, int threshold, TinyAFReal *out,
                    TinyAFReal *tmp) {
    for (int i = 0; i < n; i++) {
        int m = 0;
        for (int j = MAX(0, i - width / 2); j < MIN(n, i + width / 2 + 1); j++)
            if (isfinite(x[j]))
                tmp[m++] = threshold ? ABS(x[j]) : x[j];
        if (!m) {
            out[i] = NAN;
            continue;
        }
        sort_real(tmp, m);
        out[i] = threshold
                     ? R(5.2) * ((quantile(tmp, m, R(0.75)) - quantile(tmp, m, R(0.25))) / R(2.0))
                     : quantile(tmp, m, R(0.5));
    }
}
static int reflect(int i, int n) {
    if (n < 2)
        return 0;
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * n - 2 - i;
    }
    return i;
}
static TinyAFReal npmin(TinyAFReal a, TinyAFReal b) {
    return isnan(a) || isnan(b) ? NAN : MIN(a, b);
}
static TinyAFReal npmax(TinyAFReal a, TinyAFReal b) {
    return isnan(a) || isnan(b) ? NAN : MAX(a, b);
}
static int artifacts(int n, TinyAFWorkspace *w) {
    TinyAFReal *rr = w->scratch.correction.rr, *d = w->scratch.correction.drr,
               *th1 = w->scratch.correction.th1, *s12 = w->scratch.correction.s12,
               *s22 = w->scratch.correction.s22, *med = w->scratch.correction.med,
               *m = w->scratch.correction.mrr, *th2 = w->scratch.correction.th2,
               *tmp = w->scratch.correction.temp;
    int(*a)[128] = w->scratch.correction.artifact, *counts = w->scratch.correction.count;
    memset(counts, 0, 4 * sizeof(int));
    if (n < 1)
        return 0;
    rr[0] = R(1.0);
    for (int i = 1; i < n; i++)
        rr[i] = (TinyAFReal)(w->peaks[i] - w->peaks[i - 1]) / R(100.0);
    if (n > 1)
        rr[0] = pair_sum(rr + 1, n - 1) / (TinyAFReal)(n - 1);
    d[0] = 0;
    for (int i = 1; i < n; i++)
        d[i] = rr[i] - rr[i - 1];
    if (n > 1)
        d[0] = pair_sum(d + 1, n - 1) / (TinyAFReal)(n - 1);
    rolling(d, n, 91, 1, th1, tmp);
    for (int i = 0; i < n; i++)
        d[i] = th1[i] != 0 ? d[i] / th1[i] : NAN;
    for (int i = 0; i < n; i++) {
        s12[i] = s22[i] = 0;
        if (d[i] > 0)
            s12[i] = npmax(d[reflect(i - 1, n)], d[reflect(i + 1, n)]);
        else if (d[i] < 0)
            s12[i] = npmin(d[reflect(i - 1, n)], d[reflect(i + 1, n)]);
        if (d[i] >= 0)
            s22[i] = npmin(d[reflect(i + 1, n)], d[reflect(i + 2, n)]);
        else if (d[i] < 0)
            s22[i] = npmax(d[reflect(i + 1, n)], d[reflect(i + 2, n)]);
    }
    rolling(rr, n, 11, 0, med, tmp);
    for (int i = 0; i < n; i++) {
        m[i] = rr[i] - med[i];
        if (m[i] < 0)
            m[i] *= R(2.0);
    }
    rolling(m, n, 91, 1, th2, tmp);
    for (int i = 0; i < n; i++)
        m[i] = th2[i] != 0 ? m[i] / th2[i] : NAN;
    int i = 0;
    while (i < n - 2) {
        if (ABS(d[i]) <= R(1.0)) {
            i++;
            continue;
        }
        if ((d[i] > R(1.0) && s12[i] < -R(0.13) * d[i] - R(0.17)) ||
            (d[i] < -R(1.0) && s12[i] > -R(0.13) * d[i] + R(0.17))) {
            if (counts[0] >= P)
                return TINYAF_E_CAPACITY;
            a[0][counts[0]++] = i++;
            continue;
        }
        if (!(ABS(d[i]) > R(1.0) || ABS(m[i]) > R(3.0))) {
            i++;
            continue;
        }
        int candidates[2] = {i, i + 1}, nc = ABS(d[i + 1]) < ABS(d[i + 2]) ? 2 : 1;
        for (int k = 0; k < nc; k++) {
            int j = candidates[k];
            int eq3 = d[j] > R(1.0) && s22[j]<-R(1.0), eq4 = ABS(m[j])> R(3.0),
                eq5 = d[j] < -R(1.0) && s22[j] > R(1.0);
            if (!(eq3 || eq4 || eq5)) {
                i++;
                continue;
            }
            int kind = 3;
            if (eq5 && ABS(rr[j] + rr[j + 1] - med[j]) < th2[j])
                kind = 2;
            else if (eq3 && ABS(rr[j] / R(2.0) - med[j]) < th2[j])
                kind = 1;
            if (counts[kind] >= P)
                return TINYAF_E_CAPACITY;
            a[kind][counts[kind]++] = j;
            i++;
        }
    }
    return counts[0] + counts[1] + counts[2] + counts[3];
}
static int contains(const int *x, int n, int value) {
    for (int i = 0; i < n; i++)
        if (x[i] == value)
            return 1;
    return 0;
}
static void update_indices(const int *src, int ns, int *x, int *n, int delta) {
    for (int s = 0; s < ns; s++)
        for (int i = 0; i < *n; i++)
            if (x[i] > src[s])
                x[i] += delta;
    sort_int(x, *n);
    int m = 0;
    for (int i = 0; i < *n; i++)
        if (!i || x[i] != x[i - 1])
            x[m++] = x[i];
    *n = m;
}
static int correct_once(int n, TinyAFWorkspace *w) {
    int(*a)[128] = w->scratch.correction.artifact, *count = w->scratch.correction.count,
    *tmp = w->scratch.correction.corrected, *peaks = w->peaks;
    if (count[2]) {
        int m = 0;
        for (int i = 0; i < n; i++)
            if (!contains(a[2], count[2], i))
                tmp[m++] = peaks[i];
        n = m;
        memcpy(peaks, tmp, (size_t)n * sizeof(int));
        for (int k = 0; k < 4; k++)
            if (k != 2)
                update_indices(a[2], count[2], a[k], &count[k], -1);
    }
    if (count[1]) {
        int m = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < count[1]; j++)
                if (a[1][j] == i && i > 1) {
                    if (m >= P)
                        return TINYAF_E_CAPACITY;
                    tmp[m++] = peaks[i - 1] + (peaks[i] - peaks[i - 1]) / 2;
                }
            if (m >= P)
                return TINYAF_E_CAPACITY;
            tmp[m++] = peaks[i];
        }
        n = m;
        memcpy(peaks, tmp, (size_t)n * sizeof(int));
        update_indices(a[1], count[1], a[0], &count[0], 1);
        update_indices(a[1], count[1], a[3], &count[3], 1);
    }
    for (int k = 0; k < 4; k += 3)
        if (count[k]) {
            int m = 0;
            for (int i = 0; i < n; i++)
                if (!(i > 1 && i < n - 1 && contains(a[k], count[k], i)))
                    tmp[m++] = peaks[i];
            for (int j = 0; j < count[k]; j++) {
                int i = a[k][j];
                if (i > 1 && i < n - 1) {
                    if (m >= P)
                        return TINYAF_E_CAPACITY;
                    tmp[m++] = peaks[i - 1] + (peaks[i + 1] - peaks[i - 1]) / 2;
                }
            }
            sort_int(tmp, m);
            n = m;
            memcpy(peaks, tmp, (size_t)n * sizeof(int));
        }
    return n;
}
static int correct(int n, TinyAFWorkspace *w) {
    int count = artifacts(n, w);
    if (count < 0)
        return count;
    n = correct_once(n, w);
    if (n < 0)
        return n;
    for (;;) {
        int prev = count;
        count = artifacts(n, w);
        if (count < 0)
            return count;
        if (count >= prev)
            break;
        n = correct_once(n, w);
        if (n < 0)
            return n;
    }
    return n;
}

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}
static void conv(const float *x, int ci, int length, const float *weights, const float *bias,
                 int co, int outlen, int kernel, int stride, int pad, float *out) {
    for (int o = 0; o < co; o++)
        for (int t = 0; t < outlen; t++) {
            float s = 0.0f;
            for (int c = 0; c < ci; c++)
                for (int k = 0; k < kernel; k++) {
                    int p = t * stride + k - pad;
                    if (p >= 0 && p < length)
                        s += weights[(o * ci + c) * kernel + k] * x[c * length + p];
                }
            out[o * outlen + t] = relu(s + bias[o]);
        }
}
static void dense(const float *x, int n, const float *weights, const float *bias, int m, float *y,
                  int activation) {
    for (int o = 0; o < m; o++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++)
            s += weights[o * n + j] * x[j];
        s += bias[o];
        y[o] = activation ? relu(s) : s;
    }
}
static void beat_samples(const float *trace, int a, int b, float *out) {
    int n = b - a;
    for (int i = 0; i < 100; i++) {
        if (i == 99) {
            out[i] = trace[b - 1];
            continue;
        }
        TinyAFReal step = R(1.0) / (TinyAFReal)(n - 1), t = (TinyAFReal)i * (R(1.0) / R(99.0));
        int j = (int)(t / step);
        j = MIN(j, n - 2);
        while (j > 0 && (TinyAFReal)j * step > t)
            j--;
        while (j < n - 2 && (TinyAFReal)(j + 1) * step <= t)
            j++;
        TinyAFReal left = (TinyAFReal)j * step,
                   right = j + 1 == n - 1 ? R(1.0) : (TinyAFReal)(j + 1) * step;
        TinyAFReal v = trace[a + j];
        if (t != left)
            v += (((TinyAFReal)trace[a + j + 1] - v) / (right - left)) * (t - left);
        out[i] = (float)v;
    }
}
static int valid_intervals(int n, TinyAFWorkspace *w) {
    int m = 0;
    for (int i = 0; i < n; i++)
        if (w->peaks[i] >= 0 && w->peaks[i] < N)
            w->peaks[m++] = w->peaks[i];
    return m;
}
static float model(int n, TinyAFWorkspace *w) {
    float *c1 = w->scratch.model.conv1, *c2 = w->scratch.model.conv2,
          *emb = w->scratch.model.embedding, *seq = w->scratch.model.sequence,
          *pool = w->scratch.model.pooled, *hidden = w->scratch.model.hidden,
          *beat = w->scratch.model.beat;
    float rr_values[20] = {0};
    int a[20] = {0}, b[20] = {0}, live = 0;
    for (int j = 0; j < n - 1 && live < 20; j++) {
        if (w->peaks[j + 1] - w->peaks[j] < 2)
            continue;
        a[live] = w->peaks[j];
        b[live] = w->peaks[j + 1];
        rr_values[live] = (float)(b[live] - a[live]) / 100.0f;
        live++;
    }
    for (int j = 0; j < 20; j++) {
#if TINYAF_OPTIMIZED
        /* Every padding beat is identical. Encode it once and reuse its
         * embedding, including the empty-slot RR channel, for later slots. */
        if (j > live) {
            for (int c = 0; c <= EMBED; c++)
                emb[c * 20 + j] = emb[c * 20 + j - 1];
            continue;
        }
#endif
        if (j < live)
            beat_samples(w->trace, a[j], b[j], beat);
        else
            memset(beat, 0, 100 * sizeof(float));
        conv(beat, 1, 100, CONV1_WEIGHT, CONV1_BIAS, CONV1_OUT, 17, CONV1_KERNEL, CONV1_STRIDE,
             CONV1_PAD, c1);
        conv(c1, CONV1_OUT, 17, CONV2_WEIGHT, CONV2_BIAS, CONV2_OUT, 9, CONV2_KERNEL, CONV2_STRIDE,
             CONV2_PAD, c2);
        dense(c2, PER_BEAT, BEAT_ENC_WEIGHT, BEAT_ENC_BIAS, EMBED, hidden, 1);
        for (int c = 0; c < EMBED; c++)
            emb[c * 20 + j] = hidden[c];
        emb[EMBED * 20 + j] = (rr_values[j] - 0.783f) / (0.210f + 1e-8f);
    }
    conv(emb, EMBED + 1, 20, BEAT_CONV_WEIGHT, BEAT_CONV_BIAS, BEAT_CH, 20, 3, 1, 1, seq);
    const int starts[8] = {0, 2, 5, 7, 10, 12, 15, 17};
    for (int c = 0; c < BEAT_CH; c++)
        for (int bin = 0; bin < 8; bin++) {
            float num = 0.0f, den = 0.0f;
            for (int j = starts[bin]; j < starts[bin] + 3; j++)
                if (rr_values[j] != 0.0f) {
                    num += seq[c * 20 + j];
                    den += 1.0f;
                }
            /* Preserve the TWO pool divisions even though sum/live is algebraically
             * equivalent: Torch rounds numerator and denominator independently. */
            num /= 3.0f;
            den /= 3.0f;
            pool[c * 8 + bin] = num / (den > 1e-6f ? den : 1e-6f);
        }
    dense(pool, BEAT_CH * 8, FC1_WEIGHT, FC1_BIAS, HIDDEN, hidden, 1);
    float logit;
    dense(hidden, HIDDEN, FC_WEIGHT, FC_BIAS, 1, &logit, 0);
    return logit;
}
static int finish(TinyAFWorkspace *w, AfibPredictionResult *out) {
    int n = detect(w);
    if (n < 0)
        return n;
    if (n == 0)
        return TINYAF_E_NO_BEATS;
    n = correct(n, w);
    if (n < 0)
        return n;
    n = valid_intervals(n, w);
    int live = 0;
    for (int j = 0; j < n - 1 && live < 20; j++)
        if (w->peaks[j + 1] - w->peaks[j] >= 2)
            live++;
    if (!live)
        return TINYAF_E_NO_BEATS;
    float logit = model(n, w);
    if (!isfinite(logit))
        return TINYAF_E_NONFINITE;
    TinyAFReal z = EXP(-ABS((TinyAFReal)logit));
    TinyAFReal probability = logit >= 0 ? R(1.0) / (R(1.0) + z) : z / (R(1.0) + z);
    out->ok = 1;
    out->n_beats = live;
    out->logit = logit;
    out->probability = (float)probability;
    out->prediction = probability >= (TinyAFReal)out->threshold;
    return 0;
}
static int initialize_result(AfibPredictionResult *out, int n, int fs, float threshold) {
    if (!out)
        return TINYAF_E_ARGUMENT;
    memset(out, 0, sizeof(*out));
    out->raw_len = n;
    out->sampling_rate = fs;
    out->prediction = -1;
    out->logit = out->probability = NAN;
    out->threshold = threshold == TINYAF_USE_DEFAULT_THRESHOLD ? MODEL_THRESHOLD : threshold;
    if (!isfinite(out->threshold) || out->threshold < 0.0f || out->threshold > 1.0f)
        return TINYAF_E_THRESHOLD;
    return 0;
}
int tinyaf_predict_adc(const uint16_t *raw, int n, float threshold, AfibPredictionResult *out,
                       TinyAFWorkspace *w) {
    int status = initialize_result(out, n, 256, threshold);
    if (status)
        return status;
    if (!raw || !w)
        return TINYAF_E_ARGUMENT;
    if (n != 2560 || n > TINYAF_MAX_INPUT_SAMPLES)
        return TINYAF_E_LENGTH;
    for (int i = 0; i < n; i++) {
        if (raw[i] > 4095)
            return TINYAF_E_ADC;
#if TINYAF_OPTIMIZED
        /* Removing a constant before the DC-rejecting pipeline avoids losing
         * small ECG differences to the unsigned ADC baseline in fp32. */
        w->scratch.filter[i + 9] = (TinyAFReal)((int)raw[i] - (int)raw[0]) * R(0.0030303);
#else
        w->scratch.filter[i + 9] = (TinyAFReal)raw[i] * R(0.0030303);
#endif
    }
    status = condition_loaded(n, 256, w);
    return status ? status : finish(w, out);
}
#if TINYAF_ENABLE_NATIVE
static int load_native(const double *raw, int n, int fs, TinyAFWorkspace *w) {
    if (!raw || !w)
        return TINYAF_E_ARGUMENT;
    if (fs <= 0 || fs > TINYAF_MAX_INPUT_SAMPLES / 10 || n != 10 * fs ||
        n > TINYAF_MAX_INPUT_SAMPLES || n < 10)
        return TINYAF_E_LENGTH;
    for (int i = 0; i < n; i++) {
        if (!isfinite(raw[i]))
            return TINYAF_E_NONFINITE;
        TinyAFReal value = (TinyAFReal)raw[i];
        if (!isfinite(value))
            return TINYAF_E_NONFINITE;
#if TINYAF_OPTIMIZED
        value -= (TinyAFReal)raw[0];
#endif
        w->scratch.filter[i + 9] = value;
    }
    return condition_loaded(n, fs, w);
}
int tinyaf_predict_native(const double *raw, int n, int fs, float threshold,
                          AfibPredictionResult *out, TinyAFWorkspace *w) {
    int status = initialize_result(out, n, fs, threshold);
    if (status)
        return status;
    status = load_native(raw, n, fs, w);
    return status ? status : finish(w, out);
}
#endif
int predict_c(const uint16_t *raw, int n, float threshold, float *prob) {
    if (!prob)
        return TINYAF_E_ARGUMENT;
    *prob = NAN;
    AfibPredictionResult out;
    int status = tinyaf_predict_adc(raw, n, threshold, &out, &default_workspace);
    if (!status)
        *prob = out.probability;
    return status;
}
#ifdef TINYAF_DIAGNOSTICS
int tinyaf_debug_native(const double *raw, int n, int fs, float *trace, float *beats, float *rr,
                        int *peaks, TinyAFWorkspace *w) {
    int status = load_native(raw, n, fs, w);
    if (status)
        return status;
    memcpy(trace, w->trace, 1000 * sizeof(float));
    n = detect(w);
    if (n < 0)
        return n;
    n = correct(n, w);
    if (n < 0)
        return n;
    n = valid_intervals(n, w);
    memcpy(peaks, w->peaks, (size_t)n * sizeof(int));
    memset(beats, 0, 2000 * sizeof(float));
    memset(rr, 0, 20 * sizeof(float));
    int k = 0;
    for (int i = 0; i < n - 1 && k < 20; i++)
        if (w->peaks[i + 1] - w->peaks[i] >= 2) {
            beat_samples(w->trace, w->peaks[i], w->peaks[i + 1], beats + k * 100);
            rr[k++] = (float)(w->peaks[i + 1] - w->peaks[i]);
        }
    return n;
}
int tinyaf_debug_correct(int *peaks, int n, TinyAFWorkspace *w) {
    if (n < 0 || n > P)
        return TINYAF_E_CAPACITY;
    memcpy(w->peaks, peaks, (size_t)n * sizeof(int));
    n = correct(n, w);
    if (n >= 0)
        memcpy(peaks, w->peaks, (size_t)n * sizeof(int));
    return n;
}
#endif
