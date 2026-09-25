#define OL __attribute__((overloadable))
OL void intel_sub_group_2d_block_read_32b_8r16x1c(global void*, int, int, int, int2, private uint*);
OL void intel_sub_group_2d_block_read_32b_16r16x1c(global void*, int, int, int, int2, private uint*);
OL float intel_sub_group_shuffle(float x, uint c);
#define NB (TN/16)
/* Y[T][N] = X[T][K] . Wt[K][N], F32. Sub-group tile TM x TN; group WM x WN sub-groups. */
__attribute__((intel_reqd_sub_group_size(16)))
kernel void sgemm(global const float *X, global const float *W, global float *Y, int T, int K, int N) {
  int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  int gn = (N + TN*WN - 1) / (TN*WN), g = get_group_id(0);
  int n0 = ((g % gn) * WN + sg % WN) * TN;
  int m0 = ((g / gn) * WM + sg / WN) * TM;
  if (n0 >= N || m0 >= T) return;
  float acc[TM][NB];
  _Pragma("unroll") for (int i = 0; i < TM; i++) _Pragma("unroll") for (int j = 0; j < NB; j++) acc[i][j] = 0;
  for (int k = 0; k < K; k += 16) {
#ifdef UNIFORM
    float16 ar[TM];
    _Pragma("unroll") for (int i = 0; i < TM; i++) ar[i] = vload16(0, X + (size_t)min(m0 + i, T - 1) * K + k);
#else
    float a[TM];
#if TM >= 16
    _Pragma("unroll") for (int i = 0; i < TM; i += 16)
      intel_sub_group_2d_block_read_32b_16r16x1c((global void*)X, K*4, T, K*4, (int2)(k, m0 + i), (private uint*)&a[i]);
#else
    intel_sub_group_2d_block_read_32b_8r16x1c((global void*)X, K*4, T, K*4, (int2)(k, m0), (private uint*)a);
#endif
#endif
    float b[NB][16];
    _Pragma("unroll") for (int j = 0; j < NB; j++)
      intel_sub_group_2d_block_read_32b_16r16x1c((global void*)W, N*4, K, N*4, (int2)(n0 + 16*j, k), (private uint*)b[j]);
    _Pragma("unroll") for (int kk = 0; kk < 16; kk++)
      _Pragma("unroll") for (int i = 0; i < TM; i++) {
#ifdef UNIFORM
        float ai = ar[i][kk];
#else
        float ai = intel_sub_group_shuffle(a[i], kk);
#endif
        _Pragma("unroll") for (int j = 0; j < NB; j++) acc[i][j] = fma(ai, b[j][kk], acc[i][j]);
      }
  }
  _Pragma("unroll") for (int i = 0; i < TM; i++) {
    int t = m0 + i; if (t >= T) break;
    _Pragma("unroll") for (int j = 0; j < NB; j++) Y[(size_t)t * N + n0 + 16*j + lane] = acc[i][j];
  }
}
