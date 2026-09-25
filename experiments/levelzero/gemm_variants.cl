#define OL __attribute__((overloadable))
OL float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);
OL void intel_sub_group_2d_block_read_16b_8r16x1c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_16b_8r16x2c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_16b_16r16x2c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_16b_32r16x2c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_16b_32r16x1c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_transform_16b_32r16x1c(global void*, int, int, int, int2, private uint*);
OL void intel_sub_group_2d_block_read_transform_16b_32r16x2c(global void*, int, int, int, int2, private uint*);
OL void intel_sub_group_2d_block_read_transform_16b_16r16x1c(global void*, int, int, int, int2, private uint*);
OL void intel_sub_group_2d_block_prefetch_16b_32r16x2c(global void*, int, int, int, int2);
OL void intel_sub_group_2d_block_prefetch_16b_32r16x1c(global void*, int, int, int, int2);
OL void intel_sub_group_2d_block_prefetch_16b_8r16x2c(global void*, int, int, int, int2);
OL void intel_sub_group_2d_block_write_32b_8r16x1c(global void*, int, int, int, int2, private uint*);

/* Sub-group tile TM x TN, work-group WM x WN sub-groups (n fastest), K step KSTEP (16 or 32),
 * split-K over get_group_id(1): partial sums at Y + split*T*N. */
#define MB (TM/8)
#define NB (TN/16)
__attribute__((intel_reqd_sub_group_size(16)))
kernel void gemm(global const half *X, global const half *W, global float *Y, int T, int K, int N) {
  int sg = get_sub_group_id();
  int gn = (N + TN*WN - 1) / (TN*WN);
  int g = get_group_id(0);
  int n0 = ((g % gn) * WN + sg % WN) * TN;
  int m0 = ((g / gn) * WM + sg / WN) * TM;
  int splits = get_num_groups(1), ks = K / splits, kb = get_group_id(1) * ks;
#ifndef SB
  if (n0 >= N || m0 >= T) return;
#endif
  Y += (long)get_group_id(1) * T * N;
  float8 acc[MB][NB];
  _Pragma("unroll") for (int i = 0; i < MB; i++) _Pragma("unroll") for (int j = 0; j < NB; j++) acc[i][j] = 0;
  for (int k = kb; k < kb + ks; k += KSTEP) {
#if KSTEP == 64
    short8 a[4][MB]; int8 b[4][NB];
    #ifdef PFD
    if (k + PFD < K) {
      _Pragma("unroll") for (int i = 0; i < MB; i += 4)
        intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)X, K*2, T, K*2, (int2)(k + PFD, m0 + i*8));
      _Pragma("unroll") for (int j = 0; j < NB; j++)
        intel_sub_group_2d_block_prefetch_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + j*16, k + PFD)),
        intel_sub_group_2d_block_prefetch_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + j*16, k + PFD + 32));
    }
#endif
_Pragma("unroll") for (int q = 0; q < 2; q++) {
      _Pragma("unroll") for (int i = 0; i < MB; i += 2) {
        short8 t[4];
        intel_sub_group_2d_block_read_16b_16r16x2c((global void*)X, K*2, T, K*2, (int2)(k + 32*q, m0 + i*8), (private ushort*)t);
        a[2*q][i] = t[0]; a[2*q][i+1] = t[1]; a[2*q+1][i] = t[2]; a[2*q+1][i+1] = t[3];
      }
      _Pragma("unroll") for (int j = 0; j < NB; j++) {
        int8 t[2];
        intel_sub_group_2d_block_read_transform_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + j*16, k + 32*q), (private uint*)t);
        b[2*q][j] = t[0]; b[2*q+1][j] = t[1];
      }
    }
    _Pragma("unroll") for (int s = 0; s < 4; s++)
    _Pragma("unroll") for (int i = 0; i < MB; i++) _Pragma("unroll") for (int j = 0; j < NB; j++)
      acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[s][i], b[s][j], acc[i][j]);
#elif KSTEP == 32
    short8 a[2][MB]; int8 b[2][NB];
#if MB == 1
    { short8 t[2];
      intel_sub_group_2d_block_read_16b_8r16x2c((global void*)X, K*2, T, K*2, (int2)(k, m0), (private ushort*)t);
      a[0][0] = t[0]; a[1][0] = t[1]; }
#ifdef SB
    if (((k - kb) / 32) % (SB / 32) == (SB / 32) - 1) barrier(CLK_LOCAL_MEM_FENCE);
#endif
#ifdef CPF
    /* cooperative prefetch: sub-group sg prefetches a slice of the group's A and B tiles CPF ahead */
    if (k + CPF < kb + ks) {
      int nsg = WM * WN;
      /* A: group rows (TM*WM) x 32 k; B: 32 k x group cols (TN*WN) */
      int ga = (get_group_id(0) / gn) * WM * TM, gb = (get_group_id(0) % gn) * WN * TN;
      int rows_a = TM * WM, cols_b = TN * WN;
      for (int p = sg; p < rows_a / 8 + cols_b / 16; p += nsg) {
        if (p < rows_a / 8) intel_sub_group_2d_block_prefetch_16b_8r16x2c((global void*)X, K*2, T, K*2, (int2)(k + CPF, ga + 8 * p));
        else intel_sub_group_2d_block_prefetch_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(gb + 16 * (p - rows_a / 8), k + CPF));
      }
    }
#endif

#else
    _Pragma("unroll") for (int i = 0; i < MB; i += 2) {
      short8 t[4];
      intel_sub_group_2d_block_read_16b_16r16x2c((global void*)X, K*2, T, K*2, (int2)(k, m0 + i*8), (private ushort*)t);
      /* 16r16x2c: block 0 (cols k..k+15) rows 0..15, then block 1 */
      a[0][i] = t[0]; a[0][i+1] = t[1]; a[1][i] = t[2]; a[1][i+1] = t[3];
    }
#endif
    _Pragma("unroll") for (int j = 0; j < NB; j++) {
      int8 t[2];
      intel_sub_group_2d_block_read_transform_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + j*16, k), (private uint*)t);
      b[0][j] = t[0]; b[1][j] = t[1];
    }
    _Pragma("unroll") for (int s = 0; s < 2; s++)
    _Pragma("unroll") for (int i = 0; i < MB; i++) _Pragma("unroll") for (int j = 0; j < NB; j++)
      acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[s][i], b[s][j], acc[i][j]);
#else
    short8 a[MB]; int8 b[NB];
    _Pragma("unroll") for (int i = 0; i < MB; i++)
      intel_sub_group_2d_block_read_16b_8r16x1c((global void*)X, K*2, T, K*2, (int2)(k, m0 + i*8), (private ushort*)&a[i]);
    _Pragma("unroll") for (int j = 0; j < NB; j++)
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + j*16, k), (private uint*)&b[j]);
    _Pragma("unroll") for (int i = 0; i < MB; i++) _Pragma("unroll") for (int j = 0; j < NB; j++)
      acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], b[j], acc[i][j]);
#endif
  }
  _Pragma("unroll") for (int i = 0; i < MB; i++) _Pragma("unroll") for (int j = 0; j < NB; j++)
    intel_sub_group_2d_block_write_32b_8r16x1c((global void*)Y, N*4, T, N*4, (int2)(n0 + j*16, m0 + i*8), (private uint*)&acc[i][j]);
}
