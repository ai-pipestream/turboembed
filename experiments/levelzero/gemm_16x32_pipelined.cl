#define OL __attribute__((overloadable))
OL float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);
OL void intel_sub_group_2d_block_read_16b_16r16x2c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_transform_16b_32r16x1c(global void*, int, int, int, int2, private uint*);
OL void intel_sub_group_2d_block_write_32b_8r16x1c(global void*, int, int, int, int2, private uint*);
/* 16 x 32 a sub-group, k 32 a step, the next step's operands loaded before this step's products */
#define LOAD(k, A, B) do { \
    short8 ta_[4]; int8 tb0_[2], tb1_[2]; \
    intel_sub_group_2d_block_read_16b_16r16x2c((global void*)X, K*2, T, K*2, (int2)(k, m0), (private ushort*)ta_); \
    intel_sub_group_2d_block_read_transform_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(n0, k), (private uint*)tb0_); \
    intel_sub_group_2d_block_read_transform_16b_32r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + 16, k), (private uint*)tb1_); \
    A##0 = ta_[0]; A##1 = ta_[1]; A##2 = ta_[2]; A##3 = ta_[3]; \
    B##0 = tb0_[0]; B##1 = tb0_[1]; B##2 = tb1_[0]; B##3 = tb1_[1]; \
  } while (0)
/* A: [h0 rows0-7, h0 rows8-15, h1 rows0-7, h1 rows8-15]; B: [j0 h0, j0 h1, j1 h0, j1 h1] */
#define D(i, j, a, b) acc##i##j = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc##i##j)
#define MAD(A, B) do { D(0,0,A##0,B##0); D(0,1,A##0,B##2); D(1,0,A##1,B##0); D(1,1,A##1,B##2); \
                       D(0,0,A##2,B##1); D(0,1,A##2,B##3); D(1,0,A##3,B##1); D(1,1,A##3,B##3); } while (0)
__attribute__((intel_reqd_sub_group_size(16)))
kernel void gemm(global const half *X, global const half *W, global float *Y, int T, int K, int N) {
  int sg = get_sub_group_id();
  int gn = (N + 32*WN - 1) / (32*WN), g = get_group_id(0);
  int n0 = ((g % gn) * WN + sg % WN) * 32;
  int m0 = ((g / gn) * WM + sg / WN) * 16;
  if (n0 >= N || m0 >= T) return;
  float8 acc00 = 0, acc01 = 0, acc10 = 0, acc11 = 0;
  short8 p0, p1, p2, p3, q0, q1, q2, q3; int8 u0, u1, u2, u3, v0, v1, v2, v3;
  LOAD(0, p, u);
  int k = 0;
  for (; k + 64 <= K; k += 64) {
    LOAD(k + 32, q, v);
    MAD(p, u);
    if (k + 64 < K) LOAD(k + 64, p, u);
    MAD(q, v);
  }
  if (k < K) MAD(p, u);
  intel_sub_group_2d_block_write_32b_8r16x1c((global void*)Y, N*4, T, N*4, (int2)(n0, m0), (private uint*)&acc00);
  intel_sub_group_2d_block_write_32b_8r16x1c((global void*)Y, N*4, T, N*4, (int2)(n0 + 16, m0), (private uint*)&acc01);
  intel_sub_group_2d_block_write_32b_8r16x1c((global void*)Y, N*4, T, N*4, (int2)(n0, m0 + 8), (private uint*)&acc10);
  intel_sub_group_2d_block_write_32b_8r16x1c((global void*)Y, N*4, T, N*4, (int2)(n0 + 16, m0 + 8), (private uint*)&acc11);
}
