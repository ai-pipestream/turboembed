#define OL __attribute__((overloadable))
OL float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);
OL void intel_sub_group_2d_block_read_16b_32r16x1c(global void*, int, int, int, int2, private ushort*);
OL void intel_sub_group_2d_block_read_transform_16b_16r16x1c(global void*, int, int, int, int2, private uint*);
OL void intel_sub_group_2d_block_prefetch_16b_32r16x1c(global void*, int, int, int, int2);
OL void intel_sub_group_2d_block_prefetch_16b_32r16x2c(global void*, int, int, int, int2);
OL void intel_sub_group_2d_block_write_32b_8r16x1c(global void*, int, int, int, int2, private uint*);
/* 32 x 64 a sub-group, k 16 a stage, next stage loaded before this stage's 16 products */
#define LOAD(k, A, B) do { \
    short8 ta_[4]; int8 t0_, t1_, t2_, t3_; \
    intel_sub_group_2d_block_read_16b_32r16x1c((global void*)X, K*2, T, K*2, (int2)(k, m0), (private ushort*)ta_); \
    intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)W, N*2, K, N*2, (int2)(n0, k), (private uint*)&t0_); \
    intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + 16, k), (private uint*)&t1_); \
    intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + 32, k), (private uint*)&t2_); \
    intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)W, N*2, K, N*2, (int2)(n0 + 48, k), (private uint*)&t3_); \
    A##0 = ta_[0]; A##1 = ta_[1]; A##2 = ta_[2]; A##3 = ta_[3]; B##0 = t0_; B##1 = t1_; B##2 = t2_; B##3 = t3_; \
  } while (0)
#define D(i, j, a, b) c##i##j = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c##i##j)
#define ROW(i, A, B) D(i,0,A##i,B##0); D(i,1,A##i,B##1); D(i,2,A##i,B##2); D(i,3,A##i,B##3)
#define MAD(A, B) do { ROW(0, A, B); ROW(1, A, B); ROW(2, A, B); ROW(3, A, B); } while (0)
#define ST(i, j) intel_sub_group_2d_block_write_32b_8r16x1c((global void*)Y, N*4, T, N*4, (int2)(n0 + 16*j, m0 + 8*i), (private uint*)&c##i##j)
__attribute__((intel_reqd_sub_group_size(16)))
kernel void gemm(global const half *X, global const half *W, global float *Y, int T, int K, int N) {
  int sg = get_sub_group_id();
  int gn = (N + 64*WN - 1) / (64*WN), g = get_group_id(0);
  int n0 = ((g % gn) * WN + sg % WN) * 64;
  int m0 = ((g / gn) * WM + sg / WN) * 32;
  if (n0 >= N || m0 >= T) return;
  float8 c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0,c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
  short8 p0,p1,p2,p3,q0,q1,q2,q3; int8 u0,u1,u2,u3,v0,v1,v2,v3;
  LOAD(0, p, u);
  int k = 0;
#ifdef PFD
  for (int p = 0; p < PFD && p < K; p += 32) {
    intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)X, K*2, T, K*2, (int2)(p, m0));
    intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)W, N*2, K, N*2, (int2)(n0, p));
    intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)W, N*2, K, N*2, (int2)(n0 + 32, p));
  }
#endif
  for (; k + 32 <= K; k += 32) {
#ifdef PFD
    if (k + PFD < K) {
      intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)X, K*2, T, K*2, (int2)(k + PFD, m0));
      intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)W, N*2, K, N*2, (int2)(n0, k + PFD));
      intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)W, N*2, K, N*2, (int2)(n0 + 32, k + PFD));
    }
#endif
    LOAD(k + 16, q, v);
    MAD(p, u);
    if (k + 32 < K) LOAD(k + 32, p, u);
    MAD(q, v);
  }
  ST(0,0);ST(0,1);ST(0,2);ST(0,3);ST(1,0);ST(1,1);ST(1,2);ST(1,3);ST(2,0);ST(2,1);ST(2,2);ST(2,3);ST(3,0);ST(3,1);ST(3,2);ST(3,3);
}
