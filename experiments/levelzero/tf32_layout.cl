__attribute__((overloadable)) float8 intel_sub_group_tf32_tf32_matrix_mad_k8(float4 a, float8 b, float8 acc);
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpastest(__global const float *A, __global const float *B, __global float *C) {
  int lane = get_sub_group_local_id();
  float4 a = vload4(lane, A);           /* raw: lane*4+i */
  float8 b; for (int j=0;j<8;j++) b[j] = B[j*16+lane];   /* guess: lane = n, b[k] */
  float8 c = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, (float8)0);
  for (int m=0;m<8;m++) C[m*16+lane] = c[m];
}
