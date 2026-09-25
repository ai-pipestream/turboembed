__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpastest(__global const half *A, __global const half *B, __global float *C) {
  int lane = get_sub_group_local_id();
  short8 a; int8 b;
  for (int m=0;m<8;m++) a[m] = as_short(A[m*16+lane]);
  for (int j=0;j<8;j++) { ushort lo = as_ushort(B[(2*j)*16+lane]), hi = as_ushort(B[(2*j+1)*16+lane]); b[j] = (int)((uint)lo | ((uint)hi<<16)); }
  float8 acc = 0, c;
  __asm__ volatile("{\n"
    ".decl AA v_type=G type=ud num_elts=64 align=GRF alias=<%3, 0>\n"
    ".decl BB v_type=G type=ud num_elts=128 align=GRF alias=<%2, 0>\n"
    "dpas.hf.hf.8.8 (M1, 16) %0.0 %1.0 BB.0 AA(0,0)\n"
    "}\n" : "=rw"(c) : "rw"(acc), "rw"(b), "rw"(a));
  for (int m=0;m<8;m++) C[m*16+lane] = c[m];
}
