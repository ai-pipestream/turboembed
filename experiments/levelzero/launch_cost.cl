__kernel void nop(__global int *p) { if (get_global_id(0) == 0) p[0] += 1; }
