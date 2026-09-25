#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#define C(x) do{ze_result_t r=(x); if(r){printf("%s -> %x\n",#x,r);return 1;}}while(0)
int main(int argc,char**argv){
  ze_init_driver_type_desc_t d={ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,0,ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n=1; ze_driver_handle_t drv; C(zeInitDrivers(&n,&drv,&d)); uint32_t nd=1; ze_device_handle_t dev; C(zeDeviceGet(drv,&nd,&dev));
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,2); long sz=ftell(f); rewind(f); unsigned char*il=malloc(sz); if(fread(il,1,sz,f)!=sz) return 1;
  ze_module_desc_t md={ZE_STRUCTURE_TYPE_MODULE_DESC,0,ZE_MODULE_FORMAT_IL_SPIRV,sz,il,"",0};
  ze_module_handle_t mod; ze_module_build_log_handle_t log; ze_result_t r=zeModuleCreate(ctx,dev,&md,&mod,&log);
  if(r){size_t ls=0; zeModuleBuildLogGetString(log,&ls,0); char*s=malloc(ls); zeModuleBuildLogGetString(log,&ls,s); printf("build %x: %s\n",r,s); return 1;}
  ze_kernel_desc_t kd={ZE_STRUCTURE_TYPE_KERNEL_DESC,0,0,"dpastest"}; ze_kernel_handle_t k; C(zeKernelCreate(mod,&kd,&k));
  ze_device_mem_alloc_desc_t dd={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC}; ze_host_mem_alloc_desc_t hd={ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  float *A,*B,*Cm; C(zeMemAllocShared(ctx,&dd,&hd,64*4,64,dev,(void**)&A)); C(zeMemAllocShared(ctx,&dd,&hd,8*16*4,64,dev,(void**)&B)); C(zeMemAllocShared(ctx,&dd,&hd,8*16*4,64,dev,(void**)&Cm));
  for(int i=0;i<64;i++) A[i]=i+1;           /* value = raw index + 1 */
  for(int kk=0;kk<8;kk++) for(int nn=0;nn<16;nn++) B[kk*16+nn]=(nn<8 && kk==nn)?1:0;
  C(zeKernelSetGroupSize(k,16,1,1)); C(zeKernelSetArgumentValue(k,0,8,&A)); C(zeKernelSetArgumentValue(k,1,8,&B)); C(zeKernelSetArgumentValue(k,2,8,&Cm));
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,0};
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl)); ze_group_count_t g={1,1,1};
  C(zeCommandListAppendLaunchKernel(cl,k,&g,0,0,0)); C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  /* C[m][n] for n<8 = A[m][k=n]; print raw index (lane*4+i) that landed at (m,k) */
  for(int m=0;m<8;m++){ for(int nn=0;nn<8;nn++){ int v=(int)Cm[m*16+nn]-1; printf("%3d(l%2d,i%d) ",v,v/4,v%4);} printf("\n"); }
}
