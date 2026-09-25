#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define C(x) do{ze_result_t r=(x); if(r){printf("%s -> %x\n",#x,r);return 1;}}while(0)
static double now(){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3;}
int main(int argc,char**argv){
  int N=atoi(argv[2]), EV=atoi(argv[3]);
  ze_init_driver_type_desc_t d={ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,0,ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n=1; ze_driver_handle_t drv; C(zeInitDrivers(&n,&drv,&d)); uint32_t nd=1; ze_device_handle_t dev; C(zeDeviceGet(drv,&nd,&dev));
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,2); long sz=ftell(f); rewind(f); unsigned char*il=malloc(sz); if(fread(il,1,sz,f)!=sz) return 1;
  ze_module_desc_t md={ZE_STRUCTURE_TYPE_MODULE_DESC,0,ZE_MODULE_FORMAT_IL_SPIRV,sz,il,"",0};
  ze_module_handle_t mod; C(zeModuleCreate(ctx,dev,&md,&mod,0));
  ze_kernel_desc_t kd={ZE_STRUCTURE_TYPE_KERNEL_DESC,0,0,"nop"}; ze_kernel_handle_t k; C(zeKernelCreate(mod,&kd,&k));
  ze_device_mem_alloc_desc_t dd={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC}; void*p; C(zeMemAllocDevice(ctx,&dd,64,64,dev,&p));
  C(zeKernelSetGroupSize(k,16,1,1)); C(zeKernelSetArgumentValue(k,0,8,&p));
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,ZE_COMMAND_QUEUE_FLAG_IN_ORDER,ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,0};
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl)); ze_group_count_t g={1,1,1};
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,0,ZE_EVENT_POOL_FLAG_HOST_VISIBLE|ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,256};
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,1,&dev,&pool));
  ze_event_handle_t ev[256]; for(int i=0;i<256;i++){ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC,0,i,0,ZE_EVENT_SCOPE_FLAG_HOST}; C(zeEventCreate(pool,&ed,&ev[i]));}
  double best=1e9;
  for(int it=0;it<50;it++){
    for(int i=0;i<N;i++) zeEventHostReset(ev[i]);
    double t0=now();
    for(int i=0;i<N;i++) C(zeCommandListAppendLaunchKernel(cl,k,&g,EV?ev[i]:0,0,0));
    C(zeCommandListHostSynchronize(cl,UINT64_MAX));
    double t=now()-t0; if(it>5 && t<best) best=t;
  }
  printf("N=%d events=%d: best %.1f us, %.2f us per launch\n",N,EV,best,best/N);
}
