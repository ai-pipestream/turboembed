#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#define C(x) do{ze_result_t r=(x); if(r){printf("%s -> %x\n",#x,r);exit(1);}}while(0)
/* lin spv kernel T K N flags WM WN TM */
int main(int argc,char**argv){
  int T=atoi(argv[3]),K=atoi(argv[4]),N=atoi(argv[5]),flags=atoi(argv[6]),WM=atoi(argv[7]),WN=atoi(argv[8]),TM=atoi(argv[9]);
  ze_init_driver_type_desc_t d={ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,0,ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n=1; ze_driver_handle_t drv; C(zeInitDrivers(&n,&drv,&d)); uint32_t nd=1; ze_device_handle_t dev; C(zeDeviceGet(drv,&nd,&dev));
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,2); long sz=ftell(f); rewind(f); unsigned char*il=malloc(sz); if(fread(il,1,sz,f)!=sz) return 1;
  const char*o=getenv("OPTS")?getenv("OPTS"):"";
  ze_module_desc_t md={ZE_STRUCTURE_TYPE_MODULE_DESC,0,ZE_MODULE_FORMAT_IL_SPIRV,sz,il,o,0};
  ze_module_handle_t mod; C(zeModuleCreate(ctx,dev,&md,&mod,0));
  ze_kernel_desc_t kd={ZE_STRUCTURE_TYPE_KERNEL_DESC,0,0,argv[2]}; ze_kernel_handle_t k; C(zeKernelCreate(mod,&kd,&k));
  ze_device_mem_alloc_desc_t dd={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *x,*w,*b,*y; C(zeMemAllocDevice(ctx,&dd,(size_t)T*K*2,64,dev,&x)); C(zeMemAllocDevice(ctx,&dd,(size_t)K*N*2,64,dev,&w)); C(zeMemAllocDevice(ctx,&dd,N*4,64,dev,&b)); C(zeMemAllocDevice(ctx,&dd,(size_t)T*N*4,64,dev,&y));
  ze_host_mem_alloc_desc_t hd={ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC}; size_t big=(size_t)T*K*2>(size_t)K*N*2?(size_t)T*K*2:(size_t)K*N*2; void*h; C(zeMemAllocHost(ctx,&hd,big,64,&h));
  unsigned short*hs=h; srand(1); for(size_t i=0;i<big/2;i++){ _Float16 v=(_Float16)((rand()%2001-1000)/1000.f); __builtin_memcpy(&hs[i],&v,2);}
  ze_command_queue_desc_t q0={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,0}; ze_command_list_handle_t c0; C(zeCommandListCreateImmediate(ctx,dev,&q0,&c0));
  C(zeCommandListAppendMemoryCopy(c0,x,h,(size_t)T*K*2,0,0,0)); C(zeCommandListAppendMemoryCopy(c0,w,h,(size_t)K*N*2,0,0,0)); C(zeCommandListAppendMemoryCopy(c0,b,h,N*4,0,0,0));
  C(zeKernelSetGroupSize(k,16*WM*WN,1,1));
  void* p[]={&x,&w,&b,&y}; for(int i=0;i<4;i++) C(zeKernelSetArgumentValue(k,i,8,p[i]));
  C(zeKernelSetArgumentValue(k,4,4,&T)); C(zeKernelSetArgumentValue(k,5,4,&N)); C(zeKernelSetArgumentValue(k,6,4,&K)); C(zeKernelSetArgumentValue(k,7,4,&flags));
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,0};
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl)); ze_group_count_t g={(N+32*WN-1)/(32*WN),(T+TM*WM-1)/(TM*WM),1};
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,0,ZE_EVENT_POOL_FLAG_HOST_VISIBLE|ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,400};
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,1,&dev,&pool));
  ze_event_handle_t ev[400]; for(int i=0;i<400;i++){ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC,0,i,0,ZE_EVENT_SCOPE_FLAG_HOST}; C(zeEventCreate(pool,&ed,&ev[i]));}
  for(int i=0;i<400;i++) C(zeCommandListAppendLaunchKernel(cl,k,&g,ev[i],0,0));
  C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  ze_device_properties_t dp={ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; C(zeDeviceGetProperties(dev,&dp));
  static double ts[200]; for(int i=200;i<400;i++){ze_kernel_timestamp_result_t t; C(zeEventQueryKernelTimestamp(ev[i],&t)); ts[i-200]=(double)(t.context.kernelEnd-t.context.kernelStart)*dp.timerResolution;}
  for(int a=0;a<200;a++)for(int bb=a+1;bb<200;bb++) if(ts[bb]<ts[a]){double q=ts[a];ts[a]=ts[bb];ts[bb]=q;}
  printf("%-26s T%d K%d N%d flags%d: median %.1f us  %.1f TFLOPS\n",argv[2],T,K,N,flags,ts[100]/1e3,2.0*T*K*N/ts[100]/1e3);
}
