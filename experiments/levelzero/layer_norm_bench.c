#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define C(x) do{ze_result_t r=(x); if(r){printf("%s -> %x\n",#x,r);exit(1);}}while(0)
/* fln spv kernel TM T K */
int main(int argc,char**argv){
  int TM=atoi(argv[3]),T=atoi(argv[4]),K=atoi(argv[5]),H=384;
  ze_init_driver_type_desc_t d={ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,0,ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n=1; ze_driver_handle_t drv; C(zeInitDrivers(&n,&drv,&d)); uint32_t nd=1; ze_device_handle_t dev; C(zeDeviceGet(drv,&nd,&dev));
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,2); long sz=ftell(f); rewind(f); unsigned char*il=malloc(sz); if(fread(il,1,sz,f)!=sz) return 1;
  const char*o=getenv("OPTS")?getenv("OPTS"):"";
  ze_module_desc_t md={ZE_STRUCTURE_TYPE_MODULE_DESC,0,ZE_MODULE_FORMAT_IL_SPIRV,sz,il,o,0};
  ze_module_handle_t mod; C(zeModuleCreate(ctx,dev,&md,&mod,0));
  ze_kernel_desc_t kd={ZE_STRUCTURE_TYPE_KERNEL_DESC,0,0,argv[2]}; ze_kernel_handle_t k; C(zeKernelCreate(mod,&kd,&k));
  ze_kernel_properties_t kp={ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES}; C(zeKernelGetProperties(k,&kp));
  ze_device_mem_alloc_desc_t dd={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *act,*wt,*bias,*x,*xh,*lw,*lb;
  C(zeMemAllocDevice(ctx,&dd,(size_t)T*K*2,64,dev,&act)); C(zeMemAllocDevice(ctx,&dd,(size_t)K*H*2,64,dev,&wt)); C(zeMemAllocDevice(ctx,&dd,H*4,64,dev,&bias));
  C(zeMemAllocDevice(ctx,&dd,(size_t)T*H*4,64,dev,&x)); C(zeMemAllocDevice(ctx,&dd,(size_t)T*H*2,64,dev,&xh)); C(zeMemAllocDevice(ctx,&dd,H*4,64,dev,&lw)); C(zeMemAllocDevice(ctx,&dd,H*4,64,dev,&lb));
  { ze_host_mem_alloc_desc_t hd={ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC}; void*h; size_t big=(size_t)T*K*2 > (size_t)K*H*2 ? (size_t)T*K*2 : (size_t)K*H*2; if((size_t)T*H*4>big) big=(size_t)T*H*4;
    C(zeMemAllocHost(ctx,&hd,big,64,&h));
    ze_command_queue_desc_t q0={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,0}; ze_command_list_handle_t c0; C(zeCommandListCreateImmediate(ctx,dev,&q0,&c0));
    unsigned short*hs=h; srand(1); for(size_t i=0;i<big/2;i++){ _Float16 v=(_Float16)((rand()%2001-1000)/1000.f); __builtin_memcpy(&hs[i],&v,2);} 
    C(zeCommandListAppendMemoryCopy(c0,act,h,(size_t)T*K*2,0,0,0)); C(zeCommandListAppendMemoryCopy(c0,wt,h,(size_t)K*H*2,0,0,0));
    float*hf=h; for(size_t i=0;i<(size_t)T*H;i++) hf[i]=(rand()%2001-1000)/1000.f; C(zeCommandListAppendMemoryCopy(c0,x,h,(size_t)T*H*4,0,0,0));
    for(int i=0;i<H;i++) hf[i]=1.0f; C(zeCommandListAppendMemoryCopy(c0,lw,h,H*4,0,0,0)); for(int i=0;i<H;i++) hf[i]=0.f; C(zeCommandListAppendMemoryCopy(c0,lb,h,H*4,0,0,0)); C(zeCommandListAppendMemoryCopy(c0,bias,h,H*4,0,0,0)); }
  float eps=1e-12f;
  int BL=getenv("BL")?atoi(getenv("BL")):1; C(zeKernelSetGroupSize(k,16*H/32*BL,1,1)); TM*=BL;
  void* p[]={&act,&wt,&bias,&x,&xh,&lw,&lb}; for(int i=0;i<7;i++) C(zeKernelSetArgumentValue(k,i,8,p[i]));
  C(zeKernelSetArgumentValue(k,7,4,&eps)); C(zeKernelSetArgumentValue(k,8,4,&T)); C(zeKernelSetArgumentValue(k,9,4,&H)); C(zeKernelSetArgumentValue(k,10,4,&K));
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,0};
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl)); ze_group_count_t g={(T+TM-1)/TM,1,1};
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,0,ZE_EVENT_POOL_FLAG_HOST_VISIBLE|ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,400};
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,1,&dev,&pool));
  ze_event_handle_t ev[400]; for(int i=0;i<400;i++){ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC,0,i,0,ZE_EVENT_SCOPE_FLAG_HOST}; C(zeEventCreate(pool,&ed,&ev[i]));}
  for(int i=0;i<400;i++) C(zeCommandListAppendLaunchKernel(cl,k,&g,ev[i],0,0));
  C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  ze_device_properties_t dp={ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; C(zeDeviceGetProperties(dev,&dp));
  static double ts_[400]; int nts=0; double best=1e9,sum=0; for(int i=200;i<400;i++){ze_kernel_timestamp_result_t ts; C(zeEventQueryKernelTimestamp(ev[i],&ts)); double t=(double)(ts.context.kernelEnd-ts.context.kernelStart)*dp.timerResolution; sum+=t; ts_[nts++]=t; if(t<best)best=t;} for(int a=0;a<nts;a++)for(int b=a+1;b<nts;b++) if(ts_[b]<ts_[a]){double q=ts_[a];ts_[a]=ts_[b];ts_[b]=q;} double med=ts_[nts/2];
  printf("%-24s %s T%d K%d spill %u: median %.1f us best %.1f us %.1f TFLOPS\n",argv[2],o[0]?"large":"", T,K,kp.spillMemSize,med/1e3,best/1e3,2.0*T*K*H/med/1e3);
}
