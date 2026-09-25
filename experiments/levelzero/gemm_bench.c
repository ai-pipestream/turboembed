#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
static uint16_t f2h(float f){ _Float16 h=(_Float16)f; uint16_t u; __builtin_memcpy(&u,&h,2); return u;}
static float h2f(uint16_t u){ _Float16 h; __builtin_memcpy(&h,&u,2); return (float)h;}
#define C(x) do{ze_result_t r=(x); if(r){printf("%s -> %x\n",#x,r);exit(1);}}while(0)
static double now(){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9;}
/* gemm spv T K N TM TN sgs transpose */
int main(int argc,char**argv){
  int T=atoi(argv[2]),K=atoi(argv[3]),N=atoi(argv[4]),TM=atoi(argv[5]),TN=atoi(argv[6]),WM=atoi(argv[7]),WN=atoi(argv[8]),SP=atoi(argv[9]),TR=0,SGS=WM*WN;
  ze_init_driver_type_desc_t d={ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,0,ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n=1; ze_driver_handle_t drv; C(zeInitDrivers(&n,&drv,&d)); uint32_t nd=1; ze_device_handle_t dev; C(zeDeviceGet(drv,&nd,&dev));
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,2); long sz=ftell(f); rewind(f); unsigned char*il=malloc(sz); fread(il,1,sz,f);
  const char* opts=getenv("OPTS")?getenv("OPTS"):"";
  ze_module_desc_t md={ZE_STRUCTURE_TYPE_MODULE_DESC,0,ZE_MODULE_FORMAT_IL_SPIRV,sz,il,opts,0};
  ze_module_handle_t mod; ze_module_build_log_handle_t log; ze_result_t r=zeModuleCreate(ctx,dev,&md,&mod,&log);
  if(r){size_t ls=0; zeModuleBuildLogGetString(log,&ls,0); char*s=malloc(ls); zeModuleBuildLogGetString(log,&ls,s); printf("build %x: %s\n",r,s); return 1;}
  ze_kernel_desc_t kd={ZE_STRUCTURE_TYPE_KERNEL_DESC,0,0,"gemm"}; ze_kernel_handle_t k; C(zeKernelCreate(mod,&kd,&k));
  ze_kernel_properties_t kp={ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES}; C(zeKernelGetProperties(k,&kp));
  ze_device_mem_alloc_desc_t dd={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC}; ze_host_mem_alloc_desc_t hd={ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  uint16_t *X,*W; float *Y;
  C(zeMemAllocShared(ctx,&dd,&hd,(size_t)T*K*2,64,dev,(void**)&X)); C(zeMemAllocShared(ctx,&dd,&hd,(size_t)N*K*2,64,dev,(void**)&W)); C(zeMemAllocShared(ctx,&dd,&hd,(size_t)T*N*4*SP,64,dev,(void**)&Y));
  float *fx=malloc(4L*T*K), *fw=malloc(4L*N*K);
  srand(1);
  for(long i=0;i<(long)T*K;i++){fx[i]=h2f(f2h((rand()%2001-1000)/1000.f)); X[i]=f2h(fx[i]);}
  for(long i=0;i<(long)N*K;i++){fw[i]=h2f(f2h((rand()%2001-1000)/1000.f));}
  for(int nn=0;nn<N;nn++)for(int kk=0;kk<K;kk++){ uint16_t h=f2h(fw[(long)nn*K+kk]); if(TR) W[(long)nn*K+kk]=h; else W[(long)kk*N+nn]=h; }
  for(long i=0;i<(long)T*N*SP;i++) Y[i]=NAN;
  int groups=((T+TM*WM-1)/(TM*WM))*((N+TN*WN-1)/(TN*WN));
  C(zeKernelSetGroupSize(k,16*SGS,1,1));
  C(zeKernelSetArgumentValue(k,0,8,&X)); C(zeKernelSetArgumentValue(k,1,8,&W)); C(zeKernelSetArgumentValue(k,2,8,&Y));
  C(zeKernelSetArgumentValue(k,3,4,&T)); C(zeKernelSetArgumentValue(k,4,4,&K)); C(zeKernelSetArgumentValue(k,5,4,&N));
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,0};
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl)); ze_group_count_t g={groups,SP,1};
  C(zeCommandListAppendLaunchKernel(cl,k,&g,0,0,0)); C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  int bad=0; double maxe=0;
  for(int i=0;i<T;i+= (T>64? 7:1))for(int j=0;j<N;j++){double s=0; for(int kk=0;kk<K;kk++) s+=(double)fx[(long)i*K+kk]*fw[(long)j*K+kk]; double y=0; for(int p=0;p<SP;p++) y+=Y[(long)p*T*N+(long)i*N+j]; double e=fabs(s-y); if(!(e<1e-2)){ if(bad<5) printf("Y[%d][%d]=%f want %f\n",i,j,y,s); bad++;} if(e>maxe)maxe=e;}
  /* timing with kernel timestamps */
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,0,ZE_EVENT_POOL_FLAG_HOST_VISIBLE|ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,400};
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,1,&dev,&pool));
  ze_event_handle_t ev[400]; for(int i=0;i<400;i++){ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC,0,i,0,ZE_EVENT_SCOPE_FLAG_HOST}; C(zeEventCreate(pool,&ed,&ev[i]));}
  for(int i=0;i<400;i++) C(zeCommandListAppendLaunchKernel(cl,k,&g,ev[i],0,0));
  C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  ze_device_properties_t dp={ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; C(zeDeviceGetProperties(dev,&dp));
  static double ts_[400]; int nts=0; double best=1e9,sum=0; for(int i=200;i<400;i++){ze_kernel_timestamp_result_t ts; C(zeEventQueryKernelTimestamp(ev[i],&ts)); double t=(double)(ts.context.kernelEnd-ts.context.kernelStart)*dp.timerResolution; sum+=t; ts_[nts++]=t; if(t<best)best=t;} for(int a=0;a<nts;a++)for(int b=a+1;b<nts;b++) if(ts_[b]<ts_[a]){double q=ts_[a];ts_[a]=ts_[b];ts_[b]=q;} double med=ts_[nts/2];
  double avg=med;
  printf("T%d K%d N%d TM%d TN%d W%dx%d SP%d spill%u: bad %d maxe %.2g avg %.1f us best %.1f us -> %.1f TFLOPS\n",T,K,N,TM,TN,WM,WN,SP,kp.spillMemSize,bad,maxe,avg/1e3,best/1e3,2.0*T*K*N/avg/1e3);
}
