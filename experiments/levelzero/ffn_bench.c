#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define C(x) do{ze_result_t r=(x); if(r){printf("%s -> %x (line %d)\n",#x,r,__LINE__);exit(1);}}while(0)
static ze_context_handle_t ctx; static ze_device_handle_t dev; static ze_command_list_handle_t cl, c0; static double tres;
static void* dalloc(size_t n){ ze_device_mem_alloc_desc_t dd={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC}; void*p; C(zeMemAllocDevice(ctx,&dd,n,64,dev,&p)); return p; }
static void up(void*d,const void*h,size_t n){ C(zeCommandListAppendMemoryCopy(c0,d,h,n,0,0,0)); }
static void down(void*h,const void*d,size_t n){ C(zeCommandListAppendMemoryCopy(c0,h,d,n,0,0,0)); }
static unsigned short f2h(float f){ _Float16 h=(_Float16)f; unsigned short u; memcpy(&u,&h,2); return u; }
static float h2f(unsigned short u){ _Float16 h; memcpy(&h,&u,2); return (float)h; }
static void setargs(ze_kernel_handle_t k, int n, void**v, size_t*s){ for(int i=0;i<n;i++) C(zeKernelSetArgumentValue(k,i,s[i],v[i])); }
static double timeit(ze_kernel_handle_t *ks, ze_group_count_t *gs, int nk, int iters){
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,0,ZE_EVENT_POOL_FLAG_HOST_VISIBLE|ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,(uint32_t)(iters*nk)};
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,1,&dev,&pool));
  ze_event_handle_t *ev=malloc(sizeof(*ev)*iters*nk); for(int i=0;i<iters*nk;i++){ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC,0,(uint32_t)i,0,ZE_EVENT_SCOPE_FLAG_HOST}; C(zeEventCreate(pool,&ed,&ev[i]));}
  for(int it=0;it<iters;it++) for(int j=0;j<nk;j++) C(zeCommandListAppendLaunchKernel(cl,ks[j],&gs[j],ev[it*nk+j],0,0));
  C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  double *t=malloc(sizeof(double)*iters); for(int it=0;it<iters;it++){ t[it]=0; for(int j=0;j<nk;j++){ze_kernel_timestamp_result_t ts; C(zeEventQueryKernelTimestamp(ev[it*nk+j],&ts)); t[it]+=(double)(ts.context.kernelEnd-ts.context.kernelStart)*tres;} }
  for(int a=iters/2;a<iters;a++)for(int b=a+1;b<iters;b++) if(t[b]<t[a]){double q=t[a];t[a]=t[b];t[b]=q;}
  return t[iters/2+iters/4]/1e3;
}
int main(int argc,char**argv){
  int T=atoi(argv[2]),H=384,I=1536; int BL=4;
  ze_init_driver_type_desc_t d={ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,0,ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t n=1; ze_driver_handle_t drv; C(zeInitDrivers(&n,&drv,&d)); uint32_t nd=1; C(zeDeviceGet(drv,&nd,&dev));
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC}; C(zeContextCreate(drv,&cd,&ctx));
  ze_device_properties_t dp={ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; C(zeDeviceGetProperties(dev,&dp)); tres=dp.timerResolution;
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,2); long sz=ftell(f); rewind(f); unsigned char*il=malloc(sz); if(fread(il,1,sz,f)!=(size_t)sz) return 1;
  ze_module_desc_t md={ZE_STRUCTURE_TYPE_MODULE_DESC,0,ZE_MODULE_FORMAT_IL_SPIRV,sz,il,"",0};
  ze_module_handle_t mod; ze_module_build_log_handle_t lg; ze_result_t r=zeModuleCreate(ctx,dev,&md,&mod,&lg); if(r){size_t ls=0; zeModuleBuildLogGetString(lg,&ls,0); char*s=malloc(ls); zeModuleBuildLogGetString(lg,&ls,s); printf("%s\n",s); return 1;}
  ze_kernel_handle_t kin,kln,kmlp; ze_kernel_desc_t kd={ZE_STRUCTURE_TYPE_KERNEL_DESC,0,0,"linear_dpas_to_half"}; C(zeKernelCreate(mod,&kd,&kin));
  kd.pKernelName="linear_dpas_layer_norm"; C(zeKernelCreate(mod,&kd,&kln)); kd.pKernelName="linear_dpas_mlp"; C(zeKernelCreate(mod,&kd,&kmlp));
  ze_command_queue_desc_t qs={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,0}; C(zeCommandListCreateImmediate(ctx,dev,&qs,&c0));
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,0,0,0,0,ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,0}; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl));
  srand(1);
  unsigned short *hx=malloc(T*H*2),*hw1=malloc(H*I*2),*hw2=malloc(I*H*2); float *hb1=malloc(I*4),*hb2=malloc(H*4),*hlw=malloc(H*4),*hlb=malloc(H*4);
  for(int i=0;i<T*H;i++) hx[i]=f2h((rand()%2001-1000)/1000.f);
  for(int i=0;i<H*I;i++) hw1[i]=f2h((rand()%2001-1000)/20000.f*1.5f);
  for(int i=0;i<I*H;i++) hw2[i]=f2h((rand()%2001-1000)/20000.f);
  for(int i=0;i<I;i++) hb1[i]=(rand()%2001-1000)/5000.f; for(int i=0;i<H;i++){hb2[i]=(rand()%2001-1000)/5000.f; hlw[i]=1+(rand()%200-100)/1000.f; hlb[i]=(rand()%200-100)/1000.f;}
  void *xa=dalloc(T*H*2),*xb=dalloc(T*H*2),*w1=dalloc(H*I*2),*w2=dalloc(I*H*2),*b1=dalloc(I*4),*b2=dalloc(H*4),*lw=dalloc(H*4),*lb=dalloc(H*4),*mid=dalloc((size_t)T*I*2);
  up(w1,hw1,H*I*2); up(w2,hw2,I*H*2); up(b1,hb1,I*4); up(b2,hb2,H*4); up(lw,hlw,H*4); up(lb,hlb,H*4);
  float eps=1e-12f; void*zero=0; int flags=3;
  /* unfused */
  C(zeKernelSetGroupSize(kin,256,1,1)); void*ai[]={&xa,&w1,&b1,&mid,&T,&I,&H,&flags}; size_t si[]={8,8,8,8,4,4,4,4}; setargs(kin,8,ai,si);
  C(zeKernelSetGroupSize(kln,16*(H/32)*BL,1,1)); void*al[]={&mid,&w2,&b2,&zero,&xa,&lw,&lb,&eps,&T,&H,&I}; size_t sl[]={8,8,8,8,8,8,8,4,4,4,4}; setargs(kln,11,al,sl);
  C(zeKernelSetGroupSize(kmlp,16*(H/32)*BL,1,1)); void*am[]={&w1,&b1,&w2,&b2,&zero,&xb,&lw,&lb,&eps,&T,&H,&I}; size_t sm[]={8,8,8,8,8,8,8,8,4,4,4,4}; setargs(kmlp,12,am,sm);
  ze_group_count_t gin={(I+63)/64,(T+127)/128,1}, gln={(T+16*BL-1)/(16*BL),1,1};
  up(xa,hx,T*H*2); up(xb,hx,T*H*2);
  C(zeCommandListAppendLaunchKernel(cl,kin,&gin,0,0,0)); C(zeCommandListAppendLaunchKernel(cl,kln,&gln,0,0,0)); C(zeCommandListAppendLaunchKernel(cl,kmlp,&gln,0,0,0)); C(zeCommandListHostSynchronize(cl,UINT64_MAX));
  unsigned short *ra=malloc(T*H*2),*rb=malloc(T*H*2); down(ra,xa,T*H*2); down(rb,xb,T*H*2);
  double md2=0,ma=0; for(int i=0;i<T*H;i++){ double dd=fabs(h2f(ra[i])-h2f(rb[i])); if(dd>md2) md2=dd; if(fabs(h2f(ra[i]))>ma) ma=fabs(h2f(ra[i])); }
  printf("T%d: max |unfused - fused| %.3g (max |x| %.3g)\n",T,md2,ma);
  ze_kernel_handle_t k2[2]={kin,kln}; ze_group_count_t g2[2]={gin,gln};
  printf("unfused (in + out/LN): %.1f us   fused MLP: %.1f us\n", timeit(k2,g2,2,200), timeit(&kmlp,&gln,1,200));
}
