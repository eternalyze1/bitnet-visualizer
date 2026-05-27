var zt=Object.defineProperty;var Rt=(q,O,L)=>O in q?zt(q,O,{enumerable:!0,configurable:!0,writable:!0,value:L}):q[O]=L;var p=(q,O,L)=>Rt(q,typeof O!="symbol"?O+"":O,L);var C=(q,O,L)=>new Promise((he,de)=>{var pe=H=>{try{te(L.next(H))}catch(ne){de(ne)}},me=H=>{try{te(L.throw(H))}catch(ne){de(ne)}},te=H=>H.done?he(H.value):Promise.resolve(H.value).then(pe,me);te((L=L.apply(q,O)).next())});(function(){"use strict";var G;var q=1179993927,O=0,L=1,he=16,de=17,pe=18,me=27,te=28,H=30,ne=34,ze=35,ge=36;function Re(r){switch(r){case O:return 4;case L:return 2;case he:return 1;case de:return 2;case pe:return 4;case me:return 8;case te:return 8;case H:return 2;case ne:return 54/256;case ze:return 66/256;case ge:return .25;default:throw new Error(`Unsupported GGML type: ${r}`)}}var Le=class{constructor(r){p(this,"view");p(this,"offset");p(this,"textDecoder",new TextDecoder("utf-8"));this.view=new DataView(r),this.offset=0}parse(){const r=this.readHeader(),e=this.readMetadata(Number(r.metadataKVCount)),t=this.readTensorInfos(Number(r.tensorCount)),n=e["general.alignment"]||32,s=Math.ceil(this.offset/n)*n;return{header:r,metadata:e,tensors:t,tensorDataOffset:s}}readHeader(){const r=this.readU32();if(r!==q)throw new Error(`Invalid GGUF magic: 0x${r.toString(16)} (expected 0x${q.toString(16)})`);const e=this.readU32();if(e<2||e>3)throw new Error(`Unsupported GGUF version: ${e}`);const t=this.readU64(),n=this.readU64();return{magic:r,version:e,tensorCount:t,metadataKVCount:n}}readMetadata(r){const e={};for(let t=0;t<r;t++){const n=this.readString(),s=this.readMetadataValue();e[n]=s}return e}readMetadataValue(){const r=this.readU32();return this.readValueOfType(r)}readValueOfType(r){switch(r){case 0:return this.readU8();case 1:return this.readI8();case 2:return this.readU16();case 3:return this.readI16();case 4:return this.readU32();case 5:return this.readI32();case 6:return this.readF32();case 7:return this.readU8()!==0;case 8:return this.readString();case 10:return this.readU64();case 11:return this.readI64();case 12:return this.readF64();case 9:{const e=this.readU32(),t=Number(this.readU64()),n=[];for(let s=0;s<t;s++)n.push(this.readValueOfType(e));return n}default:throw new Error(`Unknown GGUF metadata type: ${r}`)}}readTensorInfos(r){const e=[];for(let t=0;t<r;t++){const n=this.readString(),s=this.readU32(),i=[];for(let u=0;u<s;u++)i.push(this.readU64());const o=this.readU32(),a=this.readU64();e.push({name:n,nDimensions:s,shape:i,type:o,offset:a})}return e}readU8(){const r=this.view.getUint8(this.offset);return this.offset+=1,r}readI8(){const r=this.view.getInt8(this.offset);return this.offset+=1,r}readU16(){const r=this.view.getUint16(this.offset,!0);return this.offset+=2,r}readI16(){const r=this.view.getInt16(this.offset,!0);return this.offset+=2,r}readU32(){const r=this.view.getUint32(this.offset,!0);return this.offset+=4,r}readI32(){const r=this.view.getInt32(this.offset,!0);return this.offset+=4,r}readU64(){const r=this.view.getBigUint64(this.offset,!0);return this.offset+=8,r}readI64(){const r=this.view.getBigInt64(this.offset,!0);return this.offset+=8,r}readF32(){const r=this.view.getFloat32(this.offset,!0);return this.offset+=4,r}readF64(){const r=this.view.getFloat64(this.offset,!0);return this.offset+=8,r}readString(){const r=Number(this.readU64()),e=new Uint8Array(this.view.buffer,this.offset,r);return this.offset+=r,this.textDecoder.decode(e)}},Ue=8;function Oe(r){const e=new DataView(r),t=Number(e.getBigUint64(0,!0)),n=new Uint8Array(r,Ue,t),s=new TextDecoder().decode(n),i=JSON.parse(s);delete i.__metadata__;const o=Ue+t;return{header:i,dataOffset:o}}function Ie(r){switch(r){case"F32":return"f32";case"F16":return"f16";case"I8":return"i8";case"I32":return"i32";case"U8":return"u8";default:throw new Error(`Unsupported safetensors dtype: ${r}`)}}function qe(r,e){const t=[];for(const[n,s]of Object.entries(r)){const[i,o]=s.data_offsets;t.push({name:n,dtype:Ie(s.dtype),shape:s.shape,offset:e+i,size:o-i})}return t}function j(r){return r.hiddenSize/r.numAttentionHeads}var He=class{constructor(r){p(this,"cache",new Map);p(this,"device");this.device=r}getOrCreate(r,e,t="main",n){const s=n?`${r}:${JSON.stringify(n)}`:r,i=this.cache.get(s);if(i)return i;const o=this.device.createShaderModule({code:e}),a=this.device.createComputePipeline({layout:"auto",compute:{module:o,entryPoint:t,constants:n}}),u=a.getBindGroupLayout(0),d={pipeline:a,bindGroupLayout:u};return this.cache.set(s,d),d}clear(){this.cache.clear()}};function v(r,e){const t=Math.max(Math.ceil(e.byteLength/4)*4,4),n=r.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(n.getMappedRange()).set(new Uint8Array(e)),n.unmap(),n}function Ve(r){return r<=1?1:1<<32-Math.clz32(r-1)}function Ne(r,e){return`${r}:${e}`}var $e=class{constructor(r,e=256){p(this,"device");p(this,"alignment");p(this,"free",new Map);p(this,"bufferToEntry",new Map);this.device=r,this.alignment=e}alignSize(r){return Math.ceil(r/this.alignment)*this.alignment}acquire(r,e){const t=this.alignSize(r),n=Ve(t),s=Ne(e,n),i=this.free.get(s);if(i&&i.length>0){const u=i.pop();return u.inUse=!0,u.buffer}const o=this.device.createBuffer({size:n,usage:e}),a={buffer:o,size:n,inUse:!0,key:s};return this.bufferToEntry.set(o,a),o}release(r){const e=this.bufferToEntry.get(r);if(!e||!e.inUse)return;e.inUse=!1;let t=this.free.get(e.key);t||(t=[],this.free.set(e.key,t)),t.push(e)}stats(){let r=0,e=0,t=0;for(const n of this.bufferToEntry.values())r++,t+=n.size,n.inUse&&e++;return{totalBuffers:r,inUse:e,totalBytes:t}}trim(){for(const[r,e]of this.free){for(const t of e)t.buffer.destroy(),this.bufferToEntry.delete(t.buffer);e.length=0}this.free.clear()}destroy(){for(const r of this.bufferToEntry.values())r.buffer.destroy();this.bufferToEntry.clear(),this.free.clear()}};function ie(){return new Map}function Y(r){r.clear()}function z(r,e,t,n,s){const i=s.map(u=>u.resource.buffer),o=r.get(t);if(o&&o.bufs.length===i.length){let u=!0;for(let d=0;d<i.length;d++)if(o.bufs[d]!==i[d]){u=!1;break}if(u)return o.bg}const a=e.createBindGroup({layout:n,entries:s});return r.set(t,{bg:a,bufs:i}),a}var _e=`// RMSNorm: x_i * w_i / sqrt(mean(x²) + eps)
//
// Two-pass within one dispatch:
//   1. Compute sum of squares (workgroup reduction)
//   2. Normalize: x_i * w_i * rsqrt(mean_sq + eps)
//
// Layout:
//   input:  [N, D] f32
//   weight: [D]    f32 (learnable scale)
//   output: [N, D] f32

struct Params {
  N: u32,
  D: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.D;

  // Pass 1: Sum of squares
  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let val = input[row_offset + col];
    local_sum += val * val;
  }

  shared_sum[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  let rms = inverseSqrt(shared_sum[0] / f32(params.D) + params.eps);

  // Pass 2: Normalize
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = input[row_offset + col] * rms * weight[col];
  }
}
`,ye=`// Elementwise operations: add, multiply
//
// Used for residual connections and gating.
//
// Layout:
//   a:      [N] f32
//   b:      [N] f32
//   output: [N] f32

struct Params {
  N: u32,
  op: u32,  // 0 = add, 1 = multiply
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  if (params.op == 0u) {
    output[idx] = a[idx] + b[idx];
  } else {
    output[idx] = a[idx] * b[idx];
  }
}
`,Fe=class{constructor(r,e,t,n,s,i,o,a){p(this,"device");p(this,"pipelines");p(this,"pool");p(this,"config");p(this,"inputLayerNorm");p(this,"postAttnLayerNorm");p(this,"attention");p(this,"ffn");p(this,"decodeNormUniform");p(this,"decodeAddUniform");p(this,"prefillNormUniform");p(this,"prefillAddUniform");p(this,"bgCache",ie());this.device=r,this.pipelines=e,this.pool=t,this.config=n,this.inputLayerNorm=s,this.postAttnLayerNorm=i,this.attention=o,this.ffn=a}initDecodeUniforms(r){{const t=new ArrayBuffer(12),n=new DataView(t);n.setUint32(0,1,!0),n.setUint32(4,this.config.hiddenSize,!0),n.setFloat32(8,this.config.rmsNormEps,!0),this.decodeNormUniform=v(this.device,t)}{const t=new ArrayBuffer(8),n=new DataView(t);n.setUint32(0,this.config.hiddenSize,!0),n.setUint32(4,0,!0),this.decodeAddUniform=v(this.device,t)}const e=t=>this.device.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.prefillNormUniform=e(12),this.prefillAddUniform=e(8),this.attention.initDecodeUniforms(r),this.ffn.initDecodeUniforms()}forward(r,e,t,n){const s=this.config.hiddenSize,i=this.dispatchRMSNorm(n,r,this.inputLayerNorm,e,"attnNorm"),o=this.attention.forward(i,e,t,n);this.pool.release(i);const a=this.dispatchAdd(n,r,o,e*s,e,"attnAdd");this.pool.release(o);const u=this.dispatchRMSNorm(n,a,this.postAttnLayerNorm,e,"ffnNorm"),d=this.ffn.forward(u,e,n);this.pool.release(u);const l=this.dispatchAdd(n,a,d,e*s,e,"ffnAdd");return this.pool.release(a),this.pool.release(d),l}dispatchRMSNorm(r,e,t,n,s){const{pipeline:i,bindGroupLayout:o}=this.pipelines.getOrCreate("rmsnorm",_e),a=this.config.hiddenSize,u=this.pool.acquire(n*a*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let d;if(n===1&&this.decodeNormUniform)d=this.decodeNormUniform;else if(this.prefillNormUniform){const f=new ArrayBuffer(12),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,a,!0),m.setFloat32(8,this.config.rmsNormEps,!0),this.device.queue.writeBuffer(this.prefillNormUniform,0,new Uint8Array(f)),d=this.prefillNormUniform}else{const f=new ArrayBuffer(12),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,a,!0),m.setFloat32(8,this.config.rmsNormEps,!0),d=v(this.device,f)}const l=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:d}}],h=n===1&&s?z(this.bgCache,this.device,s,o,l):this.device.createBindGroup({layout:o,entries:l}),c=r.beginComputePass();return c.setPipeline(i),c.setBindGroup(0,h),c.dispatchWorkgroups(n),c.end(),u}dispatchAdd(r,e,t,n,s,i){const{pipeline:o,bindGroupLayout:a}=this.pipelines.getOrCreate("elementwise_0",ye),u=this.pool.acquire(n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let d;if(s===1&&this.decodeAddUniform)d=this.decodeAddUniform;else if(this.prefillAddUniform){const f=new ArrayBuffer(8),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,0,!0),this.device.queue.writeBuffer(this.prefillAddUniform,0,new Uint8Array(f)),d=this.prefillAddUniform}else{const f=new ArrayBuffer(8),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,0,!0),d=v(this.device,f)}const l=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:d}}],h=s===1&&i?z(this.bgCache,this.device,i,a,l):this.device.createBindGroup({layout:a,entries:l}),c=r.beginComputePass();return c.setPipeline(o),c.setBindGroup(0,h),c.dispatchWorkgroups(Math.ceil(n/256)),c.end(),u}clearBGCache(){Y(this.bgCache),this.attention.clearBGCache(),this.ffn.clearBGCache()}destroyPreAllocated(){this.attention.destroyPreAllocated()}},We=`// Per-token absmax activation quantization: f32 → int8
//
// Two-pass approach:
//   Pass 1: Compute absmax per row (token)
//   Pass 2: Scale and round to [-127, 127]
//
// This shader combines both passes using workgroup reduction.
//
// Layout:
//   input:  [N, D] f32
//   output: [N, D] i32 (int8 stored as i32 for compute compatibility)
//   scales: [N]    f32 (per-token absmax / 127)

struct Params {
  N: u32,  // number of tokens
  D: u32,  // hidden dimension
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.D;

  // Pass 1: Find absmax
  var local_max: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_max = max(local_max, abs(input[row_offset + col]));
  }

  shared_max[tid] = local_max;
  workgroupBarrier();

  // Reduction for max
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
    }
    workgroupBarrier();
  }

  let absmax = shared_max[0];
  let scale = select(absmax / 127.0, 1.0, absmax == 0.0);

  if (tid == 0u) {
    scales[row] = scale;
  }

  workgroupBarrier();

  // Pass 2: Quantize
  let inv_scale = select(127.0 / absmax, 0.0, absmax == 0.0);
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let val = input[row_offset + col];
    let quantized = clamp(i32(round(val * inv_scale)), -127, 127);
    output[row_offset + col] = quantized;
  }
}
`,Ke=`// Ternary GEMV: packed ternary weights × int8 activations → i32 accumulator
//
// Weight packing (I2_S / Eddie-Wang1120 llama.cpp fork):
//   128-element block interleaving for SIMD. Each 32-byte block stores 128 elements
//   in 4 groups of 32. Byte[gp] within a block stores:
//     bits[7:6] = element at group0 (offset 0*32 + gp)
//     bits[5:4] = element at group1 (offset 1*32 + gp)
//     bits[3:2] = element at group2 (offset 2*32 + gp)
//     bits[1:0] = element at group3 (offset 3*32 + gp)
//   code mapping: {0=-1, 1=0, 2=+1}
//
// Layout:
//   weights: [M, K/16] u32  (packed ternary)
//   input:   [K]       i32  (int8 stored as i32)
//   scales:  [M]       f32  (per-row weight scale)
//   input_scale: f32         (activation absmax scale)
//   output:  [M]       f32
//
// Each workgroup processes one output row.
// Threads cooperatively reduce over the K dimension.

struct Params {
  M: u32,       // output rows
  K: u32,       // input dimension (unpacked)
  K_packed: u32, // K / 16
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<i32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> input_scale: f32;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<i32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.M) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.K_packed;

  var acc: i32 = 0;

  // Each thread processes a strided slice of packed u32 columns
  for (var col = tid; col < params.K_packed; col += WORKGROUP_SIZE) {
    let packed = weights[row_offset + col];

    // I2_S block interleaving: 128 elements per 32-byte (8 u32) block
    let block = col / 8u;
    let base_gp = (col % 8u) * 4u;

    // Process byte-by-byte: 4 bytes per u32, each byte encodes 4 groups
    for (var bi = 0u; bi < 4u; bi++) {
      let byte_val = (packed >> (bi * 8u)) & 0xFFu;
      let gp = base_gp + bi;
      let base = block * 128u + gp;

      let w0 = i32((byte_val >> 6u) & 3u) - 1;
      let w1 = i32((byte_val >> 4u) & 3u) - 1;
      let w2 = i32((byte_val >> 2u) & 3u) - 1;
      let w3 = i32(byte_val & 3u) - 1;

      acc += w0 * input[base]
           + w1 * input[base + 32u]
           + w2 * input[base + 64u]
           + w3 * input[base + 96u];
    }
  }

  // Workgroup reduction
  shared_sums[tid] = acc;
  workgroupBarrier();

  // Tree reduction
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the dequantized result
  if (tid == 0u) {
    let sum = f32(shared_sums[0]);
    output[row] = sum * scales[row] * input_scale;
  }
}
`,je=`// Ternary GEMM: batched matrix multiply for prompt processing
// Output[N,M] = Input[N,K] × TernaryWeights[M,K]^T
//
// Weight packing (I2_S / Eddie-Wang1120 llama.cpp fork):
//   128-element block interleaving. Each 32-byte block stores 128 elements
//   in 4 groups of 32. Byte[gp] within a block stores:
//     bits[7:6] = group0 (offset 0*32+gp), bits[5:4] = group1 (offset 1*32+gp)
//     bits[3:2] = group2 (offset 2*32+gp), bits[1:0] = group3 (offset 3*32+gp)
//   code mapping: {0=-1, 1=0, 2=+1}
// Input: int8 activations stored as i32
// Output: f32 (dequantized)
//
// 2D tiling: 16×16 workgroup, 4×4 per-thread output tile

struct Params {
  M: u32,        // output rows (weight rows)
  N: u32,        // output cols (batch / seq_len)
  K: u32,        // inner dimension (unpacked)
  K_packed: u32,  // K / 16
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<i32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> input_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_M: u32 = 64u;  // rows per workgroup
const TILE_N: u32 = 64u;  // cols per workgroup
const TILE_K: u32 = 32u;  // K-tile for shared memory (unpacked units)
const THREADS_M: u32 = 16u;
const THREADS_N: u32 = 16u;
const THREAD_TILE_M: u32 = 4u; // TILE_M / THREADS_M
const THREAD_TILE_N: u32 = 4u; // TILE_N / THREADS_N

var<workgroup> shared_w: array<i32, 2048>; // TILE_M × TILE_K
var<workgroup> shared_x: array<i32, 2048>; // TILE_K × TILE_N

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_row = wg_id.x * TILE_M;
  let wg_col = wg_id.y * TILE_N;
  let tid_m = local_id.x;
  let tid_n = local_id.y;

  // Per-thread accumulators (4×4 tile)
  var acc: array<i32, 16>; // THREAD_TILE_M × THREAD_TILE_N
  for (var i = 0u; i < 16u; i++) {
    acc[i] = 0;
  }

  // Loop over K in tiles
  let k_tiles = (params.K + TILE_K - 1u) / TILE_K;
  for (var kt = 0u; kt < k_tiles; kt++) {
    let k_base = kt * TILE_K;

    // Cooperatively load weight tile into shared memory (batch unpack: 1 u32 → 4 elements)
    // TILE_K=32 matches the I2_S group size, so within a K-tile all elements share
    // the same block and group — only gp varies 0..31. Each u32 holds 4 bytes (4 gp values).
    let linear_id = tid_m * THREADS_N + tid_n;
    let block = k_base / 128u;
    let group = (k_base % 128u) / 32u;
    let group_shift = 6u - 2u * group;

    // 64 rows × 8 u32/row = 512 u32s total, 256 threads → 2 loads each
    for (var ld = 0u; ld < 2u; ld++) {
      let flat_idx = linear_id + ld * 256u;
      let local_row = flat_idx / 8u;
      let u32_in_row = flat_idx % 8u;
      let base_gp = u32_in_row * 4u;
      let global_row = wg_row + local_row;

      var w0: i32 = 0; var w1: i32 = 0; var w2: i32 = 0; var w3: i32 = 0;
      if (global_row < params.M && k_base < params.K) {
        let packed = weights[global_row * params.K_packed + block * 8u + u32_in_row];
        w0 = i32((packed >> group_shift) & 3u) - 1;
        w1 = i32((packed >> (8u + group_shift)) & 3u) - 1;
        w2 = i32((packed >> (16u + group_shift)) & 3u) - 1;
        w3 = i32((packed >> (24u + group_shift)) & 3u) - 1;
      }
      let sm_base = local_row * TILE_K + base_gp;
      shared_w[sm_base]      = w0;
      shared_w[sm_base + 1u] = w1;
      shared_w[sm_base + 2u] = w2;
      shared_w[sm_base + 3u] = w3;
    }

    // Cooperatively load input tile into shared memory
    let load_count_x = (TILE_K * TILE_N) / (THREADS_M * THREADS_N);
    for (var ld = 0u; ld < load_count_x; ld++) {
      let idx = linear_id + ld * (THREADS_M * THREADS_N);
      let local_k = idx / TILE_N;
      let local_col = idx % TILE_N;
      let global_k = k_base + local_k;
      let global_col = wg_col + local_col;

      var x_val: i32 = 0;
      if (global_k < params.K && global_col < params.N) {
        x_val = input[global_col * params.K + global_k];
      }
      shared_x[local_k * TILE_N + local_col] = x_val;
    }

    workgroupBarrier();

    // Compute per-thread 4×4 accumulation
    for (var k = 0u; k < TILE_K; k++) {
      for (var tm = 0u; tm < THREAD_TILE_M; tm++) {
        let w = shared_w[(tid_m * THREAD_TILE_M + tm) * TILE_K + k];
        for (var tn = 0u; tn < THREAD_TILE_N; tn++) {
          let x = shared_x[k * TILE_N + tid_n * THREAD_TILE_N + tn];
          acc[tm * THREAD_TILE_N + tn] += w * x;
        }
      }
    }

    workgroupBarrier();
  }

  // Write results with dequantization
  for (var tm = 0u; tm < THREAD_TILE_M; tm++) {
    let global_row = wg_row + tid_m * THREAD_TILE_M + tm;
    if (global_row >= params.M) { continue; }
    let w_scale = scales[global_row];
    for (var tn = 0u; tn < THREAD_TILE_N; tn++) {
      let global_col = wg_col + tid_n * THREAD_TILE_N + tn;
      if (global_col >= params.N) { continue; }
      let scale = w_scale * input_scales[global_col];
      output[global_col * params.M + global_row] = f32(acc[tm * THREAD_TILE_N + tn]) * scale;
    }
  }
}
`,T=class{constructor(r,e,t,n,s,i,o,a){p(this,"device");p(this,"pipelines");p(this,"pool");p(this,"packedWeights");p(this,"weightScales");p(this,"normWeight");p(this,"inDim");p(this,"outDim");p(this,"kPacked");p(this,"decodeNormUniform");p(this,"decodeQuantUniform");p(this,"decodeGemvParamsUniform");p(this,"decodeGemvScaleUniform");p(this,"prefillNormUniform");p(this,"prefillQuantUniform");p(this,"prefillGemmUniform");p(this,"bgCache",ie());this.device=r,this.pipelines=e,this.pool=t,this.packedWeights=n,this.weightScales=s,this.normWeight=i,this.inDim=o,this.outDim=a,this.kPacked=Math.ceil(o/16)}initDecodeUniforms(){if(this.normWeight){const e=new ArrayBuffer(12),t=new DataView(e);t.setUint32(0,1,!0),t.setUint32(4,this.inDim,!0),t.setFloat32(8,1e-5,!0),this.decodeNormUniform=v(this.device,e)}{const e=new ArrayBuffer(8),t=new DataView(e);t.setUint32(0,1,!0),t.setUint32(4,this.inDim,!0),this.decodeQuantUniform=v(this.device,e)}{const e=new ArrayBuffer(12),t=new DataView(e);t.setUint32(0,this.outDim,!0),t.setUint32(4,this.inDim,!0),t.setUint32(8,this.kPacked,!0),this.decodeGemvParamsUniform=v(this.device,e)}this.decodeGemvScaleUniform=this.device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const r=e=>this.device.createBuffer({size:e,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.normWeight&&(this.prefillNormUniform=r(12)),this.prefillQuantUniform=r(8),this.prefillGemmUniform=r(16)}forward(r,e,t){let n,s,i;this.normWeight?(i=this.pool.acquire(e*this.inDim*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),this.dispatchRMSNorm(t,r,i,e)):i=r,n=this.pool.acquire(e*this.inDim*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),s=this.pool.acquire(e*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.UNIFORM),this.dispatchQuantize(t,i,n,s,e);const o=this.pool.acquire(e*this.outDim*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);return e===1?this.dispatchGEMV(t,n,s,o):this.dispatchGEMM(t,n,s,o,e),this.normWeight&&this.pool.release(i),this.pool.release(n),this.pool.release(s),o}dispatchRMSNorm(r,e,t,n){const{pipeline:s,bindGroupLayout:i}=this.pipelines.getOrCreate("rmsnorm",_e);let o;if(n===1&&this.decodeNormUniform)o=this.decodeNormUniform;else if(this.prefillNormUniform){const l=new ArrayBuffer(12),h=new DataView(l);h.setUint32(0,n,!0),h.setUint32(4,this.inDim,!0),h.setFloat32(8,1e-5,!0),this.device.queue.writeBuffer(this.prefillNormUniform,0,new Uint8Array(l)),o=this.prefillNormUniform}else{const l=new ArrayBuffer(12),h=new DataView(l);h.setUint32(0,n,!0),h.setUint32(4,this.inDim,!0),h.setFloat32(8,1e-5,!0),o=v(this.device,l)}const a=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.normWeight}},{binding:2,resource:{buffer:t}},{binding:3,resource:{buffer:o}}],u=n===1?z(this.bgCache,this.device,"rmsnorm",i,a):this.device.createBindGroup({layout:i,entries:a}),d=r.beginComputePass();d.setPipeline(s),d.setBindGroup(0,u),d.dispatchWorkgroups(n),d.end()}dispatchQuantize(r,e,t,n,s){const{pipeline:i,bindGroupLayout:o}=this.pipelines.getOrCreate("quantize",We);let a;if(s===1&&this.decodeQuantUniform)a=this.decodeQuantUniform;else if(this.prefillQuantUniform){const h=new ArrayBuffer(8),c=new DataView(h);c.setUint32(0,s,!0),c.setUint32(4,this.inDim,!0),this.device.queue.writeBuffer(this.prefillQuantUniform,0,new Uint8Array(h)),a=this.prefillQuantUniform}else{const h=new ArrayBuffer(8),c=new DataView(h);c.setUint32(0,s,!0),c.setUint32(4,this.inDim,!0),a=v(this.device,h)}const u=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:n}},{binding:3,resource:{buffer:a}}],d=s===1?z(this.bgCache,this.device,"quantize",o,u):this.device.createBindGroup({layout:o,entries:u}),l=r.beginComputePass();l.setPipeline(i),l.setBindGroup(0,d),l.dispatchWorkgroups(s),l.end()}dispatchGEMV(r,e,t,n){const{pipeline:s,bindGroupLayout:i}=this.pipelines.getOrCreate("ternary_gemv",Ke);let o,a;if(this.decodeGemvParamsUniform&&this.decodeGemvScaleUniform)o=this.decodeGemvParamsUniform,a=this.decodeGemvScaleUniform;else{const h=new ArrayBuffer(12),c=new DataView(h);c.setUint32(0,this.outDim,!0),c.setUint32(4,this.inDim,!0),c.setUint32(8,this.kPacked,!0),o=v(this.device,h),a=v(this.device,new ArrayBuffer(4))}r.copyBufferToBuffer(t,0,a,0,4);const u=[{binding:0,resource:{buffer:this.packedWeights}},{binding:1,resource:{buffer:e}},{binding:2,resource:{buffer:this.weightScales}},{binding:3,resource:{buffer:o}},{binding:4,resource:{buffer:a}},{binding:5,resource:{buffer:n}}],d=z(this.bgCache,this.device,"gemv",i,u),l=r.beginComputePass();l.setPipeline(s),l.setBindGroup(0,d),l.dispatchWorkgroups(this.outDim),l.end()}dispatchGEMM(r,e,t,n,s){const{pipeline:i,bindGroupLayout:o}=this.pipelines.getOrCreate("ternary_gemm",je),a=new ArrayBuffer(16),u=new DataView(a);u.setUint32(0,this.outDim,!0),u.setUint32(4,s,!0),u.setUint32(8,this.inDim,!0),u.setUint32(12,this.kPacked,!0);let d;this.prefillGemmUniform?(this.device.queue.writeBuffer(this.prefillGemmUniform,0,new Uint8Array(a)),d=this.prefillGemmUniform):d=v(this.device,a);const l=this.device.createBindGroup({layout:o,entries:[{binding:0,resource:{buffer:this.packedWeights}},{binding:1,resource:{buffer:e}},{binding:2,resource:{buffer:this.weightScales}},{binding:3,resource:{buffer:d}},{binding:4,resource:{buffer:t}},{binding:5,resource:{buffer:n}}]}),h=Math.ceil(this.outDim/64),c=Math.ceil(s/64),f=r.beginComputePass();f.setPipeline(i),f.setBindGroup(0,l),f.dispatchWorkgroups(h,c),f.end()}clearBGCache(){Y(this.bgCache)}},Ye=`// Rotary Position Embeddings (RoPE)
//
// For each pair (x[2i], x[2i+1]) at position \`pos\`:
//   theta = pos * base^(-2i/D)
//   out[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
//   out[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)
//
// Layout:
//   input:  [N, num_heads, head_dim] f32
//   output: [N, num_heads, head_dim] f32
//   Dispatched per (token, head, pair)

struct Params {
  N: u32,          // sequence length
  num_heads: u32,
  head_dim: u32,
  pos_offset: u32, // starting position (for KV-cache continuation)
  theta_base: f32, // default 10000.0 or 500000.0
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.N * params.num_heads * half_dim;

  let pair_idx = gid.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  // Decompose linear index into (token, head, dim_pair)
  let dim_pair = pair_idx % half_dim;
  let remainder = pair_idx / half_dim;
  let head = remainder % params.num_heads;
  let token = remainder / params.num_heads;

  let pos = f32(token + params.pos_offset);
  let freq_exp = -2.0 * f32(dim_pair) / f32(params.head_dim);
  let theta = pos * pow(params.theta_base, freq_exp);

  let cos_theta = cos(theta);
  let sin_theta = sin(theta);

  let base_idx = (token * params.num_heads + head) * params.head_dim + dim_pair * 2u;
  let x0 = input[base_idx];
  let x1 = input[base_idx + 1u];

  output[base_idx]      = x0 * cos_theta - x1 * sin_theta;
  output[base_idx + 1u] = x0 * sin_theta + x1 * cos_theta;
}
`,Qe=`// Numerically stable softmax
//
// For each row:
//   1. Find max value (for numerical stability)
//   2. Compute sum of exp(x - max)
//   3. Normalize: out[i] = exp(x[i] - max) / sum
//
// Layout:
//   input:  [N, D] f32
//   output: [N, D] f32

struct Params {
  N: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.D;

  // Pass 1: Find max
  var local_max: f32 = -3.402823e+38; // -FLT_MAX
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_max = max(local_max, input[row_offset + col]);
  }

  shared_val[tid] = local_max;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);
    }
    workgroupBarrier();
  }

  let row_max = shared_val[0];
  workgroupBarrier();

  // Pass 2: Sum of exp(x - max)
  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_sum += exp(input[row_offset + col] - row_max);
  }

  shared_val[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] += shared_val[tid + stride];
    }
    workgroupBarrier();
  }

  let inv_sum = 1.0 / shared_val[0];
  workgroupBarrier();

  // Pass 3: Normalize
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = exp(input[row_offset + col] - row_max) * inv_sum;
  }
}
`,Be=`// Standard f32 attention matmul kernels
//
// Two operations:
//   1. scores = Q @ K^T * scale    (score computation)
//   2. output = attn_weights @ V   (value aggregation)
//
// These use standard f32 matmul (not ternary) because Q,K,V are
// already projected through BitLinear and are f32 activations.

// ─── Kernel 1: Q @ K^T (score computation) ───
// Q:      [N, num_heads, head_dim]
// K:      [S, num_kv_heads, head_dim]  (S = total seq including cache)
// scores: [num_heads, N, S]

struct ScoreParams {
  N: u32,           // query seq length
  S: u32,           // key seq length (including cache)
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  scale: f32,       // 1/sqrt(head_dim)
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: ScoreParams;

@compute @workgroup_size(16, 16)
fn compute_scores(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  // gid.x = query position, gid.y = key position, gid.z = head
  let q_pos = gid.x;
  let k_pos = gid.y;
  let head = gid.z;

  if (q_pos >= params.N || k_pos >= params.S || head >= params.num_heads) {
    return;
  }

  // GQA: map attention head to KV head
  let kv_head = head / (params.num_heads / params.num_kv_heads);

  let q_offset = (q_pos * params.num_heads + head) * params.head_dim;
  let k_offset = (k_pos * params.num_kv_heads + kv_head) * params.head_dim;

  var dot: f32 = 0.0;
  for (var d = 0u; d < params.head_dim; d++) {
    dot += Q[q_offset + d] * K[k_offset + d];
  }

  // Causal mask: positions after query are -inf
  let is_causal = k_pos > q_pos + (params.S - params.N);
  let masked_score = select(dot * params.scale, -3.402823e+38, is_causal);

  let score_idx = (head * params.N + q_pos) * params.S + k_pos;
  scores[score_idx] = masked_score;
}

// ─── Kernel 2: Attention weights @ V ───
// attn:   [num_heads, N, S]
// V:      [S, num_kv_heads, head_dim]
// output: [N, num_heads, head_dim]

struct AttnVParams {
  N: u32,
  S: u32,
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
}

@group(0) @binding(0) var<storage, read> attn: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> attn_output: array<f32>;
@group(0) @binding(3) var<uniform> attn_v_params: AttnVParams;

@compute @workgroup_size(256)
fn attn_v(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let total = attn_v_params.N * attn_v_params.num_heads * attn_v_params.head_dim;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let d = idx % attn_v_params.head_dim;
  let remainder = idx / attn_v_params.head_dim;
  let head = remainder % attn_v_params.num_heads;
  let q_pos = remainder / attn_v_params.num_heads;

  let kv_head = head / (attn_v_params.num_heads / attn_v_params.num_kv_heads);

  var sum: f32 = 0.0;
  for (var s = 0u; s < attn_v_params.S; s++) {
    let a = attn[(head * attn_v_params.N + q_pos) * attn_v_params.S + s];
    let v = V[(s * attn_v_params.num_kv_heads + kv_head) * attn_v_params.head_dim + d];
    sum += a * v;
  }

  let out_idx = (q_pos * attn_v_params.num_heads + head) * attn_v_params.head_dim + d;
  attn_output[out_idx] = sum;
}
`,Ze=class{constructor(r,e,t,n,s,i,o,a){p(this,"device");p(this,"pipelines");p(this,"pool");p(this,"config");p(this,"hDim");p(this,"qProj");p(this,"kProj");p(this,"vProj");p(this,"oProj");p(this,"decodeRopeQUniform");p(this,"decodeRopeKUniform");p(this,"decodeScoresUniform");p(this,"decodeSoftmaxUniform");p(this,"decodeAttnVUniform");p(this,"prefillRopeQUniform");p(this,"prefillRopeKUniform");p(this,"prefillScoresUniform");p(this,"prefillSoftmaxUniform");p(this,"prefillAttnVUniform");p(this,"decodeScoresBuf");p(this,"decodeAttnWeightsBuf");p(this,"bgCache",ie());this.device=r,this.pipelines=e,this.pool=t,this.config=n,this.hDim=j(n),this.qProj=s,this.kProj=i,this.vProj=o,this.oProj=a}initDecodeUniforms(r){const e=n=>this.device.createBuffer({size:n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.decodeRopeQUniform=e(20),this.decodeRopeKUniform=e(20),this.decodeScoresUniform=e(24),this.decodeSoftmaxUniform=e(8),this.decodeAttnVUniform=e(20),this.prefillRopeQUniform=e(20),this.prefillRopeKUniform=e(20),this.prefillScoresUniform=e(24),this.prefillSoftmaxUniform=e(8),this.prefillAttnVUniform=e(20);const t=this.config.numAttentionHeads*r*4;this.decodeScoresBuf=this.device.createBuffer({size:t,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.decodeAttnWeightsBuf=this.device.createBuffer({size:t,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.qProj.initDecodeUniforms(),this.kProj.initDecodeUniforms(),this.vProj.initDecodeUniforms(),this.oProj.initDecodeUniforms()}forward(r,e,t,n){const{numAttentionHeads:s,numKeyValueHeads:i,hiddenSize:o}=this.config,a=this.qProj.forward(r,e,n),u=this.kProj.forward(r,e,n),d=this.vProj.forward(r,e,n),l=this.applyRoPE(n,a,e,s,t.seqLen,e===1?this.decodeRopeQUniform:this.prefillRopeQUniform,"ropeQ"),h=this.applyRoPE(n,u,e,i,t.seqLen,e===1?this.decodeRopeKUniform:this.prefillRopeKUniform,"ropeK");this.pool.release(a),this.pool.release(u),this.appendToCache(n,h,d,t,e),this.pool.release(h),this.pool.release(d);const c=t.seqLen+e,f=this.computeScores(n,l,t.key,e,c,e===1?this.decodeScoresUniform:this.prefillScoresUniform,e===1?this.decodeScoresBuf:void 0);this.pool.release(l);const m=this.applySoftmax(n,f,s*e,c,e===1?this.decodeSoftmaxUniform:this.prefillSoftmaxUniform,e===1?this.decodeAttnWeightsBuf:void 0);e!==1&&this.pool.release(f);const g=this.computeAttnV(n,m,t.value,e,c,e===1?this.decodeAttnVUniform:this.prefillAttnVUniform);e!==1&&this.pool.release(m);const w=this.oProj.forward(g,e,n);return this.pool.release(g),w}applyRoPE(r,e,t,n,s,i,o){const{pipeline:a,bindGroupLayout:u}=this.pipelines.getOrCreate("rope",Ye),d=t*n*this.hDim*4,l=this.pool.acquire(d,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),h=new ArrayBuffer(20),c=new DataView(h);c.setUint32(0,t,!0),c.setUint32(4,n,!0),c.setUint32(8,this.hDim,!0),c.setUint32(12,s,!0),c.setFloat32(16,this.config.ropeTheta,!0);let f;i?(this.device.queue.writeBuffer(i,0,new Uint8Array(h)),f=i):f=v(this.device,h);const m=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:f}}],g=t===1&&o?z(this.bgCache,this.device,o,u,m):this.device.createBindGroup({layout:u,entries:m}),w=t*n*(this.hDim/2),_=r.beginComputePass();return _.setPipeline(a),_.setBindGroup(0,g),_.dispatchWorkgroups(Math.ceil(w/256)),_.end(),l}appendToCache(r,e,t,n,s){const i=s*this.config.numKeyValueHeads*this.hDim*4,o=n.seqLen*this.config.numKeyValueHeads*this.hDim*4;r.copyBufferToBuffer(e,0,n.key,o,i),r.copyBufferToBuffer(t,0,n.value,o,i)}computeScores(r,e,t,n,s,i,o){const{pipeline:a,bindGroupLayout:u}=this.pipelines.getOrCreate("attention_scores",Be,"compute_scores"),{numAttentionHeads:d,numKeyValueHeads:l}=this.config,h=o!=null?o:this.pool.acquire(d*n*s*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),c=new ArrayBuffer(24),f=new DataView(c);f.setUint32(0,n,!0),f.setUint32(4,s,!0),f.setUint32(8,d,!0),f.setUint32(12,l,!0),f.setUint32(16,this.hDim,!0),f.setFloat32(20,1/Math.sqrt(this.hDim),!0);let m;i?(this.device.queue.writeBuffer(i,0,new Uint8Array(c)),m=i):m=v(this.device,c);const g=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:h}},{binding:3,resource:{buffer:m}}],w=n===1?z(this.bgCache,this.device,"scores",u,g):this.device.createBindGroup({layout:u,entries:g}),_=r.beginComputePass();return _.setPipeline(a),_.setBindGroup(0,w),_.dispatchWorkgroups(Math.ceil(n/16),Math.ceil(s/16),d),_.end(),h}applySoftmax(r,e,t,n,s,i){const{pipeline:o,bindGroupLayout:a}=this.pipelines.getOrCreate("softmax",Qe),u=i!=null?i:this.pool.acquire(t*n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),d=new ArrayBuffer(8),l=new DataView(d);l.setUint32(0,t,!0),l.setUint32(4,n,!0);let h;s?(this.device.queue.writeBuffer(s,0,new Uint8Array(d)),h=s):h=v(this.device,d);const c=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:u}},{binding:2,resource:{buffer:h}}],f=t===1?z(this.bgCache,this.device,"softmax",a,c):this.device.createBindGroup({layout:a,entries:c}),m=r.beginComputePass();return m.setPipeline(o),m.setBindGroup(0,f),m.dispatchWorkgroups(t),m.end(),u}computeAttnV(r,e,t,n,s,i){const{pipeline:o,bindGroupLayout:a}=this.pipelines.getOrCreate("attn_v",Be,"attn_v"),{numAttentionHeads:u,numKeyValueHeads:d}=this.config,l=n*u*this.hDim*4,h=this.pool.acquire(l,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),c=new ArrayBuffer(20),f=new DataView(c);f.setUint32(0,n,!0),f.setUint32(4,s,!0),f.setUint32(8,u,!0),f.setUint32(12,d,!0),f.setUint32(16,this.hDim,!0);let m;i?(this.device.queue.writeBuffer(i,0,new Uint8Array(c)),m=i):m=v(this.device,c);const g=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:h}},{binding:3,resource:{buffer:m}}],w=n===1?z(this.bgCache,this.device,"attnV",a,g):this.device.createBindGroup({layout:a,entries:g}),_=n*u*this.hDim,b=r.beginComputePass();return b.setPipeline(o),b.setBindGroup(0,w),b.dispatchWorkgroups(Math.ceil(_/256)),b.end(),h}clearBGCache(){Y(this.bgCache),this.qProj.clearBGCache(),this.kProj.clearBGCache(),this.vProj.clearBGCache(),this.oProj.clearBGCache()}destroyPreAllocated(){var r,e;(r=this.decodeScoresBuf)==null||r.destroy(),(e=this.decodeAttnWeightsBuf)==null||e.destroy(),this.decodeScoresBuf=void 0,this.decodeAttnWeightsBuf=void 0}};function Xe(r,e,t){const n=t*e.numKeyValueHeads*j(e)*4,s=r.createBuffer({size:n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),i=r.createBuffer({size:n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});return{key:s,value:i,seqLen:0,maxSeqLen:t}}var Je=`// Activation functions for BitNet FFN
//
// ReLU²: relu(x)² — used in official 2B-4T model
// SiLU:  x * sigmoid(x) — used in community models
//
// Layout:
//   input:  [N] f32
//   output: [N] f32

struct Params {
  N: u32,
  activation_type: u32,  // 0 = ReLU², 1 = SiLU
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  let x = input[idx];

  if (params.activation_type == 0u) {
    // ReLU²: max(0, x)²
    let relu_x = max(0.0, x);
    output[idx] = relu_x * relu_x;
  } else {
    // SiLU: x * sigmoid(x)
    output[idx] = x / (1.0 + exp(-x));
  }
}
`,et=class{constructor(r,e,t,n,s,i,o){p(this,"device");p(this,"pipelines");p(this,"pool");p(this,"config");p(this,"upProj");p(this,"downProj");p(this,"gateProj");p(this,"decodeActivationUniform");p(this,"decodeElementwiseUniform");p(this,"prefillActivationUniform");p(this,"prefillElementwiseUniform");p(this,"bgCache",ie());this.device=r,this.pipelines=e,this.pool=t,this.config=n,this.upProj=s,this.downProj=i,this.gateProj=o}initDecodeUniforms(){var t;const r=this.config.activation==="relu2"?0:1;{const n=new ArrayBuffer(8),s=new DataView(n);s.setUint32(0,this.config.intermediateSize,!0),s.setUint32(4,r,!0),this.decodeActivationUniform=v(this.device,n)}{const n=new ArrayBuffer(8),s=new DataView(n);s.setUint32(0,this.config.intermediateSize,!0),s.setUint32(4,1,!0),this.decodeElementwiseUniform=v(this.device,n)}const e=n=>this.device.createBuffer({size:n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.prefillActivationUniform=e(8),this.prefillElementwiseUniform=e(8),this.upProj.initDecodeUniforms(),this.downProj.initDecodeUniforms(),(t=this.gateProj)==null||t.initDecodeUniforms()}forward(r,e,t){return this.gateProj?this.forwardGated(r,e,t):this.forwardSimple(r,e,t)}forwardGated(r,e,t){const n=this.config.activation==="relu2"?0:1,s=this.gateProj.forward(r,e,t),i=this.upProj.forward(r,e,t),o=this.applyActivation(t,s,e*this.config.intermediateSize,n,e);this.pool.release(s);const a=this.applyElementwise(t,o,i,e*this.config.intermediateSize,1,e);this.pool.release(o),this.pool.release(i);const u=this.downProj.forward(a,e,t);return this.pool.release(a),u}forwardSimple(r,e,t){const n=this.config.activation==="relu2"?0:1,s=this.upProj.forward(r,e,t),i=this.applyActivation(t,s,e*this.config.intermediateSize,n,e);this.pool.release(s);const o=this.downProj.forward(i,e,t);return this.pool.release(i),o}applyActivation(r,e,t,n,s){const{pipeline:i,bindGroupLayout:o}=this.pipelines.getOrCreate(`activation_${n}`,Je),a=this.pool.acquire(t*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let u;if(s===1&&this.decodeActivationUniform)u=this.decodeActivationUniform;else if(this.prefillActivationUniform){const c=new ArrayBuffer(8),f=new DataView(c);f.setUint32(0,t,!0),f.setUint32(4,n,!0),this.device.queue.writeBuffer(this.prefillActivationUniform,0,new Uint8Array(c)),u=this.prefillActivationUniform}else{const c=new ArrayBuffer(8),f=new DataView(c);f.setUint32(0,t,!0),f.setUint32(4,n,!0),u=v(this.device,c)}const d=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:u}}],l=s===1?z(this.bgCache,this.device,"activation",o,d):this.device.createBindGroup({layout:o,entries:d}),h=r.beginComputePass();return h.setPipeline(i),h.setBindGroup(0,l),h.dispatchWorkgroups(Math.ceil(t/256)),h.end(),a}applyElementwise(r,e,t,n,s,i){const{pipeline:o,bindGroupLayout:a}=this.pipelines.getOrCreate(`elementwise_${s}`,ye),u=this.pool.acquire(n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let d;if(i===1&&this.decodeElementwiseUniform)d=this.decodeElementwiseUniform;else if(this.prefillElementwiseUniform){const f=new ArrayBuffer(8),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,s,!0),this.device.queue.writeBuffer(this.prefillElementwiseUniform,0,new Uint8Array(f)),d=this.prefillElementwiseUniform}else{const f=new ArrayBuffer(8),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,s,!0),d=v(this.device,f)}const l=[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:d}}],h=i===1?z(this.bgCache,this.device,"elementwise",a,l):this.device.createBindGroup({layout:a,entries:l}),c=r.beginComputePass();return c.setPipeline(o),c.setBindGroup(0,h),c.dispatchWorkgroups(Math.ceil(n/256)),c.end(),u}clearBGCache(){var r;Y(this.bgCache),this.upProj.clearBGCache(),this.downProj.clearBGCache(),(r=this.gateProj)==null||r.clearBGCache()}},ke=class extends Error{constructor(r){super(r),this.name="GPUDeviceError"}};function tt(r){return C(this,null,function*(){if(typeof navigator=="undefined"||!navigator.gpu)throw new ke("WebGPU is not supported in this environment. Please use a browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly).");const e=yield navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e)throw new ke("Failed to obtain WebGPU adapter. Check that your GPU drivers are up to date.");const t={};t.maxBufferSize=e.limits.maxBufferSize,t.maxStorageBufferBindingSize=e.limits.maxStorageBufferBindingSize;const n=e.limits.maxStorageBuffersPerShaderStage;t.maxStorageBuffersPerShaderStage=n,t.maxComputeWorkgroupSizeX=e.limits.maxComputeWorkgroupSizeX,t.maxComputeWorkgroupSizeY=e.limits.maxComputeWorkgroupSizeY,t.maxComputeWorkgroupSizeZ=e.limits.maxComputeWorkgroupSizeZ,t.maxComputeInvocationsPerWorkgroup=e.limits.maxComputeInvocationsPerWorkgroup,t.maxComputeWorkgroupStorageSize=e.limits.maxComputeWorkgroupStorageSize;const s=yield e.requestDevice({requiredLimits:t});return s.lost.then(i=>{console.error(`WebGPU device lost: ${i.message} (reason: ${i.reason})`)}),{device:s,adapter:e,limits:s.limits}})}var Se=class{constructor(r){p(this,"buffers",new Map);p(this,"device");this.device=r}upload(r,e){const t=this.device.createBuffer({size:Math.max(e.byteLength,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(t.getMappedRange()).set(new Uint8Array(e)),t.unmap(),this.buffers.set(r,t),t}uploadSharded(r,e,t){if(e.byteLength<=t)return[this.upload(r,e)];const n=[];let s=0,i=0;for(;s<e.byteLength;){const o=Math.min(s+t,e.byteLength),a=e.slice(s,o),u=`${r}.shard_${i}`;n.push(this.upload(u,a)),s=o,i++}return n.length>0&&!this.buffers.has(r)&&this.buffers.set(r,n[0]),n}get(r){return this.buffers.get(r)}has(r){return this.buffers.has(r)}destroy(){for(const r of this.buffers.values())r.destroy();this.buffers.clear()}};function nt(r,e,t,n){return C(this,null,function*(){const s=typeof r=="string"?r:r.href,i=it(s);t==null||t({phase:"download",loaded:0,total:0,fraction:0});const o=yield gt(s,(a,u)=>{t==null||t({phase:"download",loaded:a,total:u,fraction:u>0?a/u:0})},n);return t==null||t({phase:"parse",loaded:0,total:1,fraction:0}),i==="gguf"?rt(o,e,t):ut(o,e,t)})}function it(r){return r.endsWith(".gguf")?"gguf":r.endsWith(".safetensors")?"safetensors":"gguf"}function rt(r,e,t){return C(this,null,function*(){const s=new Le(r).parse(),i=dt(s.metadata),o=s.tensors.some(l=>l.name==="output.weight");i.tieWordEmbeddings=!o,console.debug(`[0xBitNet] config: arch=${s.metadata["general.architecture"]}, heads=${i.numAttentionHeads}, kv_heads=${i.numKeyValueHeads}, head_dim=${i.hiddenSize/i.numAttentionHeads}, hidden=${i.hiddenSize}, intermediate=${i.intermediateSize}, layers=${i.numHiddenLayers}, tied=${i.tieWordEmbeddings}`);const a=new Se(e),u=e.limits.maxStorageBufferBindingSize,d=s.tensors.length;for(let l=0;l<d;l++){const h=s.tensors[l],c=s.tensorDataOffset+Number(h.offset),f=h.shape.reduce((_,b)=>_*Number(b),1);let m;if(h.type===ge)m=Math.ceil(f/4)+32;else{const _=Re(h.type);m=Math.ceil(f*_)}const g=r.slice(c,c+m),w=at(h.name);if(console.debug(`[0xBitNet] tensor: ${h.name} → ${w} (type=${h.type}, ${m} bytes)`),h.type===ge){const _=Math.ceil(f/4),b=g.slice(0,_);a.uploadSharded(w,b,u);const E=new DataView(g,_,32).getFloat32(0,!0),A=Number(h.shape[1]),R=w.replace(".weight",".weight_scale"),M=new Float32Array(A).fill(E);a.upload(R,M.buffer)}else if(h.type===L)if(w==="model.embed_tokens.weight")a.uploadSharded(w,g,u);else if(w==="lm_head.weight")a.uploadSharded(w,g,u),i.lmHeadF16=!0;else{const _=st(new Uint16Array(g),f);a.uploadSharded(w,_.buffer,u)}else a.uploadSharded(w,g,u);t==null||t({phase:"upload",loaded:l+1,total:d,fraction:(l+1)/d})}return console.debug(`[0xBitNet] ${d} tensors loaded, tieWordEmbeddings=${i.tieWordEmbeddings}`),ot(a,i),{config:i,weights:a,metadata:s.metadata}})}function st(r,e){const t=new Float32Array(e);for(let n=0;n<e;n++){const s=r[n],i=s>>15&1,o=s>>10&31,a=s&1023;let u;o===0?u=a/1024*Math.pow(2,-14):o===31?u=a===0?1/0:NaN:u=(1+a/1024)*Math.pow(2,o-15),t[n]=i?-u:u}return t}function at(r){if(r==="token_embd.weight")return"model.embed_tokens.weight";if(r==="output_norm.weight")return"model.norm.weight";if(r==="output.weight")return"lm_head.weight";const e=r.match(/^blk\.(\d+)\.(.+)$/);if(!e)return r;const[,t,n]=e,s=`model.layers.${t}`,o={"attn_q.weight":"self_attn.q_proj.weight","attn_k.weight":"self_attn.k_proj.weight","attn_v.weight":"self_attn.v_proj.weight","attn_output.weight":"self_attn.o_proj.weight","attn_norm.weight":"input_layernorm.weight","ffn_norm.weight":"post_attention_layernorm.weight","attn_sub_norm.weight":"self_attn.sub_norm.weight","ffn_sub_norm.weight":"mlp.sub_norm.weight","ffn_up.weight":"mlp.up_proj.weight","ffn_down.weight":"mlp.down_proj.weight","ffn_gate.weight":"mlp.gate_proj.weight"}[n];return o?`${s}.${o}`:`${s}.${n}`}function ot(r,e,t){const n=[];for(let s=0;s<e.numHiddenLayers;s++){const i=`model.layers.${s}`,o=e.hiddenSize,a=e.numAttentionHeads,u=e.numKeyValueHeads,d=o/a;n.push({name:`${i}.self_attn.q_proj.weight_scale`,dim:a*d},{name:`${i}.self_attn.k_proj.weight_scale`,dim:u*d},{name:`${i}.self_attn.v_proj.weight_scale`,dim:u*d},{name:`${i}.self_attn.o_proj.weight_scale`,dim:o},{name:`${i}.mlp.up_proj.weight_scale`,dim:e.intermediateSize},{name:`${i}.mlp.down_proj.weight_scale`,dim:o},{name:`${i}.mlp.gate_proj.weight_scale`,dim:e.intermediateSize})}e.lmHeadF16||n.push({name:"lm_head.weight_scale",dim:e.vocabSize});for(const{name:s,dim:i}of n)if(!r.has(s)){const o=new Float32Array(i).fill(1);r.upload(s,o.buffer)}}function ut(r,e,t){return C(this,null,function*(){const{header:n,dataOffset:s}=Oe(r),i=qe(n,s),o=ct(i),a=new Se(e),u=e.limits.maxStorageBufferBindingSize;for(let d=0;d<i.length;d++){const l=i[d],h=r.slice(l.offset,l.offset+l.size);a.uploadSharded(l.name,h,u),t==null||t({phase:"upload",loaded:d+1,total:i.length,fraction:(d+1)/i.length})}return{config:o,weights:a}})}function dt(r){var c,f,m,g,w,_,b,S,E,A,R,M;const e=(c=r["general.architecture"])!=null?c:"bitnet",t=e.startsWith("bitnet");function n(I){var ee,U,D,x;return(x=(D=(U=(ee=r[`${e}.${I}`])!=null?ee:r[`llama.${I}`])!=null?U:r[`bitnet.${I}`])!=null?D:r[`bitnet-25.${I}`])!=null?x:r[`bitnet-b1.58.${I}`]}const s=(f=n("embedding_length"))!=null?f:2560,i=(m=n("block_count"))!=null?m:30,o=(g=n("attention.head_count"))!=null?g:20,a=(w=n("attention.head_count_kv"))!=null?w:o,u=(S=(b=n("vocab_size"))!=null?b:(_=r["tokenizer.ggml.tokens"])==null?void 0:_.length)!=null?S:128256,d=(E=n("feed_forward_length"))!=null?E:6912,l=t?"relu2":"silu",h=(A=n("rope.freq_base"))!=null?A:t?5e5:1e4;return{modelType:"bitnet",vocabSize:u,hiddenSize:s,intermediateSize:d,numHiddenLayers:i,numAttentionHeads:o,numKeyValueHeads:a,maxPositionEmbeddings:(R=n("context_length"))!=null?R:4096,rmsNormEps:(M=n("attention.layer_norm_rms_epsilon"))!=null?M:1e-5,ropeTheta:h,tieWordEmbeddings:!1,activation:l}}function ct(r){var c,f,m;const e=r.find(g=>g.name==="model.embed_tokens.weight"||g.name==="transformer.wte.weight"),t=(c=e==null?void 0:e.shape[0])!=null?c:128256,n=(f=e==null?void 0:e.shape[1])!=null?f:2560,s=r.map(g=>{const w=g.name.match(/layers\.(\d+)\./);return w?parseInt(w[1],10):-1}).filter(g=>g>=0),i=s.length>0?Math.max(...s)+1:30,o=r.find(g=>g.name.includes("q_proj.weight")),a=o?o.shape[0]/(n/32):32,u=r.find(g=>g.name.includes("k_proj.weight")),d=(m=u==null?void 0:u.shape[0])!=null?m:n,l=n/a,h=d/l;return{modelType:"bitnet",vocabSize:t,hiddenSize:n,intermediateSize:0,numHiddenLayers:i,numAttentionHeads:a,numKeyValueHeads:h,maxPositionEmbeddings:4096,rmsNormEps:1e-5,ropeTheta:1e4,tieWordEmbeddings:!1,activation:"silu"}}var ft="0xbitnet",V="models";function ce(){return new Promise((r,e)=>{const t=indexedDB.open(ft,1);t.onupgradeneeded=()=>t.result.createObjectStore(V),t.onsuccess=()=>r(t.result),t.onerror=()=>e(t.error)})}function lt(r,e){return new Promise((t,n)=>{const i=r.transaction(V,"readonly").objectStore(V).get(e);i.onsuccess=()=>t(i.result),i.onerror=()=>n(i.error)})}function ht(r,e,t){return new Promise((n,s)=>{const i=r.transaction(V,"readwrite");i.objectStore(V).put(t,e),i.oncomplete=()=>n(),i.onerror=()=>s(i.error)})}function pt(){return C(this,null,function*(){if(typeof indexedDB=="undefined")return[];try{const r=yield ce(),e=yield new Promise((t,n)=>{const i=r.transaction(V,"readonly").objectStore(V).getAllKeys();i.onsuccess=()=>t(i.result),i.onerror=()=>n(i.error)});return r.close(),e}catch(r){return[]}})}function mt(r){return C(this,null,function*(){if(typeof indexedDB=="undefined")return;const e=yield ce();yield new Promise((t,n)=>{const s=e.transaction(V,"readwrite");s.objectStore(V).delete(r),s.oncomplete=()=>t(),s.onerror=()=>n(s.error)}),e.close()})}function gt(r,e,t){return C(this,null,function*(){var l,h;if(typeof indexedDB!="undefined")try{const c=yield ce(),f=yield lt(c,r);if(c.close(),f)return e(f.byteLength,f.byteLength),f}catch(c){}const n=yield fetch(r,{signal:t});if(!n.ok)throw new Error(`Failed to fetch model: ${n.status} ${n.statusText}`);const s=parseInt((l=n.headers.get("content-length"))!=null?l:"0",10),i=(h=n.body)==null?void 0:h.getReader();if(!i){const c=yield n.arrayBuffer();return e(c.byteLength,c.byteLength),c}const o=[];let a=0;for(;;){const{done:c,value:f}=yield i.read();if(c)break;o.push(f),a+=f.byteLength,e(a,s)}const u=new Uint8Array(a);let d=0;for(const c of o)u.set(c,d),d+=c.byteLength;if(typeof indexedDB!="undefined")try{const c=yield ce();yield ht(c,r,u.buffer),c.close()}catch(c){}return u.buffer})}var _t=`// Token embedding lookup (F16 on GPU)
//
// For each token ID, copy the corresponding row from the embedding table.
// Embedding table is stored as packed F16 pairs (two f16 values per u32)
// to avoid exceeding maxStorageBufferBindingSize on most GPUs.
//
// Layout:
//   token_ids:  [N]          u32
//   embed_table: [V * D / 2] u32  (packed f16 pairs)
//   output:     [N, D]       f32

struct Params {
  N: u32,  // number of tokens
  D: u32,  // embedding dimension
  V: u32,  // vocab size
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  let total = params.N * params.D;
  if (idx >= total) {
    return;
  }

  let token = idx / params.D;
  let dim = idx % params.D;
  let token_id = token_ids[token];

  // Bounds check: treat out-of-vocab as zero
  if (token_id < params.V) {
    let flat = token_id * params.D + dim;
    let packed = embed_table[flat / 2u];
    let pair = unpack2x16float(packed);
    output[idx] = select(pair.x, pair.y, (flat & 1u) == 1u);
  } else {
    output[idx] = 0.0;
  }
}
`,bt=`// F32 GEMV for tied-embedding LM head (F16 embedding on GPU)
// logits[n, v] = sum_d( hidden[n, d] * embed[v, d] )
//
// hidden: [N, D] f32 — final hidden states
// embed:  [V * D / 2] u32 — embedding table stored as packed f16 pairs
// output: [N, V] f32 — logits
//
// Each workgroup computes one (n, v) element.
// 256 threads cooperatively reduce over D.
// 2D dispatch: v = wg_id.x + wg_id.y * 65535  (V can exceed 65535)

struct Params {
  N: u32,
  V: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> embed: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  // Decode (n, v) from 2D dispatch
  let flat_id = wg_id.x + wg_id.y * 65535u;
  let n = flat_id / params.V;
  let v = flat_id % params.V;

  if (n >= params.N || v >= params.V) {
    return;
  }

  let tid = local_id.x;

  // Each thread accumulates a strided slice of D
  // Process pairs of dimensions for efficiency
  var acc: f32 = 0.0;
  let hidden_base = n * params.D;
  let embed_base = v * params.D;

  // Process two dimensions at a time using packed f16 pairs
  let D_half = params.D / 2u;
  for (var dh = tid; dh < D_half; dh += WG_SIZE) {
    let d = dh * 2u;
    let packed = embed[embed_base / 2u + dh];
    let pair = unpack2x16float(packed);
    acc += hidden[hidden_base + d] * pair.x;
    acc += hidden[hidden_base + d + 1u] * pair.y;
  }

  // Workgroup reduction
  shared_sums[tid] = acc;
  workgroupBarrier();

  for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the result
  if (tid == 0u) {
    output[n * params.V + v] = shared_sums[0];
  }
}
`,wt=class Me{constructor(e,t,n,s,i,o,a,u,d){p(this,"device");p(this,"pipelines");p(this,"pool");p(this,"config");p(this,"embedTokens");p(this,"layers");p(this,"finalNorm");p(this,"lmHead");p(this,"kvCaches");p(this,"decodeTokenBuffer");p(this,"decodeEmbeddingUniform");p(this,"decodeFinalNormUniform");p(this,"decodeLMHeadUniform");p(this,"prefillEmbeddingUniform");p(this,"prefillFinalNormUniform");p(this,"prefillLMHeadUniform");p(this,"bgCache",ie());this.device=e,this.pipelines=t,this.pool=n,this.config=s,this.embedTokens=i,this.layers=o,this.finalNorm=a,this.lmHead=u,this.kvCaches=d}static build(e,t,n,s=4096){var m,g,w;const i=new He(e),o=new $e(e);function a(_){const b=n.get(_);if(!b)throw new Error(`Missing weight tensor: "${_}"`);return b}const u=a("model.embed_tokens.weight"),d=a("model.norm.weight"),l=[],h=[];for(let _=0;_<t.numHiddenLayers;_++){const b=`model.layers.${_}`,S=a(`${b}.input_layernorm.weight`),E=a(`${b}.post_attention_layernorm.weight`),A=(m=n.get(`${b}.self_attn.sub_norm.weight`))!=null?m:null,R=(g=n.get(`${b}.mlp.sub_norm.weight`))!=null?g:null,M=new T(e,i,o,a(`${b}.self_attn.q_proj.weight`),a(`${b}.self_attn.q_proj.weight_scale`),null,t.hiddenSize,t.numAttentionHeads*j(t)),I=new T(e,i,o,a(`${b}.self_attn.k_proj.weight`),a(`${b}.self_attn.k_proj.weight_scale`),null,t.hiddenSize,t.numKeyValueHeads*j(t)),ee=new T(e,i,o,a(`${b}.self_attn.v_proj.weight`),a(`${b}.self_attn.v_proj.weight_scale`),null,t.hiddenSize,t.numKeyValueHeads*j(t)),U=new T(e,i,o,a(`${b}.self_attn.o_proj.weight`),a(`${b}.self_attn.o_proj.weight_scale`),A,t.numAttentionHeads*j(t),t.hiddenSize),D=new Ze(e,i,o,t,M,I,ee,U),x=new T(e,i,o,a(`${b}.mlp.up_proj.weight`),a(`${b}.mlp.up_proj.weight_scale`),null,t.hiddenSize,t.intermediateSize),y=new T(e,i,o,a(`${b}.mlp.down_proj.weight`),a(`${b}.mlp.down_proj.weight_scale`),R,t.intermediateSize,t.hiddenSize);let B=null;n.has(`${b}.mlp.gate_proj.weight`)&&(B=new T(e,i,o,a(`${b}.mlp.gate_proj.weight`),a(`${b}.mlp.gate_proj.weight_scale`),null,t.hiddenSize,t.intermediateSize));const $=new et(e,i,o,t,x,y,B);l.push(new Fe(e,i,o,t,S,E,D,$)),h.push(Xe(e,t,s))}let c;t.tieWordEmbeddings||!n.has("lm_head.weight")?c=u:t.lmHeadF16?c=a("lm_head.weight"):c=new T(e,i,o,a("lm_head.weight"),a("lm_head.weight_scale"),(w=n.get("lm_head.input_norm.weight"))!=null?w:d,t.hiddenSize,t.vocabSize);const f=new Me(e,i,o,t,u,l,d,c,h);return f.initDecodeUniforms(s),f}initDecodeUniforms(e){this.decodeTokenBuffer=this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const t=n=>this.device.createBuffer({size:n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});{const n=new ArrayBuffer(12),s=new DataView(n);s.setUint32(0,1,!0),s.setUint32(4,this.config.hiddenSize,!0),s.setUint32(8,this.config.vocabSize,!0),this.decodeEmbeddingUniform=v(this.device,n)}{const n=new ArrayBuffer(12),s=new DataView(n);s.setUint32(0,1,!0),s.setUint32(4,this.config.hiddenSize,!0),s.setFloat32(8,this.config.rmsNormEps,!0),this.decodeFinalNormUniform=v(this.device,n)}if(!(this.lmHead instanceof T)){const n=new ArrayBuffer(12),s=new DataView(n);s.setUint32(0,1,!0),s.setUint32(4,this.config.vocabSize,!0),s.setUint32(8,this.config.hiddenSize,!0),this.decodeLMHeadUniform=v(this.device,n)}this.prefillEmbeddingUniform=t(12),this.prefillFinalNormUniform=t(12),this.lmHead instanceof T||(this.prefillLMHeadUniform=t(12));for(const n of this.layers)n.initDecodeUniforms(e);this.lmHead instanceof T&&this.lmHead.initDecodeUniforms()}forward(e){const t=e.length,n=this.device.createCommandEncoder();let s;if(t===1&&this.decodeTokenBuffer){const d=new ArrayBuffer(4);new DataView(d).setUint32(0,e[0],!0),this.device.queue.writeBuffer(this.decodeTokenBuffer,0,new Uint8Array(d)),s=this.decodeTokenBuffer}else s=this.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Uint32Array(s.getMappedRange()).set(e),s.unmap();let i=this.dispatchEmbedding(n,s,t);for(let d=0;d<this.layers.length;d++){const l=this.layers[d].forward(i,t,this.kvCaches[d],n);this.pool.release(i),i=l,this.kvCaches[d].seqLen+=t}const o=this.dispatchFinalNorm(n,i,t);this.pool.release(i);let a;t>1?(a=this.pool.acquire(this.config.hiddenSize*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),n.copyBufferToBuffer(o,(t-1)*this.config.hiddenSize*4,a,0,this.config.hiddenSize*4),this.pool.release(o)):a=o;let u;if(this.lmHead instanceof T)u=this.lmHead.forward(a,1,n);else{const d=this.lmHead!==this.embedTokens?this.lmHead:void 0;u=this.dispatchLMHead(n,a,1,d)}return t>1?this.pool.release(a):this.pool.release(o),this.device.queue.submit([n.finish()]),u}releaseBuffer(e){this.pool.release(e)}resetKVCache(){for(const e of this.kvCaches)e.seqLen=0;Y(this.bgCache);for(const e of this.layers)e.clearBGCache();this.lmHead instanceof T&&this.lmHead.clearBGCache()}dispose(){Y(this.bgCache);for(const e of this.layers)e.clearBGCache(),e.destroyPreAllocated();this.lmHead instanceof T&&this.lmHead.clearBGCache();for(const e of this.kvCaches)e.key.destroy(),e.value.destroy();this.pool.destroy(),this.pipelines.clear()}diagnose(e){return C(this,null,function*(){const t=e.length,n=[];this.resetKVCache();const s=this.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});new Uint32Array(s.getMappedRange()).set(e),s.unmap();let i=this.device.createCommandEncoder();const o=this.dispatchEmbedding(i,s,t);this.device.queue.submit([i.finish()]),n.push(yield this.readDiag("embedding",o,t*this.config.hiddenSize)),i=this.device.createCommandEncoder();const a=this.layers[0].forward(o,t,this.kvCaches[0],i);this.device.queue.submit([i.finish()]),this.kvCaches[0].seqLen+=t,n.push(yield this.readDiag("layer_0",a,t*this.config.hiddenSize)),this.pool.release(o),i=this.device.createCommandEncoder();const u=this.layers[1].forward(a,t,this.kvCaches[1],i);this.device.queue.submit([i.finish()]),this.kvCaches[1].seqLen+=t,n.push(yield this.readDiag("layer_1",u,t*this.config.hiddenSize)),this.pool.release(a);let d=u;for(let f=2;f<this.layers.length;f++){i=this.device.createCommandEncoder();const m=this.layers[f].forward(d,t,this.kvCaches[f],i);this.device.queue.submit([i.finish()]),this.pool.release(d),d=m,this.kvCaches[f].seqLen+=t}n.push(yield this.readDiag("last_layer",d,t*this.config.hiddenSize)),i=this.device.createCommandEncoder();const l=this.dispatchFinalNorm(i,d,t);this.device.queue.submit([i.finish()]),this.pool.release(d),n.push(yield this.readDiag("final_norm",l,t*this.config.hiddenSize));let h;t>1?(h=this.pool.acquire(this.config.hiddenSize*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),i=this.device.createCommandEncoder(),i.copyBufferToBuffer(l,(t-1)*this.config.hiddenSize*4,h,0,this.config.hiddenSize*4),this.device.queue.submit([i.finish()]),this.pool.release(l)):h=l,n.push(yield this.readDiag("lm_input",h,this.config.hiddenSize)),i=this.device.createCommandEncoder();let c;if(this.lmHead instanceof T)c=this.lmHead.forward(h,1,i);else{const f=this.lmHead!==this.embedTokens?this.lmHead:void 0;c=this.dispatchLMHead(i,h,1,f)}return this.device.queue.submit([i.finish()]),n.push(yield this.readDiag("logits_first100",c,100)),this.pool.release(h===l?l:h),this.pool.release(c),n})}readDiag(e,t,n){return C(this,null,function*(){const s=n*4,i=this.device.createBuffer({size:s,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),o=this.device.createCommandEncoder();o.copyBufferToBuffer(t,0,i,0,s),this.device.queue.submit([o.finish()]),yield i.mapAsync(GPUMapMode.READ);const a=new Float32Array(i.getMappedRange().slice(0));i.unmap(),i.destroy();let u=1/0,d=-1/0,l=0,h=0,c=0,f=0,m=0;for(let _=0;_<a.length;_++){const b=a[_];if(isNaN(b)){c++;continue}if(!isFinite(b)){f++;continue}b===0&&m++,b<u&&(u=b),b>d&&(d=b),l+=b,h+=b*b}const g=l/a.length,w=Math.sqrt(h/a.length);return{name:e,length:a.length,min:u,max:d,mean:g,rms:w,nanCount:c,infCount:f,zeroCount:m,first8:Array.from(a.slice(0,8))}})}dispatchEmbedding(e,t,n){const{pipeline:s,bindGroupLayout:i}=this.pipelines.getOrCreate("embedding",_t),o=n*this.config.hiddenSize*4,a=this.pool.acquire(o,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let u;if(n===1&&this.decodeEmbeddingUniform)u=this.decodeEmbeddingUniform;else if(this.prefillEmbeddingUniform){const f=new ArrayBuffer(12),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,this.config.hiddenSize,!0),m.setUint32(8,this.config.vocabSize,!0),this.device.queue.writeBuffer(this.prefillEmbeddingUniform,0,new Uint8Array(f)),u=this.prefillEmbeddingUniform}else{const f=new ArrayBuffer(12),m=new DataView(f);m.setUint32(0,n,!0),m.setUint32(4,this.config.hiddenSize,!0),m.setUint32(8,this.config.vocabSize,!0),u=v(this.device,f)}const d=[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:this.embedTokens}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:u}}],l=n===1?z(this.bgCache,this.device,"embedding",i,d):this.device.createBindGroup({layout:i,entries:d}),h=n*this.config.hiddenSize,c=e.beginComputePass();return c.setPipeline(s),c.setBindGroup(0,l),c.dispatchWorkgroups(Math.ceil(h/256)),c.end(),a}dispatchFinalNorm(e,t,n){const{pipeline:s,bindGroupLayout:i}=this.pipelines.getOrCreate("rmsnorm",_e),o=this.pool.acquire(n*this.config.hiddenSize*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let a;if(n===1&&this.decodeFinalNormUniform)a=this.decodeFinalNormUniform;else if(this.prefillFinalNormUniform){const h=new ArrayBuffer(12),c=new DataView(h);c.setUint32(0,n,!0),c.setUint32(4,this.config.hiddenSize,!0),c.setFloat32(8,this.config.rmsNormEps,!0),this.device.queue.writeBuffer(this.prefillFinalNormUniform,0,new Uint8Array(h)),a=this.prefillFinalNormUniform}else{const h=new ArrayBuffer(12),c=new DataView(h);c.setUint32(0,n,!0),c.setUint32(4,this.config.hiddenSize,!0),c.setFloat32(8,this.config.rmsNormEps,!0),a=v(this.device,h)}const u=[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:this.finalNorm}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:a}}],d=n===1?z(this.bgCache,this.device,"finalNorm",i,u):this.device.createBindGroup({layout:i,entries:u}),l=e.beginComputePass();return l.setPipeline(s),l.setBindGroup(0,d),l.dispatchWorkgroups(n),l.end(),o}dispatchLMHead(e,t,n,s){const i=this.config.vocabSize,o=this.config.hiddenSize,{pipeline:a,bindGroupLayout:u}=this.pipelines.getOrCreate("f32_matmul",bt),d=this.pool.acquire(n*i*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);let l;if(n===1&&this.decodeLMHeadUniform)l=this.decodeLMHeadUniform;else if(this.prefillLMHeadUniform){const b=new ArrayBuffer(12),S=new DataView(b);S.setUint32(0,n,!0),S.setUint32(4,i,!0),S.setUint32(8,o,!0),this.device.queue.writeBuffer(this.prefillLMHeadUniform,0,new Uint8Array(b)),l=this.prefillLMHeadUniform}else{const b=new ArrayBuffer(12),S=new DataView(b);S.setUint32(0,n,!0),S.setUint32(4,i,!0),S.setUint32(8,o,!0),l=v(this.device,b)}const h=s!=null?s:this.embedTokens,c=[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:h}},{binding:2,resource:{buffer:d}},{binding:3,resource:{buffer:l}}],f=n===1?z(this.bgCache,this.device,"lmHead",u,c):this.device.createBindGroup({layout:u,entries:c}),m=n*i,g=Math.min(m,65535),w=Math.ceil(m/65535),_=e.beginComputePass();return _.setPipeline(a),_.setBindGroup(0,f),_.dispatchWorkgroups(g,w),_.end(),d}},vt=(G=class{constructor(e,t,n){p(this,"config");p(this,"vocab");p(this,"reverseVocab");p(this,"merges");p(this,"mergeRanks");p(this,"bosId");p(this,"eosId");p(this,"textEncoder",new TextEncoder);p(this,"textDecoder",new TextDecoder("utf-8",{fatal:!1}));var s,i;this.config=e,this.vocab=t,this.merges=n,this.bosId=(s=e.bosToken)!=null?s:1,this.eosId=(i=e.eosToken)!=null?i:2,this.reverseVocab=new Map;for(const[o,a]of t)this.reverseVocab.set(a,o);this.mergeRanks=new Map;for(let o=0;o<n.length;o++)this.mergeRanks.set(`${n[o][0]} ${n[o][1]}`,o)}static fromGGUFMetadata(e){var l,h,c;const t=e["tokenizer.ggml.tokens"],n=e["tokenizer.ggml.merges"],s=(l=e["tokenizer.ggml.model"])!=null?l:"gpt2",i=new Map;for(let f=0;f<t.length;f++)i.set(t[f],f);const o=[];if(n)for(const f of n){const m=f.split(" ");m.length===2&&o.push([m[0],m[1]])}const a=(h=e["tokenizer.ggml.bos_token_id"])!=null?h:1,u=(c=e["tokenizer.ggml.eos_token_id"])!=null?c:2,d={type:s==="gpt2"?"bpe":"sentencepiece",vocabSize:t.length,bosToken:a,eosToken:u};return new G(d,i,o)}static fromJSON(e){var i,o,a,u,d,l;const t=new Map(Object.entries(e.vocab)),n=e.merges.map(h=>{const c=h.split(" ");return[c[0],c[1]]}),s={type:(o=(i=e.config)==null?void 0:i.type)!=null?o:"bpe",vocabSize:t.size,bosToken:(u=(a=e.config)==null?void 0:a.bosToken)!=null?u:1,eosToken:(l=(d=e.config)==null?void 0:d.eosToken)!=null?l:2};return new G(s,t,n)}encode(e,t=!0){var o;const n=[];t&&n.push(this.bosId),this.config.type==="sentencepiece"&&(e=" "+e);const s=new RegExp("(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+","gu"),i=(o=e.match(s))!=null?o:[e];for(const a of i){const u=this.bpeEncode(a);n.push(...u)}return new Uint32Array(n)}decode(e){const t=[];for(const n of e){if(n===this.bosId||n===this.eosId)continue;const s=this.reverseVocab.get(n);s!==void 0&&t.push(this.decodeToken(s))}return t.join("")}decodeToken(e){if(e.startsWith("<0x")&&e.endsWith(">")){const t=parseInt(e.slice(3,-1),16);return String.fromCharCode(t)}return this.config.type==="sentencepiece"?e.replace(/▁/g," "):this.bytesToString(e)}get eosTokenId(){return this.eosId}get bosTokenId(){return this.bosId}get eotTokenId(){return this.vocab.get("<|eot_id|>")}get imEndTokenId(){return this.vocab.get("<|im_end|>")}applyChatTemplate(e){const t=this.vocab.get("<|im_start|>"),n=this.vocab.get("<|im_end|>");if(t!==void 0&&n!==void 0)return this.applyChatML(e,t,n);const s=this.vocab.get("<|start_header_id|>"),i=this.vocab.get("<|end_header_id|>"),o=this.vocab.get("<|eot_id|>");if(s===void 0||i===void 0||o===void 0){console.warn("[0xBitNet] Chat template fallback: special tokens missing");const u=e.map(d=>d.content).join(`
`);return this.encode(u)}console.debug(`[0xBitNet] Chat template: LLaMA 3 (start_header=${s}, end_header=${i}, eot=${o})`);const a=[this.bosId];for(const u of e)a.push(s),a.push(...this.encode(u.role,!1)),a.push(i),a.push(...this.encode(`

`+u.content,!1)),a.push(o);return a.push(s),a.push(...this.encode("assistant",!1)),a.push(i),a.push(...this.encode(`

`,!1)),new Uint32Array(a)}applyChatML(e,t,n){console.debug(`[0xBitNet] Chat template: ChatML (im_start=${t}, im_end=${n})`);const s=[this.bosId];for(const i of e)s.push(t),s.push(...this.encode(i.role+`
`+i.content,!1)),s.push(n),s.push(...this.encode(`
`,!1));return s.push(t),s.push(...this.encode(`assistant
`,!1)),new Uint32Array(s)}bpeEncode(e){if(e.length===0)return[];let t;for(this.config.type==="sentencepiece"?t=[...e].map(s=>s.replace(" ","▁")):t=this.stringToBytes(e);t.length>1;){let s=1/0,i=-1;for(let a=0;a<t.length-1;a++){const u=`${t[a]} ${t[a+1]}`,d=this.mergeRanks.get(u);d!==void 0&&d<s&&(s=d,i=a)}if(i===-1)break;const o=t[i]+t[i+1];t.splice(i,2,o)}const n=[];for(const s of t){const i=this.vocab.get(s);if(i!==void 0)n.push(i);else for(const o of this.textEncoder.encode(s)){const a=`<0x${o.toString(16).toUpperCase().padStart(2,"0")}>`,u=this.vocab.get(a);u!==void 0&&n.push(u)}}return n}static getByteToUnicode(){if(G.byteToUnicode)return G.byteToUnicode;const e=new Map,t=[[33,126],[161,172],[174,255]],n=[];for(const[o,a]of t)for(let u=o;u<=a;u++)n.push(u);const s=[...n];let i=0;for(let o=0;o<256;o++)n.includes(o)||(n.push(o),s.push(256+i),i++);for(let o=0;o<n.length;o++)e.set(n[o],String.fromCharCode(s[o]));return G.byteToUnicode=e,e}static getUnicodeToByte(){if(G.unicodeToByte)return G.unicodeToByte;const e=G.getByteToUnicode(),t=new Map;for(const[n,s]of e)t.set(s,n);return G.unicodeToByte=t,t}stringToBytes(e){var i;const t=G.getByteToUnicode(),n=this.textEncoder.encode(e),s=[];for(const o of n)s.push((i=t.get(o))!=null?i:String.fromCharCode(o));return s}bytesToString(e){const t=G.getUnicodeToByte(),n=[];for(const s of e){const i=t.get(s);i!==void 0?n.push(i):n.push(s.charCodeAt(0))}return this.textDecoder.decode(new Uint8Array(n))}},p(G,"byteToUnicode",null),p(G,"unicodeToByte",null),G);const k=r=>document.querySelector(r),fe=k("#status-text"),xe=k("#status-bar"),Pe=k("#progress-fill"),Ut=k("#load-section"),re=k("#load-btn"),Ge=k("#model-url"),be=k("#cached-models"),yt=k("#app"),Bt=k("#prompt-input"),se=k("#visualize-btn"),le=k("#output"),ae=k("#heatmap-canvas");k("#heatmap-container");const oe=k("#token-select"),De=k("#downsample-select"),Ce=k("#sort-select"),Te=k("#contrast-toggle"),kt=k("#stats-content"),St=k("#layer-labels"),xt=k("#model-size-badge"),we=k("#error-banner");let N=null,Q=null,P=null,ue=null,W=[],Ee=0;const Pt="https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf";function Gt(r){console.error(r),we.textContent=r,we.style.display="block",xe.style.borderBottom="1px solid #552222",fe.textContent="Error",fe.style.color="#f87171"}function Dt(){we.style.display="none",xe.style.borderBottom="1px solid #222",fe.style.color="#888"}function Ct(r){try{const e=r.replace("https://huggingface.co/","").split("/");return e[0]+"/"+e[1]}catch(e){return r.split("/").pop()||r}}function Z(r,e){fe.textContent=r;const t=Math.max(0,Math.min(100,e*100));Pe.style.width=`${t}%`,Pe.style.background=e>=1?"#4ade80":"#818cf8"}re.onclick=()=>C(null,null,function*(){Dt();const r=Ge.value.trim()||Pt;re.disabled=!0,re.textContent="Loading…",Z("Initializing WebGPU…",.05),console.log("[load] starting, url:",r);try{if(typeof navigator=="undefined"||!("gpu"in navigator))throw new Error("WebGPU not available. Use Chrome 113+ or Edge 113+.");const e=yield tt();P=e.device,console.log("[load] GPU ready"),Z("Downloading model…",.1);const t=yield nt(r,e.device,n=>{const s=n.phase==="download"?.1+n.fraction*.6:n.phase==="parse"?.7+n.fraction*.1:.8+n.fraction*.15;Z(`${n.phase}  ${(n.fraction*100).toFixed(0)}%`,s)});console.log("[load] model loaded from GGUF"),Z("Building model…",.95),N=wt.build(e.device,t.config,t.weights),ue={hiddenSize:t.config.hiddenSize,numLayers:t.config.numHiddenLayers,vocabSize:t.config.vocabSize},console.log("[load] model built:",ue),Z("Loading tokenizer…",.98),Q=vt.fromGGUFMetadata(t.metadata),xt.textContent=`${t.config.numHiddenLayers}L · ${t.config.hiddenSize}D · v${t.config.vocabSize}`,Z("Ready — model loaded",1),Ut.style.display="none",yt.style.display="flex",ve()}catch(e){console.error("[load] FAILED:",e),Gt(`${e.message||e}`),re.disabled=!1,re.textContent="Retry Load Model"}});function X(r,e,t){return C(this,null,function*(){const n=t*4,s=r.createBuffer({size:n,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),i=r.createCommandEncoder();i.copyBufferToBuffer(e,0,s,0,n),r.queue.submit([i.finish()]),yield s.mapAsync(GPUMapMode.READ);const o=new Float32Array(s.getMappedRange().slice(0));return s.unmap(),s.destroy(),o})}function Tt(r,e,t){return C(this,null,function*(){const n=r,s=r.config.hiddenSize,i=r.config.numHiddenLayers,o=1;let a=t.createCommandEncoder();const u=n.dispatchEmbedding(a,e,o);t.queue.submit([a.finish()]);const d=[{name:"embed",data:yield X(t,u,s)}];let l=u;for(let f=0;f<i;f++){a=t.createCommandEncoder();const m=n.layers[f].forward(l,o,n.kvCaches[f],a);n.pool.release(l),t.queue.submit([a.finish()]),n.kvCaches[f].seqLen+=o,d.push({name:`L${f}`,data:yield X(t,m,s)}),l=m}a=t.createCommandEncoder();const h=n.dispatchFinalNorm(a,l,o);t.queue.submit([a.finish()]),n.pool.release(l),d.push({name:"norm",data:yield X(t,h,s)}),a=t.createCommandEncoder();const c=n.dispatchLMHead(a,h,o);return t.queue.submit([a.finish()]),{acts:d,logits:c}})}function Et(r){const e=n=>Math.max(0,Math.min(255,Math.round(n))),t=(r+1)/2;if(t<.5){const n=t*2;return[e(120*(1-n)),e(30+120*n),200]}else{const n=(t-.5)*2;return[e(50*n),e(150+80*n),e(200-120*n)]}}function J(){const r=W;if(!r.length||!ue)return;const e=ue.hiddenSize,t=r.length,n=r[0].data.length/e,s=parseInt(De.value),i=Ce.value,o=Math.ceil(e/s),a=Math.min(parseInt(oe.value),n-1);let u=r.map(U=>{const D=[];for(let x=0;x<o;x++){const y=x*s,B=Math.min(y+s,e);let $=0;for(let F=y;F<B;F++)$+=U.data[a*e+F];D.push($/(B-y))}return{name:U.name,values:D}});i==="norm"&&u.sort((U,D)=>{let x=0,y=0;for(const B of U.values)x+=B*B;for(const B of D.values)y+=B*B;return Math.sqrt(y)-Math.sqrt(x)});const d=300,l=52,h=36,c=100,f=60,m=32,g=window.innerWidth-d-m,w=window.innerHeight-l-h-c-f-m;let _,b;g/w>16/9?(b=Math.max(100,w),_=b*16/9):(_=Math.max(100,g),b=_*9/16);const S=Math.max(1,Math.floor(_/o)),E=Math.max(1,Math.floor(b/t)),A=o*S,R=t*E,M=window.devicePixelRatio||1;ae.width=A*M,ae.height=R*M,ae.style.width=A+"px",ae.style.height=R+"px";const I=ae.getContext("2d");I.scale(M,M);for(let U=0;U<t;U++){const D=u[U].values;let x=0;for(const y of D)Math.abs(y)>x&&(x=Math.abs(y));x===0&&(x=1);for(let y=0;y<o;y++){const B=D[y]/x,$=Te.checked?Math.sign(B)*Math.sqrt(Math.abs(B)):B,[F,K,Mt]=Et($);I.fillStyle=`rgb(${F},${K},${Mt})`,I.fillRect(y*S,U*E,S,E)}}St.textContent=u.map((U,D)=>{const x=D===0?" [start]":D===u.length-1?" [output]":"";return`${U.name}${x}  ‖x‖=${Math.sqrt(U.values.reduce((y,B)=>y+B*B,0)).toFixed(2)}`}).join(`
`);const ee=r.map(U=>{const D=a*e,x=new Float32Array(U.data.buffer,D*4,e);let y=1/0,B=-1/0,$=0;for(let F=0;F<e;F++){const K=x[F];K<y&&(y=K),K>B&&(B=K),$+=Math.abs(K)}return{name:U.name,min:y,max:B,meanAbs:$/e}});kt.innerHTML=ee.map(U=>`<div>${U.name}: [${U.min.toFixed(2)}, ${U.max.toFixed(2)}] μ|·|=${U.meanAbs.toFixed(4)}</div>`).join("")}function At(r,e,t,n){return C(this,null,function*(){const s=e.config.vocabSize,i=t.createBuffer({size:s*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),o=t.createCommandEncoder();o.copyBufferToBuffer(r,0,i,0,s*4),t.queue.submit([o.finish()]),yield i.mapAsync(GPUMapMode.READ);const a=new Float32Array(i.getMappedRange().slice(0));i.unmap(),i.destroy();const u=.8,d=40,l=1.1;if(n.length)for(const g of n)g>=0&&g<s&&(a[g]=a[g]>0?a[g]/l:a[g]*l);{const g=1/u;for(let w=0;w<s;w++)a[w]*=g}if(d<s){const g=new Uint32Array(d);for(let _=0;_<d;_++)g[_]=_;for(let _=(d>>1)-1;_>=0;_--)Ae(g,_,d,a);for(let _=d;_<s;_++)a[_]>a[g[0]]&&(g[0]=_,Ae(g,0,d,a));const w=a[g[0]];for(let _=0;_<s;_++)a[_]<w&&(a[_]=-1/0)}let h=-1/0;for(let g=0;g<s;g++)a[g]>h&&(h=a[g]);let c=0;for(let g=0;g<s;g++)a[g]=Math.exp(a[g]-h),c+=a[g];const f=Math.random()*c;let m=0;for(let g=0;g<s;g++)if(m+=a[g],m>=f)return g;return s-1})}function Ae(r,e,t,n){for(;;){let s=e;const i=2*e+1,o=2*e+2;if(i<t&&n[r[i]]<n[r[s]]&&(s=i),o<t&&n[r[o]]<n[r[s]]&&(s=o),s===e)break;const a=r[e];r[e]=r[s],r[s]=a,e=s}}se.onclick=()=>C(null,null,function*(){if(!N||!P||!Q||!ue)return;const r=Bt.value.trim();if(r){se.disabled=!0,se.textContent="Running…",le.textContent="Capturing activations…";try{const e=N.config.hiddenSize,t=N.config.numHiddenLayers,n=N,s=Q.encode(r);if(s.length===0)throw new Error("Empty tokenization");const i=s.length;N.resetKVCache();const o=P.createBuffer({size:s.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});new Uint32Array(o.getMappedRange()).set(s),o.unmap();let a=P.createCommandEncoder();const u=n.dispatchEmbedding(a,o,i);P.queue.submit([a.finish()]),W=[{name:"embed",data:yield X(P,u,i*e)}];let d=u;for(let m=0;m<t;m++){a=P.createCommandEncoder();const g=n.layers[m].forward(d,i,n.kvCaches[m],a);n.pool.release(d),P.queue.submit([a.finish()]),n.kvCaches[m].seqLen+=i,W.push({name:`L${m}`,data:yield X(P,g,i*e)}),d=g}a=P.createCommandEncoder();const l=n.dispatchFinalNorm(a,d,i);P.queue.submit([a.finish()]),n.pool.release(d),W.push({name:"norm",data:yield X(P,l,i*e)}),a=P.createCommandEncoder();let h=n.dispatchLMHead(a,l,i);P.queue.submit([a.finish()]),oe.innerHTML="";for(let m=0;m<i;m++){const g=document.createElement("option");g.value=String(m),g.textContent=m===i-1?`Token ${m} (last)`:`Token ${m}`,m===i-1&&(g.selected=!0),oe.appendChild(g)}Ee=i-1,le.textContent="",J();const c=P.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),f=[];for(let m=0;m<64;m++){const g=yield At(h,N,P,f);if(N.releaseBuffer(h),g===Q.eosTokenId||g===Q.eotTokenId)break;f.push(g);const w=Q.decode([g]);le.textContent+=w;const _=P.createBuffer({size:4,usage:GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});new Uint32Array(_.getMappedRange()).set([g]),_.unmap(),a=P.createCommandEncoder(),a.copyBufferToBuffer(_,0,c,0,4),P.queue.submit([a.finish()]),_.destroy();const b=yield Tt(N,c,P);h=b.logits;for(let A=0;A<W.length;A++){const R=W[A].data,M=new Float32Array(R.length+e);M.set(R),M.set(b.acts[A].data,R.length),W[A].data=M}const S=W[0].data.length/e,E=document.createElement("option");E.value=String(S-1),E.textContent=`Token ${S-1} (gen)`,E.selected=!0,oe.appendChild(E),Ee=S-1,J()}c.destroy()}catch(e){le.textContent=`Error: ${e.message||e}`,console.error("[pipeline]",e)}finally{se.disabled=!1,se.textContent="Visualize Activations"}}});function ve(){return C(this,null,function*(){try{const r=yield pt();if(be.innerHTML="",r.length===0)return;const e=document.createElement("div");e.style.cssText="font-size:0.75rem;color:#666;margin-top:0.5rem",e.textContent="Cached models:",be.appendChild(e);for(const t of r){const n=Ct(t),s=document.createElement("div");s.style.cssText="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0.5rem;border-radius:6px;background:#111;border:1px solid #333";const i=document.createElement("span");i.style.cssText="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:0.8rem;cursor:pointer",i.textContent=n,i.title=t,i.onclick=()=>{Ge.value=t};const o=document.createElement("button");o.textContent="×",o.style.cssText="padding:0.1rem 0.4rem;border-radius:4px;border:1px solid #552222;background:#1a0a0a;color:#f87171;font-size:0.7rem;cursor:pointer",o.onclick=a=>C(null,null,function*(){a.stopPropagation(),yield mt(t),ve()}),s.append(i,o),be.appendChild(s)}}catch(r){console.warn("refreshCachedModels:",r)}})}oe.onchange=()=>{J()},De.onchange=()=>{J()},Ce.onchange=()=>{J()},Te.onchange=()=>{J()},ve()})();
