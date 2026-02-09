
const mindgrab_128MB = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  // WebGPU requires buffer size to be multiple of 4 when mappedAtCreation is true
  const paddedSize = Math.ceil(size / 4) * 4;
  const buf = device.createBuffer({ size: paddedSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_2_256_32_4_8_16_4_4_3_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_134217728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_16777216:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_405:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx1+bitcast<i32>((cast1<<4u)));
  var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>((gidx0>>2u))<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast1<<6u)));
  var alu2 = (gidx0<120);
  var alu3 = (alu0<60);
  var alu4 = (3<alu0);
  var alu5 = (7<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 3; Ridx0++) {
    var cast2 = bitcast<u32>(Ridx0);
    var alu22 = (gidx1+bitcast<i32>((cast2<<4u)));
    var alu23 = (alu1+cast0+bitcast<i32>((cast2<<20u)));
    var alu24 = ((15<alu22)&(alu22<272));
    var alu25 = (alu4&alu5&alu24);
    var val0 = select(0.0f, data1_16777216[(alu23+-1052688)], alu25);
    var alu26 = ((gidx2*108)+(Ridx0*9));
    var val1 = data2_405[(alu26+1)];
    var val2 = data2_405[(alu26+3)];
    var val3 = data2_405[alu26];
    var alu27 = (alu5&alu24);
    var val4 = select(0.0f, data1_16777216[(alu23+-1052672)], alu27);
    var alu28 = (alu3&alu5&alu24);
    var val5 = select(0.0f, data1_16777216[(alu23+-1052656)], alu28);
    var val6 = data2_405[(alu26+2)];
    var alu29 = (alu4&alu24);
    var val7 = select(0.0f, data1_16777216[(alu23+-1048592)], alu29);
    var val8 = select(0.0f, data1_16777216[(alu23+-1048576)], alu24);
    var val9 = data2_405[(alu26+4)];
    var alu30 = (alu3&alu24);
    var val10 = select(0.0f, data1_16777216[(alu23+-1048560)], alu30);
    var val11 = data2_405[(alu26+5)];
    var alu31 = (alu4&alu2&alu24);
    var val12 = select(0.0f, data1_16777216[(alu23+-1044496)], alu31);
    var val13 = data2_405[(alu26+6)];
    var alu32 = (alu2&alu24);
    var val14 = select(0.0f, data1_16777216[(alu23+-1044480)], alu32);
    var val15 = data2_405[(alu26+7)];
    var alu33 = (alu3&alu2&alu24);
    var val16 = select(0.0f, data1_16777216[(alu23+-1044464)], alu33);
    var val17 = data2_405[(alu26+8)];
    var val18 = data2_405[(alu26+27)];
    var val19 = data2_405[(alu26+28)];
    var val20 = data2_405[(alu26+29)];
    var val21 = data2_405[(alu26+30)];
    var val22 = data2_405[(alu26+31)];
    var val23 = data2_405[(alu26+32)];
    var val24 = data2_405[(alu26+33)];
    var val25 = data2_405[(alu26+34)];
    var val26 = data2_405[(alu26+35)];
    var val27 = data2_405[(alu26+54)];
    var val28 = data2_405[(alu26+55)];
    var val29 = data2_405[(alu26+56)];
    var val30 = data2_405[(alu26+57)];
    var val31 = data2_405[(alu26+58)];
    var val32 = data2_405[(alu26+59)];
    var val33 = data2_405[(alu26+60)];
    var val34 = data2_405[(alu26+61)];
    var val35 = data2_405[(alu26+62)];
    var val36 = data2_405[(alu26+81)];
    var val37 = data2_405[(alu26+82)];
    var val38 = data2_405[(alu26+83)];
    var val39 = data2_405[(alu26+84)];
    var val40 = data2_405[(alu26+85)];
    var val41 = data2_405[(alu26+86)];
    var val42 = data2_405[(alu26+87)];
    var val43 = data2_405[(alu26+88)];
    var val44 = data2_405[(alu26+89)];
    var val45 = select(0.0f, data1_16777216[(alu23+-1052687)], alu25);
    var val46 = select(0.0f, data1_16777216[(alu23+-1052671)], alu27);
    var val47 = select(0.0f, data1_16777216[(alu23+-1052655)], alu28);
    var val48 = select(0.0f, data1_16777216[(alu23+-1048591)], alu29);
    var val49 = select(0.0f, data1_16777216[(alu23+-1048575)], alu24);
    var val50 = select(0.0f, data1_16777216[(alu23+-1048559)], alu30);
    var val51 = select(0.0f, data1_16777216[(alu23+-1044495)], alu31);
    var val52 = select(0.0f, data1_16777216[(alu23+-1044479)], alu32);
    var val53 = select(0.0f, data1_16777216[(alu23+-1044463)], alu33);
    var val54 = select(0.0f, data1_16777216[(alu23+-1052686)], alu25);
    var val55 = select(0.0f, data1_16777216[(alu23+-1048590)], alu29);
    var val56 = select(0.0f, data1_16777216[(alu23+-1048574)], alu24);
    var val57 = select(0.0f, data1_16777216[(alu23+-1048558)], alu30);
    var val58 = select(0.0f, data1_16777216[(alu23+-1044494)], alu31);
    var val59 = select(0.0f, data1_16777216[(alu23+-1044478)], alu32);
    var val60 = select(0.0f, data1_16777216[(alu23+-1044462)], alu33);
    var val61 = select(0.0f, data1_16777216[(alu23+-1052685)], alu25);
    var val62 = select(0.0f, data1_16777216[(alu23+-1052670)], alu27);
    var val63 = select(0.0f, data1_16777216[(alu23+-1052669)], alu27);
    var val64 = select(0.0f, data1_16777216[(alu23+-1052654)], alu28);
    var val65 = select(0.0f, data1_16777216[(alu23+-1052653)], alu28);
    var val66 = select(0.0f, data1_16777216[(alu23+-1048589)], alu29);
    var val67 = select(0.0f, data1_16777216[(alu23+-1048573)], alu24);
    var val68 = select(0.0f, data1_16777216[(alu23+-1048557)], alu30);
    var val69 = select(0.0f, data1_16777216[(alu23+-1044493)], alu31);
    var val70 = select(0.0f, data1_16777216[(alu23+-1044477)], alu32);
    var val71 = select(0.0f, data1_16777216[(alu23+-1044461)], alu33);
    acc0[0] = (acc0[0]+(val0*val3)+(val4*val1)+(val5*val6)+(val7*val2)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
    acc0[1] = (acc0[1]+(val0*val18)+(val4*val19)+(val5*val20)+(val7*val21)+(val8*val22)+(val10*val23)+(val12*val24)+(val14*val25)+(val16*val26));
    acc0[2] = (acc0[2]+(val0*val27)+(val4*val28)+(val5*val29)+(val7*val30)+(val8*val31)+(val10*val32)+(val12*val33)+(val14*val34)+(val16*val35));
    acc0[3] = (acc0[3]+(val0*val36)+(val4*val37)+(val5*val38)+(val7*val39)+(val8*val40)+(val10*val41)+(val12*val42)+(val14*val43)+(val16*val44));
    acc0[4] = (acc0[4]+(val45*val3)+(val46*val1)+(val47*val6)+(val48*val2)+(val49*val9)+(val50*val11)+(val51*val13)+(val52*val15)+(val53*val17));
    acc0[5] = (acc0[5]+(val45*val18)+(val46*val19)+(val47*val20)+(val48*val21)+(val49*val22)+(val50*val23)+(val51*val24)+(val52*val25)+(val53*val26));
    acc0[6] = (acc0[6]+(val45*val27)+(val46*val28)+(val47*val29)+(val48*val30)+(val49*val31)+(val50*val32)+(val51*val33)+(val52*val34)+(val53*val35));
    acc0[7] = (acc0[7]+(val45*val36)+(val46*val37)+(val47*val38)+(val48*val39)+(val49*val40)+(val50*val41)+(val51*val42)+(val52*val43)+(val53*val44));
    acc0[8] = (acc0[8]+(val54*val3)+(val62*val1)+(val64*val6)+(val55*val2)+(val56*val9)+(val57*val11)+(val58*val13)+(val59*val15)+(val60*val17));
    acc0[9] = (acc0[9]+(val54*val18)+(val62*val19)+(val64*val20)+(val55*val21)+(val56*val22)+(val57*val23)+(val58*val24)+(val59*val25)+(val60*val26));
    acc0[10] = (acc0[10]+(val54*val27)+(val62*val28)+(val64*val29)+(val55*val30)+(val56*val31)+(val57*val32)+(val58*val33)+(val59*val34)+(val60*val35));
    acc0[11] = (acc0[11]+(val54*val36)+(val62*val37)+(val64*val38)+(val55*val39)+(val56*val40)+(val57*val41)+(val58*val42)+(val59*val43)+(val60*val44));
    acc0[12] = (acc0[12]+(val61*val3)+(val63*val1)+(val65*val6)+(val66*val2)+(val67*val9)+(val68*val11)+(val69*val13)+(val70*val15)+(val71*val17));
    acc0[13] = (acc0[13]+(val61*val18)+(val63*val19)+(val65*val20)+(val66*val21)+(val67*val22)+(val68*val23)+(val69*val24)+(val70*val25)+(val71*val26));
    acc0[14] = (acc0[14]+(val61*val27)+(val63*val28)+(val65*val29)+(val66*val30)+(val67*val31)+(val68*val32)+(val69*val33)+(val70*val34)+(val71*val35));
    acc0[15] = (acc0[15]+(val61*val36)+(val63*val37)+(val65*val38)+(val66*val39)+(val67*val40)+(val68*val41)+(val69*val42)+(val70*val43)+(val71*val44));
  }
  var alu51 = (alu1+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<26u)));
  data0_134217728[alu51] = acc0[0];
  data0_134217728[(alu51+1)] = acc0[4];
  data0_134217728[(alu51+2)] = acc0[8];
  data0_134217728[(alu51+3)] = acc0[12];
  data0_134217728[(alu51+16777216)] = acc0[1];
  data0_134217728[(alu51+16777217)] = acc0[5];
  data0_134217728[(alu51+16777218)] = acc0[9];
  data0_134217728[(alu51+16777219)] = acc0[13];
  data0_134217728[(alu51+33554432)] = acc0[2];
  data0_134217728[(alu51+33554433)] = acc0[6];
  data0_134217728[(alu51+33554434)] = acc0[10];
  data0_134217728[(alu51+33554435)] = acc0[14];
  data0_134217728[(alu51+50331648)] = acc0[3];
  data0_134217728[(alu51+50331649)] = acc0[7];
  data0_134217728[(alu51+50331650)] = acc0[11];
  data0_134217728[(alu51+50331651)] = acc0[15];
}`;

const r_7_256_32_4_8_16_4_3_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_117440512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_16777216:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_405:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx1+bitcast<i32>((cast1<<4u)));
  var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>((gidx0>>2u))<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast1<<6u)));
  var alu2 = (gidx0<120);
  var alu3 = (alu0<60);
  var alu4 = (3<alu0);
  var alu5 = (7<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 3; Ridx0++) {
    var cast2 = bitcast<u32>(Ridx0);
    var alu10 = (gidx1+bitcast<i32>((cast2<<4u)));
    var alu11 = (alu1+cast0+bitcast<i32>((cast2<<20u)));
    var alu12 = ((15<alu10)&(alu10<272));
    var alu13 = (alu4&alu5&alu12);
    var val0 = select(0.0f, data1_16777216[(alu11+-1052688)], alu13);
    var alu14 = ((gidx2*27)+(Ridx0*9));
    var val1 = data2_405[(alu14+216)];
    var alu15 = (alu5&alu12);
    var val2 = select(0.0f, data1_16777216[(alu11+-1052672)], alu15);
    var val3 = data2_405[(alu14+217)];
    var alu16 = (alu3&alu5&alu12);
    var val4 = select(0.0f, data1_16777216[(alu11+-1052656)], alu16);
    var val5 = data2_405[(alu14+218)];
    var alu17 = (alu4&alu12);
    var val6 = select(0.0f, data1_16777216[(alu11+-1048592)], alu17);
    var val7 = data2_405[(alu14+219)];
    var val8 = select(0.0f, data1_16777216[(alu11+-1048576)], alu12);
    var val9 = data2_405[(alu14+220)];
    var alu18 = (alu3&alu12);
    var val10 = select(0.0f, data1_16777216[(alu11+-1048560)], alu18);
    var val11 = data2_405[(alu14+221)];
    var alu19 = (alu4&alu2&alu12);
    var val12 = select(0.0f, data1_16777216[(alu11+-1044496)], alu19);
    var val13 = data2_405[(alu14+222)];
    var alu20 = (alu2&alu12);
    var val14 = select(0.0f, data1_16777216[(alu11+-1044480)], alu20);
    var val15 = data2_405[(alu14+223)];
    var alu21 = (alu3&alu2&alu12);
    var val16 = select(0.0f, data1_16777216[(alu11+-1044464)], alu21);
    var val17 = data2_405[(alu14+224)];
    var val18 = select(0.0f, data1_16777216[(alu11+-1052687)], alu13);
    var val19 = select(0.0f, data1_16777216[(alu11+-1044463)], alu21);
    var val20 = select(0.0f, data1_16777216[(alu11+-1052686)], alu13);
    var val21 = select(0.0f, data1_16777216[(alu11+-1048557)], alu18);
    var val22 = select(0.0f, data1_16777216[(alu11+-1044493)], alu19);
    var val23 = select(0.0f, data1_16777216[(alu11+-1044478)], alu20);
    var val24 = select(0.0f, data1_16777216[(alu11+-1044477)], alu20);
    var val25 = select(0.0f, data1_16777216[(alu11+-1044462)], alu21);
    var val26 = select(0.0f, data1_16777216[(alu11+-1052685)], alu13);
    var val27 = select(0.0f, data1_16777216[(alu11+-1052671)], alu15);
    var val28 = select(0.0f, data1_16777216[(alu11+-1052670)], alu15);
    var val29 = select(0.0f, data1_16777216[(alu11+-1052669)], alu15);
    var val30 = select(0.0f, data1_16777216[(alu11+-1052655)], alu16);
    var val31 = select(0.0f, data1_16777216[(alu11+-1052654)], alu16);
    var val32 = select(0.0f, data1_16777216[(alu11+-1052653)], alu16);
    var val33 = select(0.0f, data1_16777216[(alu11+-1048591)], alu17);
    var val34 = select(0.0f, data1_16777216[(alu11+-1048590)], alu17);
    var val35 = select(0.0f, data1_16777216[(alu11+-1048589)], alu17);
    var val36 = select(0.0f, data1_16777216[(alu11+-1048575)], alu12);
    var val37 = select(0.0f, data1_16777216[(alu11+-1048574)], alu12);
    var val38 = select(0.0f, data1_16777216[(alu11+-1048573)], alu12);
    var val39 = select(0.0f, data1_16777216[(alu11+-1048559)], alu18);
    var val40 = select(0.0f, data1_16777216[(alu11+-1048558)], alu18);
    var val41 = select(0.0f, data1_16777216[(alu11+-1044495)], alu19);
    var val42 = select(0.0f, data1_16777216[(alu11+-1044494)], alu19);
    var val43 = select(0.0f, data1_16777216[(alu11+-1044479)], alu20);
    var val44 = select(0.0f, data1_16777216[(alu11+-1044461)], alu21);
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
    acc0[1] = (acc0[1]+(val18*val1)+(val27*val3)+(val30*val5)+(val33*val7)+(val36*val9)+(val39*val11)+(val41*val13)+(val43*val15)+(val19*val17));
    acc0[2] = (acc0[2]+(val20*val1)+(val28*val3)+(val31*val5)+(val34*val7)+(val37*val9)+(val40*val11)+(val42*val13)+(val23*val15)+(val25*val17));
    acc0[3] = (acc0[3]+(val26*val1)+(val29*val3)+(val32*val5)+(val35*val7)+(val38*val9)+(val21*val11)+(val22*val13)+(val24*val15)+(val44*val17));
  }
  var alu27 = (alu1+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<24u)));
  data0_117440512[alu27] = acc0[0];
  data0_117440512[(alu27+1)] = acc0[1];
  data0_117440512[(alu27+2)] = acc0[2];
  data0_117440512[(alu27+3)] = acc0[3];
}`;

const r_10240_32_3_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_983040:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_134217728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_117440512:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 10240 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)));
  var alu1 = ((gidx0*96)+(lidx0*3));
  var alu2 = (alu0<174762);
  var alu3 = (alu1<524287);
  var alu4 = (alu1<524288);
  var alu5 = (174761<alu0);
  var alu6 = (524286<alu1);
  var alu7 = (524287<alu1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu11 = ((gidx0*24576)+(lidx0*768)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = select(0.0f, data1_134217728[(alu11+1)], alu4);
    var val1 = select(0.0f, data1_134217728[alu11], alu4);
    var val2 = select(0.0f, data2_117440512[(alu11+-134217728)], alu7);
    var val3 = select(0.0f, data2_117440512[(alu11+-134217727)], alu7);
    var val4 = select(0.0f, data1_134217728[(alu11+2)], alu4);
    var val5 = select(0.0f, data2_117440512[(alu11+-134217726)], alu7);
    var val6 = select(0.0f, data1_134217728[(alu11+3)], alu4);
    var val7 = select(0.0f, data2_117440512[(alu11+-134217725)], alu7);
    var val8 = select(0.0f, data1_134217728[(alu11+256)], alu3);
    var val9 = select(0.0f, data2_117440512[(alu11+-134217472)], alu6);
    var val10 = select(0.0f, data1_134217728[(alu11+257)], alu3);
    var val11 = select(0.0f, data2_117440512[(alu11+-134217471)], alu6);
    var val12 = select(0.0f, data1_134217728[(alu11+258)], alu3);
    var val13 = select(0.0f, data2_117440512[(alu11+-134217470)], alu6);
    var val14 = select(0.0f, data1_134217728[(alu11+259)], alu3);
    var val15 = select(0.0f, data2_117440512[(alu11+-134217469)], alu6);
    var val16 = select(0.0f, data1_134217728[(alu11+512)], alu2);
    var val17 = select(0.0f, data2_117440512[(alu11+-134217216)], alu5);
    var val18 = select(0.0f, data1_134217728[(alu11+513)], alu2);
    var val19 = select(0.0f, data2_117440512[(alu11+-134217215)], alu5);
    var val20 = select(0.0f, data1_134217728[(alu11+514)], alu2);
    var val21 = select(0.0f, data2_117440512[(alu11+-134217214)], alu5);
    var val22 = select(0.0f, data1_134217728[(alu11+515)], alu2);
    var val23 = select(0.0f, data2_117440512[(alu11+-134217213)], alu5);
    acc0[0] = (acc0[0]+val1+val2+val0+val3+val4+val5+val6+val7);
    acc0[1] = (acc0[1]+val8+val9+val10+val11+val12+val13+val14+val15);
    acc0[2] = (acc0[2]+val16+val17+val18+val19+val20+val21+val22+val23);
  }
  data0_983040[(alu1+1)] = acc0[1];
  data0_983040[(alu1+2)] = acc0[2];
  data0_983040[alu1] = acc0[0];
}`;

const r_40_32_3_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3840:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_983040:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 40 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu3 = ((gidx0*24576)+(lidx0*768)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_983040[(alu3+1)];
    var val1 = data1_983040[(alu3+2)];
    var val2 = data1_983040[(alu3+3)];
    var val3 = data1_983040[(alu3+256)];
    var val4 = data1_983040[(alu3+257)];
    var val5 = data1_983040[(alu3+258)];
    var val6 = data1_983040[(alu3+259)];
    var val7 = data1_983040[alu3];
    var val8 = data1_983040[(alu3+512)];
    var val9 = data1_983040[(alu3+513)];
    var val10 = data1_983040[(alu3+514)];
    var val11 = data1_983040[(alu3+515)];
    acc0[0] = (acc0[0]+val7+val0+val1+val2);
    acc0[1] = (acc0[1]+val3+val4+val5+val6);
    acc0[2] = (acc0[2]+val8+val9+val10+val11);
  }
  var alu8 = ((gidx0*96)+(lidx0*3));
  data0_3840[(alu8+1)] = acc0[1];
  data0_3840[(alu8+2)] = acc0[2];
  data0_3840[alu8] = acc0[0];
}`;

const r_15_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_15:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3840:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val0 = data1_3840[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = ((bool(lidx0))!=true);
  if (alu8) {
    data0_15[gidx0] = (acc1[0]*5.960464477539063e-08f);
  }
}`;

const r_5_1024_3_16_4_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_983040:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_134217728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_117440512:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_15:array<f32>;
@compute @workgroup_size(3,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 3 */
  var alu0 = (lidx0+(gidx1*3));
  var val0 = data3_15[alu0];
  var gidx0 = i32(gindex.x); /* 1024 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx0);
  var cast2 = bitcast<u32>(lidx1);
  var alu1 = (alu0<8);
  var alu2 = (7<alu0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu7 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Ridx0)<<2u))+(gidx1*50331648)+bitcast<i32>((cast1<<24u)));
    var val1 = select(0.0f, data1_134217728[alu7], alu1);
    var val2 = select(0.0f, data2_117440512[(alu7+-134217728)], alu2);
    var val3 = select(0.0f, data1_134217728[(alu7+1)], alu1);
    var val4 = select(0.0f, data2_117440512[(alu7+-134217727)], alu2);
    var val5 = select(0.0f, data1_134217728[(alu7+2)], alu1);
    var val6 = select(0.0f, data2_117440512[(alu7+-134217726)], alu2);
    var val7 = select(0.0f, data1_134217728[(alu7+3)], alu1);
    var val8 = select(0.0f, data2_117440512[(alu7+-134217725)], alu2);
    var val9 = select(0.0f, data1_134217728[(alu7+256)], alu1);
    var val10 = select(0.0f, data2_117440512[(alu7+-134217472)], alu2);
    var val11 = select(0.0f, data1_134217728[(alu7+257)], alu1);
    var val12 = select(0.0f, data2_117440512[(alu7+-134217471)], alu2);
    var val13 = select(0.0f, data1_134217728[(alu7+258)], alu1);
    var val14 = select(0.0f, data2_117440512[(alu7+-134217470)], alu2);
    var val15 = select(0.0f, data1_134217728[(alu7+259)], alu1);
    var val16 = select(0.0f, data2_117440512[(alu7+-134217469)], alu2);
    var val17 = select(0.0f, data1_134217728[(alu7+512)], alu1);
    var val18 = select(0.0f, data2_117440512[(alu7+-134217216)], alu2);
    var val19 = select(0.0f, data1_134217728[(alu7+513)], alu1);
    var val20 = select(0.0f, data2_117440512[(alu7+-134217215)], alu2);
    var val21 = select(0.0f, data1_134217728[(alu7+514)], alu1);
    var val22 = select(0.0f, data2_117440512[(alu7+-134217214)], alu2);
    var val23 = select(0.0f, data1_134217728[(alu7+515)], alu1);
    var val24 = select(0.0f, data2_117440512[(alu7+-134217213)], alu2);
    var val25 = select(0.0f, data1_134217728[(alu7+768)], alu1);
    var val26 = select(0.0f, data2_117440512[(alu7+-134216960)], alu2);
    var val27 = select(0.0f, data1_134217728[(alu7+769)], alu1);
    var val28 = select(0.0f, data2_117440512[(alu7+-134216959)], alu2);
    var val29 = select(0.0f, data1_134217728[(alu7+770)], alu1);
    var val30 = select(0.0f, data2_117440512[(alu7+-134216958)], alu2);
    var val31 = select(0.0f, data1_134217728[(alu7+771)], alu1);
    var val32 = select(0.0f, data2_117440512[(alu7+-134216957)], alu2);
    var alu8 = ((val1+val2)-val0);
    var alu9 = ((val9+val10)-val0);
    var alu10 = ((val17+val18)-val0);
    var alu11 = ((val25+val26)-val0);
    var alu12 = ((val3+val4)-val0);
    var alu13 = ((val11+val12)-val0);
    var alu14 = ((val19+val20)-val0);
    var alu15 = ((val27+val28)-val0);
    var alu16 = ((val5+val6)-val0);
    var alu17 = ((val13+val14)-val0);
    var alu18 = ((val21+val22)-val0);
    var alu19 = ((val29+val30)-val0);
    var alu20 = ((val7+val8)-val0);
    var alu21 = ((val15+val16)-val0);
    var alu22 = ((val23+val24)-val0);
    var alu23 = ((val31+val32)-val0);
    acc0[0] = (acc0[0]+(alu8*alu8)+(alu12*alu12)+(alu16*alu16)+(alu20*alu20));
    acc0[1] = (acc0[1]+(alu9*alu9)+(alu13*alu13)+(alu17*alu17)+(alu21*alu21));
    acc0[2] = (acc0[2]+(alu10*alu10)+(alu14*alu14)+(alu18*alu18)+(alu22*alu22));
    acc0[3] = (acc0[3]+(alu11*alu11)+(alu15*alu15)+(alu19*alu19)+(alu23*alu23));
  }
  var alu29 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+(gidx1*196608)+bitcast<i32>((cast1<<16u)));
  data0_983040[alu29] = acc0[0];
  data0_983040[(alu29+1)] = acc0[1];
  data0_983040[(alu29+2)] = acc0[2];
  data0_983040[(alu29+3)] = acc0[3];
}`;

const r_15_16_16n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_15:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3840:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val0 = data1_3840[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = ((bool(lidx0))!=true);
  if (alu8) {
    data0_15[gidx0] = (1/sqrt(((acc1[0]*5.960464477539063e-08f)+1e-05f)));
  }
}`;

const E_5_262144_3_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_251658240:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_134217728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_117440512:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_15:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_15:array<f32>;
@compute @workgroup_size(3,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32768 */
  var gidx1 = i32(gindex.y); /* 40 */
  var lidx0 = i32(lindex.x); /* 3 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = ((gidx1*13)>>6u);
  var alu1 = (gidx1-(5*alu0));
  var alu2 = (lidx0+(alu1*3));
  var alu3 = (bitcast<i32>((bitcast<u32>(gidx0)<<9u))+bitcast<i32>((bitcast<u32>(alu0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((bitcast<u32>(lidx0)<<24u))+(alu1*50331648));
  var alu4 = (alu2<8);
  var val0 = select(0.0f, data1_134217728[alu3], alu4);
  var alu5 = (7<alu2);
  var val1 = select(0.0f, data2_117440512[(alu3+-134217728)], alu5);
  var val2 = data3_15[alu2];
  var val3 = data4_15[alu2];
  var alu6 = (alu3+1);
  var val4 = select(0.0f, data1_134217728[alu6], alu4);
  var val5 = select(0.0f, data2_117440512[(alu3+-134217727)], alu5);
  var alu7 = (alu3+2);
  var val6 = select(0.0f, data1_134217728[alu7], alu4);
  var val7 = select(0.0f, data2_117440512[(alu3+-134217726)], alu5);
  var alu8 = (alu3+3);
  var val8 = select(0.0f, data1_134217728[alu8], alu4);
  var val9 = select(0.0f, data2_117440512[(alu3+-134217725)], alu5);
  var alu9 = (((val0+val1)-val2)*val3);
  var alu10 = (((val4+val5)-val2)*val3);
  var alu11 = (((val6+val7)-val2)*val3);
  var alu12 = (((val8+val9)-val2)*val3);
  data0_251658240[alu3] = ((1/(1.0f+exp2(((alu9+(0.044715f*alu9*alu9*alu9))*-2.302208198144325f))))*alu9);
  data0_251658240[alu6] = ((1/(1.0f+exp2(((alu10+(0.044715f*alu10*alu10*alu10))*-2.302208198144325f))))*alu10);
  data0_251658240[alu7] = ((1/(1.0f+exp2(((alu11+(0.044715f*alu11*alu11*alu11))*-2.302208198144325f))))*alu11);
  data0_251658240[alu8] = ((1/(1.0f+exp2(((alu12+(0.044715f*alu12*alu12*alu12))*-2.302208198144325f))))*alu12);
}`;

const r_2_256_32_4_8_16_4_4_15_3_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_134217728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx1+bitcast<i32>((cast1<<4u)));
  var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>((gidx0>>2u))<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast1<<6u)));
  var alu2 = (gidx0<124);
  var alu3 = (alu0<62);
  var alu4 = (1<alu0);
  var alu5 = (3<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast2 = bitcast<u32>(Ridx1);
      var alu22 = (gidx1+bitcast<i32>((cast2<<3u)));
      var alu23 = (alu1+cast0+bitcast<i32>((cast2<<19u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu24 = ((7<alu22)&(alu22<264));
      var alu25 = (alu4&alu5&alu24);
      var val0 = select(0.0f, data1_251658240[(alu23+-526344)], alu25);
      var alu26 = ((Ridx0*27)+(Ridx1*9)+(gidx2*1620));
      var val1 = data2_6075[(alu26+1)];
      var val2 = data2_6075[alu26];
      var alu27 = (alu5&alu24);
      var val3 = select(0.0f, data1_251658240[(alu23+-526336)], alu27);
      var alu28 = (alu3&alu5&alu24);
      var val4 = select(0.0f, data1_251658240[(alu23+-526328)], alu28);
      var val5 = data2_6075[(alu26+2)];
      var alu29 = (alu4&alu24);
      var val6 = select(0.0f, data1_251658240[(alu23+-524296)], alu29);
      var val7 = data2_6075[(alu26+3)];
      var val8 = select(0.0f, data1_251658240[(alu23+-524288)], alu24);
      var val9 = data2_6075[(alu26+4)];
      var alu30 = (alu3&alu24);
      var val10 = select(0.0f, data1_251658240[(alu23+-524280)], alu30);
      var val11 = data2_6075[(alu26+5)];
      var alu31 = (alu4&alu2&alu24);
      var val12 = select(0.0f, data1_251658240[(alu23+-522248)], alu31);
      var val13 = data2_6075[(alu26+6)];
      var alu32 = (alu2&alu24);
      var val14 = select(0.0f, data1_251658240[(alu23+-522240)], alu32);
      var val15 = data2_6075[(alu26+7)];
      var alu33 = (alu3&alu2&alu24);
      var val16 = select(0.0f, data1_251658240[(alu23+-522232)], alu33);
      var val17 = data2_6075[(alu26+8)];
      var val18 = data2_6075[(alu26+405)];
      var val19 = data2_6075[(alu26+406)];
      var val20 = data2_6075[(alu26+407)];
      var val21 = data2_6075[(alu26+408)];
      var val22 = data2_6075[(alu26+409)];
      var val23 = data2_6075[(alu26+410)];
      var val24 = data2_6075[(alu26+411)];
      var val25 = data2_6075[(alu26+412)];
      var val26 = data2_6075[(alu26+413)];
      var val27 = data2_6075[(alu26+810)];
      var val28 = data2_6075[(alu26+811)];
      var val29 = data2_6075[(alu26+812)];
      var val30 = data2_6075[(alu26+813)];
      var val31 = data2_6075[(alu26+814)];
      var val32 = data2_6075[(alu26+815)];
      var val33 = data2_6075[(alu26+816)];
      var val34 = data2_6075[(alu26+817)];
      var val35 = data2_6075[(alu26+818)];
      var val36 = data2_6075[(alu26+1215)];
      var val37 = data2_6075[(alu26+1216)];
      var val38 = data2_6075[(alu26+1217)];
      var val39 = data2_6075[(alu26+1218)];
      var val40 = data2_6075[(alu26+1219)];
      var val41 = data2_6075[(alu26+1220)];
      var val42 = data2_6075[(alu26+1221)];
      var val43 = data2_6075[(alu26+1222)];
      var val44 = data2_6075[(alu26+1223)];
      var val45 = select(0.0f, data1_251658240[(alu23+-526343)], alu25);
      var val46 = select(0.0f, data1_251658240[(alu23+-526335)], alu27);
      var val47 = select(0.0f, data1_251658240[(alu23+-526327)], alu28);
      var val48 = select(0.0f, data1_251658240[(alu23+-524295)], alu29);
      var val49 = select(0.0f, data1_251658240[(alu23+-524287)], alu24);
      var val50 = select(0.0f, data1_251658240[(alu23+-524279)], alu30);
      var val51 = select(0.0f, data1_251658240[(alu23+-522247)], alu31);
      var val52 = select(0.0f, data1_251658240[(alu23+-522239)], alu32);
      var val53 = select(0.0f, data1_251658240[(alu23+-522231)], alu33);
      var val54 = select(0.0f, data1_251658240[(alu23+-526342)], alu25);
      var val55 = select(0.0f, data1_251658240[(alu23+-524294)], alu29);
      var val56 = select(0.0f, data1_251658240[(alu23+-524286)], alu24);
      var val57 = select(0.0f, data1_251658240[(alu23+-524278)], alu30);
      var val58 = select(0.0f, data1_251658240[(alu23+-522246)], alu31);
      var val59 = select(0.0f, data1_251658240[(alu23+-522238)], alu32);
      var val60 = select(0.0f, data1_251658240[(alu23+-522230)], alu33);
      var val61 = select(0.0f, data1_251658240[(alu23+-526341)], alu25);
      var val62 = select(0.0f, data1_251658240[(alu23+-526334)], alu27);
      var val63 = select(0.0f, data1_251658240[(alu23+-526333)], alu27);
      var val64 = select(0.0f, data1_251658240[(alu23+-526326)], alu28);
      var val65 = select(0.0f, data1_251658240[(alu23+-526325)], alu28);
      var val66 = select(0.0f, data1_251658240[(alu23+-524293)], alu29);
      var val67 = select(0.0f, data1_251658240[(alu23+-524285)], alu24);
      var val68 = select(0.0f, data1_251658240[(alu23+-524277)], alu30);
      var val69 = select(0.0f, data1_251658240[(alu23+-522245)], alu31);
      var val70 = select(0.0f, data1_251658240[(alu23+-522237)], alu32);
      var val71 = select(0.0f, data1_251658240[(alu23+-522229)], alu33);
      acc0[0] = (acc0[0]+(val0*val2)+(val3*val1)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val0*val18)+(val3*val19)+(val4*val20)+(val6*val21)+(val8*val22)+(val10*val23)+(val12*val24)+(val14*val25)+(val16*val26));
      acc0[2] = (acc0[2]+(val0*val27)+(val3*val28)+(val4*val29)+(val6*val30)+(val8*val31)+(val10*val32)+(val12*val33)+(val14*val34)+(val16*val35));
      acc0[3] = (acc0[3]+(val0*val36)+(val3*val37)+(val4*val38)+(val6*val39)+(val8*val40)+(val10*val41)+(val12*val42)+(val14*val43)+(val16*val44));
      acc0[4] = (acc0[4]+(val45*val2)+(val46*val1)+(val47*val5)+(val48*val7)+(val49*val9)+(val50*val11)+(val51*val13)+(val52*val15)+(val53*val17));
      acc0[5] = (acc0[5]+(val45*val18)+(val46*val19)+(val47*val20)+(val48*val21)+(val49*val22)+(val50*val23)+(val51*val24)+(val52*val25)+(val53*val26));
      acc0[6] = (acc0[6]+(val45*val27)+(val46*val28)+(val47*val29)+(val48*val30)+(val49*val31)+(val50*val32)+(val51*val33)+(val52*val34)+(val53*val35));
      acc0[7] = (acc0[7]+(val45*val36)+(val46*val37)+(val47*val38)+(val48*val39)+(val49*val40)+(val50*val41)+(val51*val42)+(val52*val43)+(val53*val44));
      acc0[8] = (acc0[8]+(val54*val2)+(val62*val1)+(val64*val5)+(val55*val7)+(val56*val9)+(val57*val11)+(val58*val13)+(val59*val15)+(val60*val17));
      acc0[9] = (acc0[9]+(val54*val18)+(val62*val19)+(val64*val20)+(val55*val21)+(val56*val22)+(val57*val23)+(val58*val24)+(val59*val25)+(val60*val26));
      acc0[10] = (acc0[10]+(val54*val27)+(val62*val28)+(val64*val29)+(val55*val30)+(val56*val31)+(val57*val32)+(val58*val33)+(val59*val34)+(val60*val35));
      acc0[11] = (acc0[11]+(val54*val36)+(val62*val37)+(val64*val38)+(val55*val39)+(val56*val40)+(val57*val41)+(val58*val42)+(val59*val43)+(val60*val44));
      acc0[12] = (acc0[12]+(val61*val2)+(val63*val1)+(val65*val5)+(val66*val7)+(val67*val9)+(val68*val11)+(val69*val13)+(val70*val15)+(val71*val17));
      acc0[13] = (acc0[13]+(val61*val18)+(val63*val19)+(val65*val20)+(val66*val21)+(val67*val22)+(val68*val23)+(val69*val24)+(val70*val25)+(val71*val26));
      acc0[14] = (acc0[14]+(val61*val27)+(val63*val28)+(val65*val29)+(val66*val30)+(val67*val31)+(val68*val32)+(val69*val33)+(val70*val34)+(val71*val35));
      acc0[15] = (acc0[15]+(val61*val36)+(val63*val37)+(val65*val38)+(val66*val39)+(val67*val40)+(val68*val41)+(val69*val42)+(val70*val43)+(val71*val44));
    }
  }
  var alu52 = (alu1+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<26u)));
  data0_134217728[alu52] = acc0[0];
  data0_134217728[(alu52+1)] = acc0[4];
  data0_134217728[(alu52+2)] = acc0[8];
  data0_134217728[(alu52+3)] = acc0[12];
  data0_134217728[(alu52+16777216)] = acc0[1];
  data0_134217728[(alu52+16777217)] = acc0[5];
  data0_134217728[(alu52+16777218)] = acc0[9];
  data0_134217728[(alu52+16777219)] = acc0[13];
  data0_134217728[(alu52+33554432)] = acc0[2];
  data0_134217728[(alu52+33554433)] = acc0[6];
  data0_134217728[(alu52+33554434)] = acc0[10];
  data0_134217728[(alu52+33554435)] = acc0[14];
  data0_134217728[(alu52+50331648)] = acc0[3];
  data0_134217728[(alu52+50331649)] = acc0[7];
  data0_134217728[(alu52+50331650)] = acc0[11];
  data0_134217728[(alu52+50331651)] = acc0[15];
}`;

const r_7_256_32_4_8_16_4_15_3_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_117440512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx1+bitcast<i32>((cast1<<4u)));
  var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>((gidx0>>2u))<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast1<<6u)));
  var alu2 = (gidx0<124);
  var alu3 = (alu0<62);
  var alu4 = (1<alu0);
  var alu5 = (3<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast2 = bitcast<u32>(Ridx1);
      var alu10 = (gidx1+bitcast<i32>((cast2<<3u)));
      var alu11 = (alu1+cast0+bitcast<i32>((cast2<<19u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu12 = ((7<alu10)&(alu10<264));
      var alu13 = (alu4&alu5&alu12);
      var val0 = select(0.0f, data1_251658240[(alu11+-526344)], alu13);
      var alu14 = ((Ridx0*27)+(Ridx1*9)+(gidx2*405));
      var val1 = data2_6075[(alu14+3240)];
      var alu15 = (alu5&alu12);
      var val2 = select(0.0f, data1_251658240[(alu11+-526336)], alu15);
      var val3 = data2_6075[(alu14+3241)];
      var alu16 = (alu3&alu5&alu12);
      var val4 = select(0.0f, data1_251658240[(alu11+-526328)], alu16);
      var val5 = data2_6075[(alu14+3242)];
      var alu17 = (alu4&alu12);
      var val6 = select(0.0f, data1_251658240[(alu11+-524296)], alu17);
      var val7 = data2_6075[(alu14+3243)];
      var val8 = select(0.0f, data1_251658240[(alu11+-524288)], alu12);
      var val9 = data2_6075[(alu14+3244)];
      var alu18 = (alu3&alu12);
      var val10 = select(0.0f, data1_251658240[(alu11+-524280)], alu18);
      var val11 = data2_6075[(alu14+3245)];
      var alu19 = (alu4&alu2&alu12);
      var val12 = select(0.0f, data1_251658240[(alu11+-522248)], alu19);
      var val13 = data2_6075[(alu14+3246)];
      var alu20 = (alu2&alu12);
      var val14 = select(0.0f, data1_251658240[(alu11+-522240)], alu20);
      var val15 = data2_6075[(alu14+3247)];
      var alu21 = (alu3&alu2&alu12);
      var val16 = select(0.0f, data1_251658240[(alu11+-522232)], alu21);
      var val17 = data2_6075[(alu14+3248)];
      var val18 = select(0.0f, data1_251658240[(alu11+-526343)], alu13);
      var val19 = select(0.0f, data1_251658240[(alu11+-526335)], alu15);
      var val20 = select(0.0f, data1_251658240[(alu11+-526327)], alu16);
      var val21 = select(0.0f, data1_251658240[(alu11+-524295)], alu17);
      var val22 = select(0.0f, data1_251658240[(alu11+-524287)], alu12);
      var val23 = select(0.0f, data1_251658240[(alu11+-524279)], alu18);
      var val24 = select(0.0f, data1_251658240[(alu11+-522247)], alu19);
      var val25 = select(0.0f, data1_251658240[(alu11+-522239)], alu20);
      var val26 = select(0.0f, data1_251658240[(alu11+-522231)], alu21);
      var val27 = select(0.0f, data1_251658240[(alu11+-526342)], alu13);
      var val28 = select(0.0f, data1_251658240[(alu11+-526341)], alu13);
      var val29 = select(0.0f, data1_251658240[(alu11+-526334)], alu15);
      var val30 = select(0.0f, data1_251658240[(alu11+-526333)], alu15);
      var val31 = select(0.0f, data1_251658240[(alu11+-526326)], alu16);
      var val32 = select(0.0f, data1_251658240[(alu11+-526325)], alu16);
      var val33 = select(0.0f, data1_251658240[(alu11+-524294)], alu17);
      var val34 = select(0.0f, data1_251658240[(alu11+-524293)], alu17);
      var val35 = select(0.0f, data1_251658240[(alu11+-524286)], alu12);
      var val36 = select(0.0f, data1_251658240[(alu11+-524285)], alu12);
      var val37 = select(0.0f, data1_251658240[(alu11+-524278)], alu18);
      var val38 = select(0.0f, data1_251658240[(alu11+-522246)], alu19);
      var val39 = select(0.0f, data1_251658240[(alu11+-522238)], alu20);
      var val40 = select(0.0f, data1_251658240[(alu11+-522230)], alu21);
      var val41 = select(0.0f, data1_251658240[(alu11+-524277)], alu18);
      var val42 = select(0.0f, data1_251658240[(alu11+-522245)], alu19);
      var val43 = select(0.0f, data1_251658240[(alu11+-522237)], alu20);
      var val44 = select(0.0f, data1_251658240[(alu11+-522229)], alu21);
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val18*val1)+(val19*val3)+(val20*val5)+(val21*val7)+(val22*val9)+(val23*val11)+(val24*val13)+(val25*val15)+(val26*val17));
      acc0[2] = (acc0[2]+(val27*val1)+(val29*val3)+(val31*val5)+(val33*val7)+(val35*val9)+(val37*val11)+(val38*val13)+(val39*val15)+(val40*val17));
      acc0[3] = (acc0[3]+(val28*val1)+(val30*val3)+(val32*val5)+(val34*val7)+(val36*val9)+(val41*val11)+(val42*val13)+(val43*val15)+(val44*val17));
    }
  }
  var alu28 = (alu1+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<24u)));
  data0_117440512[alu28] = acc0[0];
  data0_117440512[(alu28+1)] = acc0[1];
  data0_117440512[(alu28+2)] = acc0[2];
  data0_117440512[(alu28+3)] = acc0[3];
}`;

const r_2_256_32_4_8_16_4_4_15_3_3_3n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_134217728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0>>2u));
  var alu0 = (gidx0&3);
  var cast2 = bitcast<u32>(alu0);
  var alu1 = (lidx0+bitcast<i32>((cast1<<3u)));
  var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast2<<6u)));
  var alu3 = (alu1<252);
  var alu4 = ((lidx1+bitcast<i32>((cast2<<4u)))<63);
  var alu5 = (0<(lidx1+alu0));
  var alu6 = (3<alu1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast3 = bitcast<u32>(Ridx1);
      var alu23 = (gidx1+bitcast<i32>((cast3<<2u)));
      var alu24 = (alu2+cast0+bitcast<i32>((cast3<<18u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu25 = ((3<alu23)&(alu23<260));
      var alu26 = (alu5&alu6&alu25);
      var val0 = select(0.0f, data1_251658240[(alu24+-263172)], alu26);
      var alu27 = ((Ridx0*27)+(Ridx1*9)+(gidx2*1620));
      var val1 = data2_6075[(alu27+1)];
      var val2 = data2_6075[(alu27+2)];
      var val3 = data2_6075[alu27];
      var alu28 = (alu6&alu25);
      var val4 = select(0.0f, data1_251658240[(alu24+-263168)], alu28);
      var alu29 = (alu4&alu6&alu25);
      var val5 = select(0.0f, data1_251658240[(alu24+-263164)], alu29);
      var alu30 = (alu5&alu25);
      var val6 = select(0.0f, data1_251658240[(alu24+-262148)], alu30);
      var val7 = data2_6075[(alu27+3)];
      var val8 = select(0.0f, data1_251658240[(alu24+-262144)], alu25);
      var val9 = data2_6075[(alu27+4)];
      var alu31 = (alu4&alu25);
      var val10 = select(0.0f, data1_251658240[(alu24+-262140)], alu31);
      var val11 = data2_6075[(alu27+5)];
      var alu32 = (alu5&alu3&alu25);
      var val12 = select(0.0f, data1_251658240[(alu24+-261124)], alu32);
      var val13 = data2_6075[(alu27+6)];
      var alu33 = (alu3&alu25);
      var val14 = select(0.0f, data1_251658240[(alu24+-261120)], alu33);
      var val15 = data2_6075[(alu27+7)];
      var alu34 = (alu4&alu3&alu25);
      var val16 = select(0.0f, data1_251658240[(alu24+-261116)], alu34);
      var val17 = data2_6075[(alu27+8)];
      var val18 = data2_6075[(alu27+405)];
      var val19 = data2_6075[(alu27+406)];
      var val20 = data2_6075[(alu27+407)];
      var val21 = data2_6075[(alu27+408)];
      var val22 = data2_6075[(alu27+409)];
      var val23 = data2_6075[(alu27+410)];
      var val24 = data2_6075[(alu27+411)];
      var val25 = data2_6075[(alu27+412)];
      var val26 = data2_6075[(alu27+413)];
      var val27 = data2_6075[(alu27+810)];
      var val28 = data2_6075[(alu27+811)];
      var val29 = data2_6075[(alu27+812)];
      var val30 = data2_6075[(alu27+813)];
      var val31 = data2_6075[(alu27+814)];
      var val32 = data2_6075[(alu27+815)];
      var val33 = data2_6075[(alu27+816)];
      var val34 = data2_6075[(alu27+817)];
      var val35 = data2_6075[(alu27+818)];
      var val36 = data2_6075[(alu27+1215)];
      var val37 = data2_6075[(alu27+1216)];
      var val38 = data2_6075[(alu27+1217)];
      var val39 = data2_6075[(alu27+1218)];
      var val40 = data2_6075[(alu27+1219)];
      var val41 = data2_6075[(alu27+1220)];
      var val42 = data2_6075[(alu27+1221)];
      var val43 = data2_6075[(alu27+1222)];
      var val44 = data2_6075[(alu27+1223)];
      var val45 = select(0.0f, data1_251658240[(alu24+-263171)], alu26);
      var val46 = select(0.0f, data1_251658240[(alu24+-263167)], alu28);
      var val47 = select(0.0f, data1_251658240[(alu24+-263163)], alu29);
      var val48 = select(0.0f, data1_251658240[(alu24+-262147)], alu30);
      var val49 = select(0.0f, data1_251658240[(alu24+-262143)], alu25);
      var val50 = select(0.0f, data1_251658240[(alu24+-262139)], alu31);
      var val51 = select(0.0f, data1_251658240[(alu24+-261123)], alu32);
      var val52 = select(0.0f, data1_251658240[(alu24+-261119)], alu33);
      var val53 = select(0.0f, data1_251658240[(alu24+-261115)], alu34);
      var val54 = select(0.0f, data1_251658240[(alu24+-263170)], alu26);
      var val55 = select(0.0f, data1_251658240[(alu24+-262146)], alu30);
      var val56 = select(0.0f, data1_251658240[(alu24+-262142)], alu25);
      var val57 = select(0.0f, data1_251658240[(alu24+-262138)], alu31);
      var val58 = select(0.0f, data1_251658240[(alu24+-261122)], alu32);
      var val59 = select(0.0f, data1_251658240[(alu24+-261118)], alu33);
      var val60 = select(0.0f, data1_251658240[(alu24+-261114)], alu34);
      var val61 = select(0.0f, data1_251658240[(alu24+-263169)], alu26);
      var val62 = select(0.0f, data1_251658240[(alu24+-263166)], alu28);
      var val63 = select(0.0f, data1_251658240[(alu24+-263165)], alu28);
      var val64 = select(0.0f, data1_251658240[(alu24+-263162)], alu29);
      var val65 = select(0.0f, data1_251658240[(alu24+-263161)], alu29);
      var val66 = select(0.0f, data1_251658240[(alu24+-262145)], alu30);
      var val67 = select(0.0f, data1_251658240[(alu24+-262141)], alu25);
      var val68 = select(0.0f, data1_251658240[(alu24+-262137)], alu31);
      var val69 = select(0.0f, data1_251658240[(alu24+-261121)], alu32);
      var val70 = select(0.0f, data1_251658240[(alu24+-261117)], alu33);
      var val71 = select(0.0f, data1_251658240[(alu24+-261113)], alu34);
      acc0[0] = (acc0[0]+(val0*val3)+(val4*val1)+(val5*val2)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val0*val18)+(val4*val19)+(val5*val20)+(val6*val21)+(val8*val22)+(val10*val23)+(val12*val24)+(val14*val25)+(val16*val26));
      acc0[2] = (acc0[2]+(val0*val27)+(val4*val28)+(val5*val29)+(val6*val30)+(val8*val31)+(val10*val32)+(val12*val33)+(val14*val34)+(val16*val35));
      acc0[3] = (acc0[3]+(val0*val36)+(val4*val37)+(val5*val38)+(val6*val39)+(val8*val40)+(val10*val41)+(val12*val42)+(val14*val43)+(val16*val44));
      acc0[4] = (acc0[4]+(val45*val3)+(val46*val1)+(val47*val2)+(val48*val7)+(val49*val9)+(val50*val11)+(val51*val13)+(val52*val15)+(val53*val17));
      acc0[5] = (acc0[5]+(val45*val18)+(val46*val19)+(val47*val20)+(val48*val21)+(val49*val22)+(val50*val23)+(val51*val24)+(val52*val25)+(val53*val26));
      acc0[6] = (acc0[6]+(val45*val27)+(val46*val28)+(val47*val29)+(val48*val30)+(val49*val31)+(val50*val32)+(val51*val33)+(val52*val34)+(val53*val35));
      acc0[7] = (acc0[7]+(val45*val36)+(val46*val37)+(val47*val38)+(val48*val39)+(val49*val40)+(val50*val41)+(val51*val42)+(val52*val43)+(val53*val44));
      acc0[8] = (acc0[8]+(val54*val3)+(val62*val1)+(val64*val2)+(val55*val7)+(val56*val9)+(val57*val11)+(val58*val13)+(val59*val15)+(val60*val17));
      acc0[9] = (acc0[9]+(val54*val18)+(val62*val19)+(val64*val20)+(val55*val21)+(val56*val22)+(val57*val23)+(val58*val24)+(val59*val25)+(val60*val26));
      acc0[10] = (acc0[10]+(val54*val27)+(val62*val28)+(val64*val29)+(val55*val30)+(val56*val31)+(val57*val32)+(val58*val33)+(val59*val34)+(val60*val35));
      acc0[11] = (acc0[11]+(val54*val36)+(val62*val37)+(val64*val38)+(val55*val39)+(val56*val40)+(val57*val41)+(val58*val42)+(val59*val43)+(val60*val44));
      acc0[12] = (acc0[12]+(val61*val3)+(val63*val1)+(val65*val2)+(val66*val7)+(val67*val9)+(val68*val11)+(val69*val13)+(val70*val15)+(val71*val17));
      acc0[13] = (acc0[13]+(val61*val18)+(val63*val19)+(val65*val20)+(val66*val21)+(val67*val22)+(val68*val23)+(val69*val24)+(val70*val25)+(val71*val26));
      acc0[14] = (acc0[14]+(val61*val27)+(val63*val28)+(val65*val29)+(val66*val30)+(val67*val31)+(val68*val32)+(val69*val33)+(val70*val34)+(val71*val35));
      acc0[15] = (acc0[15]+(val61*val36)+(val63*val37)+(val65*val38)+(val66*val39)+(val67*val40)+(val68*val41)+(val69*val42)+(val70*val43)+(val71*val44));
    }
  }
  var alu53 = (alu2+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<26u)));
  data0_134217728[alu53] = acc0[0];
  data0_134217728[(alu53+1)] = acc0[4];
  data0_134217728[(alu53+2)] = acc0[8];
  data0_134217728[(alu53+3)] = acc0[12];
  data0_134217728[(alu53+16777216)] = acc0[1];
  data0_134217728[(alu53+16777217)] = acc0[5];
  data0_134217728[(alu53+16777218)] = acc0[9];
  data0_134217728[(alu53+16777219)] = acc0[13];
  data0_134217728[(alu53+33554432)] = acc0[2];
  data0_134217728[(alu53+33554433)] = acc0[6];
  data0_134217728[(alu53+33554434)] = acc0[10];
  data0_134217728[(alu53+33554435)] = acc0[14];
  data0_134217728[(alu53+50331648)] = acc0[3];
  data0_134217728[(alu53+50331649)] = acc0[7];
  data0_134217728[(alu53+50331650)] = acc0[11];
  data0_134217728[(alu53+50331651)] = acc0[15];
}`;

const r_7_256_32_4_8_16_4_15_3_3_3n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_117440512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0>>2u));
  var alu0 = (gidx0&3);
  var cast2 = bitcast<u32>(alu0);
  var alu1 = (lidx0+bitcast<i32>((cast1<<3u)));
  var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast2<<6u)));
  var alu3 = (alu1<252);
  var alu4 = ((lidx1+bitcast<i32>((cast2<<4u)))<63);
  var alu5 = (0<(lidx1+alu0));
  var alu6 = (3<alu1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast3 = bitcast<u32>(Ridx1);
      var alu11 = (gidx1+bitcast<i32>((cast3<<2u)));
      var alu12 = (alu2+cast0+bitcast<i32>((cast3<<18u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu13 = ((3<alu11)&(alu11<260));
      var alu14 = (alu5&alu6&alu13);
      var val0 = select(0.0f, data1_251658240[(alu12+-263172)], alu14);
      var alu15 = ((Ridx0*27)+(Ridx1*9)+(gidx2*405));
      var val1 = data2_6075[(alu15+3240)];
      var alu16 = (alu6&alu13);
      var val2 = select(0.0f, data1_251658240[(alu12+-263168)], alu16);
      var val3 = data2_6075[(alu15+3241)];
      var alu17 = (alu4&alu6&alu13);
      var val4 = select(0.0f, data1_251658240[(alu12+-263164)], alu17);
      var val5 = data2_6075[(alu15+3242)];
      var alu18 = (alu5&alu13);
      var val6 = select(0.0f, data1_251658240[(alu12+-262148)], alu18);
      var val7 = data2_6075[(alu15+3243)];
      var val8 = select(0.0f, data1_251658240[(alu12+-262144)], alu13);
      var val9 = data2_6075[(alu15+3244)];
      var alu19 = (alu4&alu13);
      var val10 = select(0.0f, data1_251658240[(alu12+-262140)], alu19);
      var val11 = data2_6075[(alu15+3245)];
      var alu20 = (alu5&alu3&alu13);
      var val12 = select(0.0f, data1_251658240[(alu12+-261124)], alu20);
      var val13 = data2_6075[(alu15+3246)];
      var alu21 = (alu3&alu13);
      var val14 = select(0.0f, data1_251658240[(alu12+-261120)], alu21);
      var val15 = data2_6075[(alu15+3247)];
      var alu22 = (alu4&alu3&alu13);
      var val16 = select(0.0f, data1_251658240[(alu12+-261116)], alu22);
      var val17 = data2_6075[(alu15+3248)];
      var val18 = select(0.0f, data1_251658240[(alu12+-263171)], alu14);
      var val19 = select(0.0f, data1_251658240[(alu12+-263170)], alu14);
      var val20 = select(0.0f, data1_251658240[(alu12+-263167)], alu16);
      var val21 = select(0.0f, data1_251658240[(alu12+-263166)], alu16);
      var val22 = select(0.0f, data1_251658240[(alu12+-263163)], alu17);
      var val23 = select(0.0f, data1_251658240[(alu12+-263162)], alu17);
      var val24 = select(0.0f, data1_251658240[(alu12+-262147)], alu18);
      var val25 = select(0.0f, data1_251658240[(alu12+-262146)], alu18);
      var val26 = select(0.0f, data1_251658240[(alu12+-262143)], alu13);
      var val27 = select(0.0f, data1_251658240[(alu12+-262139)], alu19);
      var val28 = select(0.0f, data1_251658240[(alu12+-262138)], alu19);
      var val29 = select(0.0f, data1_251658240[(alu12+-261123)], alu20);
      var val30 = select(0.0f, data1_251658240[(alu12+-261119)], alu21);
      var val31 = select(0.0f, data1_251658240[(alu12+-261115)], alu22);
      var val32 = select(0.0f, data1_251658240[(alu12+-263169)], alu14);
      var val33 = select(0.0f, data1_251658240[(alu12+-263165)], alu16);
      var val34 = select(0.0f, data1_251658240[(alu12+-263161)], alu17);
      var val35 = select(0.0f, data1_251658240[(alu12+-262145)], alu18);
      var val36 = select(0.0f, data1_251658240[(alu12+-262142)], alu13);
      var val37 = select(0.0f, data1_251658240[(alu12+-262141)], alu13);
      var val38 = select(0.0f, data1_251658240[(alu12+-261122)], alu20);
      var val39 = select(0.0f, data1_251658240[(alu12+-261118)], alu21);
      var val40 = select(0.0f, data1_251658240[(alu12+-261114)], alu22);
      var val41 = select(0.0f, data1_251658240[(alu12+-262137)], alu19);
      var val42 = select(0.0f, data1_251658240[(alu12+-261121)], alu20);
      var val43 = select(0.0f, data1_251658240[(alu12+-261117)], alu21);
      var val44 = select(0.0f, data1_251658240[(alu12+-261113)], alu22);
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val18*val1)+(val20*val3)+(val22*val5)+(val24*val7)+(val26*val9)+(val27*val11)+(val29*val13)+(val30*val15)+(val31*val17));
      acc0[2] = (acc0[2]+(val19*val1)+(val21*val3)+(val23*val5)+(val25*val7)+(val36*val9)+(val28*val11)+(val38*val13)+(val39*val15)+(val40*val17));
      acc0[3] = (acc0[3]+(val32*val1)+(val33*val3)+(val34*val5)+(val35*val7)+(val37*val9)+(val41*val11)+(val42*val13)+(val43*val15)+(val44*val17));
    }
  }
  var alu29 = (alu2+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<24u)));
  data0_117440512[alu29] = acc0[0];
  data0_117440512[(alu29+1)] = acc0[1];
  data0_117440512[(alu29+2)] = acc0[2];
  data0_117440512[(alu29+3)] = acc0[3];
}`;

const r_2_256_32_4_8_16_4_4_15_3_3_3n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_134217728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>(lidx1);
  var cast2 = bitcast<u32>((gidx0>>2u));
  var cast3 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx0+bitcast<i32>((cast2<<3u)));
  var alu1 = (bitcast<i32>((cast1<<2u))+bitcast<i32>((cast3<<6u)));
  var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((cast2<<11u))+alu1);
  var alu3 = (alu0<254);
  var alu4 = ((lidx1+bitcast<i32>((cast3<<4u)))<63);
  var alu5 = (alu1<251);
  var alu6 = (0<(bitcast<i32>((cast1<<1u))+bitcast<i32>((cast3<<5u))));
  var alu7 = (0<alu1);
  var alu8 = (1<alu0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast4 = bitcast<u32>(Ridx1);
      var alu25 = (gidx1+bitcast<i32>((cast4<<1u)));
      var alu26 = (alu2+cast0+bitcast<i32>((cast4<<17u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu27 = ((1<alu25)&(alu25<258));
      var val0 = select(0.0f, data1_251658240[(alu26+-131586)], (alu6&alu8&alu27));
      var alu28 = ((Ridx0*27)+(Ridx1*9)+(gidx2*1620));
      var val1 = data2_6075[(alu28+7)];
      var val2 = data2_6075[alu28];
      var alu29 = (alu8&alu27);
      var val3 = select(0.0f, data1_251658240[(alu26+-131584)], alu29);
      var val4 = data2_6075[(alu28+1)];
      var val5 = select(0.0f, data1_251658240[(alu26+-131582)], alu29);
      var val6 = data2_6075[(alu28+2)];
      var val7 = select(0.0f, data1_251658240[(alu26+-131074)], (alu6&alu27));
      var val8 = data2_6075[(alu28+3)];
      var val9 = select(0.0f, data1_251658240[(alu26+-131072)], alu27);
      var val10 = data2_6075[(alu28+4)];
      var val11 = select(0.0f, data1_251658240[(alu26+-131070)], alu27);
      var val12 = data2_6075[(alu28+5)];
      var val13 = select(0.0f, data1_251658240[(alu26+-130562)], (alu6&alu3&alu27));
      var val14 = data2_6075[(alu28+6)];
      var alu30 = (alu3&alu27);
      var val15 = select(0.0f, data1_251658240[(alu26+-130560)], alu30);
      var val16 = select(0.0f, data1_251658240[(alu26+-130558)], alu30);
      var val17 = data2_6075[(alu28+8)];
      var val18 = data2_6075[(alu28+405)];
      var val19 = data2_6075[(alu28+406)];
      var val20 = data2_6075[(alu28+407)];
      var val21 = data2_6075[(alu28+408)];
      var val22 = data2_6075[(alu28+409)];
      var val23 = data2_6075[(alu28+410)];
      var val24 = data2_6075[(alu28+411)];
      var val25 = data2_6075[(alu28+412)];
      var val26 = data2_6075[(alu28+413)];
      var val27 = data2_6075[(alu28+810)];
      var val28 = data2_6075[(alu28+811)];
      var val29 = data2_6075[(alu28+812)];
      var val30 = data2_6075[(alu28+813)];
      var val31 = data2_6075[(alu28+814)];
      var val32 = data2_6075[(alu28+815)];
      var val33 = data2_6075[(alu28+816)];
      var val34 = data2_6075[(alu28+817)];
      var val35 = data2_6075[(alu28+818)];
      var val36 = data2_6075[(alu28+1215)];
      var val37 = data2_6075[(alu28+1216)];
      var val38 = data2_6075[(alu28+1217)];
      var val39 = data2_6075[(alu28+1218)];
      var val40 = data2_6075[(alu28+1219)];
      var val41 = data2_6075[(alu28+1220)];
      var val42 = data2_6075[(alu28+1221)];
      var val43 = data2_6075[(alu28+1222)];
      var val44 = data2_6075[(alu28+1223)];
      var val45 = select(0.0f, data1_251658240[(alu26+-131585)], (alu7&alu8&alu27));
      var val46 = select(0.0f, data1_251658240[(alu26+-131583)], alu29);
      var val47 = select(0.0f, data1_251658240[(alu26+-131581)], alu29);
      var val48 = select(0.0f, data1_251658240[(alu26+-131073)], (alu7&alu27));
      var val49 = select(0.0f, data1_251658240[(alu26+-131071)], alu27);
      var val50 = select(0.0f, data1_251658240[(alu26+-131069)], alu27);
      var val51 = select(0.0f, data1_251658240[(alu26+-130561)], (alu7&alu3&alu27));
      var val52 = select(0.0f, data1_251658240[(alu26+-130559)], alu30);
      var val53 = select(0.0f, data1_251658240[(alu26+-130557)], alu30);
      var val54 = select(0.0f, data1_251658240[(alu26+-131580)], (alu4&alu8&alu27));
      var val55 = select(0.0f, data1_251658240[(alu26+-131068)], (alu4&alu27));
      var val56 = select(0.0f, data1_251658240[(alu26+-130556)], (alu4&alu3&alu27));
      var val57 = select(0.0f, data1_251658240[(alu26+-131579)], (alu5&alu8&alu27));
      var val58 = select(0.0f, data1_251658240[(alu26+-131067)], (alu5&alu27));
      var val59 = select(0.0f, data1_251658240[(alu26+-130555)], (alu5&alu3&alu27));
      acc0[0] = (acc0[0]+(val0*val2)+(val3*val4)+(val5*val6)+(val7*val8)+(val9*val10)+(val11*val12)+(val13*val14)+(val15*val1)+(val16*val17));
      acc0[1] = (acc0[1]+(val0*val18)+(val3*val19)+(val5*val20)+(val7*val21)+(val9*val22)+(val11*val23)+(val13*val24)+(val15*val25)+(val16*val26));
      acc0[2] = (acc0[2]+(val0*val27)+(val3*val28)+(val5*val29)+(val7*val30)+(val9*val31)+(val11*val32)+(val13*val33)+(val15*val34)+(val16*val35));
      acc0[3] = (acc0[3]+(val0*val36)+(val3*val37)+(val5*val38)+(val7*val39)+(val9*val40)+(val11*val41)+(val13*val42)+(val15*val43)+(val16*val44));
      acc0[4] = (acc0[4]+(val45*val2)+(val46*val4)+(val47*val6)+(val48*val8)+(val49*val10)+(val50*val12)+(val51*val14)+(val52*val1)+(val53*val17));
      acc0[5] = (acc0[5]+(val45*val18)+(val46*val19)+(val47*val20)+(val48*val21)+(val49*val22)+(val50*val23)+(val51*val24)+(val52*val25)+(val53*val26));
      acc0[6] = (acc0[6]+(val45*val27)+(val46*val28)+(val47*val29)+(val48*val30)+(val49*val31)+(val50*val32)+(val51*val33)+(val52*val34)+(val53*val35));
      acc0[7] = (acc0[7]+(val45*val36)+(val46*val37)+(val47*val38)+(val48*val39)+(val49*val40)+(val50*val41)+(val51*val42)+(val52*val43)+(val53*val44));
      acc0[8] = (acc0[8]+(val3*val2)+(val5*val4)+(val54*val6)+(val9*val8)+(val11*val10)+(val55*val12)+(val15*val14)+(val16*val1)+(val56*val17));
      acc0[9] = (acc0[9]+(val3*val18)+(val5*val19)+(val54*val20)+(val9*val21)+(val11*val22)+(val55*val23)+(val15*val24)+(val16*val25)+(val56*val26));
      acc0[10] = (acc0[10]+(val3*val27)+(val5*val28)+(val54*val29)+(val9*val30)+(val11*val31)+(val55*val32)+(val15*val33)+(val16*val34)+(val56*val35));
      acc0[11] = (acc0[11]+(val3*val36)+(val5*val37)+(val54*val38)+(val9*val39)+(val11*val40)+(val55*val41)+(val15*val42)+(val16*val43)+(val56*val44));
      acc0[12] = (acc0[12]+(val46*val2)+(val47*val4)+(val57*val6)+(val49*val8)+(val50*val10)+(val58*val12)+(val52*val14)+(val53*val1)+(val59*val17));
      acc0[13] = (acc0[13]+(val46*val18)+(val47*val19)+(val57*val20)+(val49*val21)+(val50*val22)+(val58*val23)+(val52*val24)+(val53*val25)+(val59*val26));
      acc0[14] = (acc0[14]+(val46*val27)+(val47*val28)+(val57*val29)+(val49*val30)+(val50*val31)+(val58*val32)+(val52*val33)+(val53*val34)+(val59*val35));
      acc0[15] = (acc0[15]+(val46*val36)+(val47*val37)+(val57*val38)+(val49*val39)+(val50*val40)+(val58*val41)+(val52*val42)+(val53*val43)+(val59*val44));
    }
  }
  var alu49 = (alu2+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<26u)));
  data0_134217728[alu49] = acc0[0];
  data0_134217728[(alu49+1)] = acc0[4];
  data0_134217728[(alu49+2)] = acc0[8];
  data0_134217728[(alu49+3)] = acc0[12];
  data0_134217728[(alu49+16777216)] = acc0[1];
  data0_134217728[(alu49+16777217)] = acc0[5];
  data0_134217728[(alu49+16777218)] = acc0[9];
  data0_134217728[(alu49+16777219)] = acc0[13];
  data0_134217728[(alu49+33554432)] = acc0[2];
  data0_134217728[(alu49+33554433)] = acc0[6];
  data0_134217728[(alu49+33554434)] = acc0[10];
  data0_134217728[(alu49+33554435)] = acc0[14];
  data0_134217728[(alu49+50331648)] = acc0[3];
  data0_134217728[(alu49+50331649)] = acc0[7];
  data0_134217728[(alu49+50331650)] = acc0[11];
  data0_134217728[(alu49+50331651)] = acc0[15];
}`;

const r_7_256_32_4_8_16_4_15_3_3_3n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_117440512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>(lidx1);
  var cast2 = bitcast<u32>((gidx0>>2u));
  var cast3 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx0+bitcast<i32>((cast2<<3u)));
  var alu1 = (bitcast<i32>((cast1<<2u))+bitcast<i32>((cast3<<6u)));
  var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((cast2<<11u))+alu1);
  var alu3 = (alu0<254);
  var alu4 = ((lidx1+bitcast<i32>((cast3<<4u)))<63);
  var alu5 = (alu1<251);
  var alu6 = (0<(bitcast<i32>((cast1<<1u))+bitcast<i32>((cast3<<5u))));
  var alu7 = (0<alu1);
  var alu8 = (1<alu0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast4 = bitcast<u32>(Ridx1);
      var alu13 = (gidx1+bitcast<i32>((cast4<<1u)));
      var alu14 = (alu2+cast0+bitcast<i32>((cast4<<17u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu15 = ((1<alu13)&(alu13<258));
      var val0 = select(0.0f, data1_251658240[(alu14+-131586)], (alu6&alu8&alu15));
      var alu16 = ((Ridx0*27)+(Ridx1*9)+(gidx2*405));
      var val1 = data2_6075[(alu16+3240)];
      var alu17 = (alu8&alu15);
      var val2 = select(0.0f, data1_251658240[(alu14+-131584)], alu17);
      var val3 = data2_6075[(alu16+3241)];
      var val4 = select(0.0f, data1_251658240[(alu14+-131582)], alu17);
      var val5 = data2_6075[(alu16+3242)];
      var val6 = select(0.0f, data1_251658240[(alu14+-131074)], (alu6&alu15));
      var val7 = data2_6075[(alu16+3243)];
      var val8 = select(0.0f, data1_251658240[(alu14+-131072)], alu15);
      var val9 = data2_6075[(alu16+3244)];
      var val10 = select(0.0f, data1_251658240[(alu14+-131070)], alu15);
      var val11 = data2_6075[(alu16+3245)];
      var val12 = select(0.0f, data1_251658240[(alu14+-130562)], (alu6&alu3&alu15));
      var val13 = data2_6075[(alu16+3246)];
      var alu18 = (alu3&alu15);
      var val14 = select(0.0f, data1_251658240[(alu14+-130560)], alu18);
      var val15 = data2_6075[(alu16+3247)];
      var val16 = select(0.0f, data1_251658240[(alu14+-131583)], alu17);
      var val17 = select(0.0f, data1_251658240[(alu14+-131580)], (alu4&alu8&alu15));
      var val18 = select(0.0f, data1_251658240[(alu14+-131073)], (alu7&alu15));
      var val19 = select(0.0f, data1_251658240[(alu14+-131071)], alu15);
      var val20 = select(0.0f, data1_251658240[(alu14+-131069)], alu15);
      var val21 = select(0.0f, data1_251658240[(alu14+-131068)], (alu4&alu15));
      var val22 = select(0.0f, data1_251658240[(alu14+-130561)], (alu7&alu3&alu15));
      var val23 = select(0.0f, data1_251658240[(alu14+-130559)], alu18);
      var val24 = select(0.0f, data1_251658240[(alu14+-130558)], alu18);
      var val25 = data2_6075[(alu16+3248)];
      var val26 = select(0.0f, data1_251658240[(alu14+-131585)], (alu7&alu8&alu15));
      var val27 = select(0.0f, data1_251658240[(alu14+-131581)], alu17);
      var val28 = select(0.0f, data1_251658240[(alu14+-130557)], alu18);
      var val29 = select(0.0f, data1_251658240[(alu14+-130556)], (alu4&alu3&alu15));
      var val30 = select(0.0f, data1_251658240[(alu14+-131579)], (alu5&alu8&alu15));
      var val31 = select(0.0f, data1_251658240[(alu14+-131067)], (alu5&alu15));
      var val32 = select(0.0f, data1_251658240[(alu14+-130555)], (alu5&alu3&alu15));
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val24*val25));
      acc0[1] = (acc0[1]+(val26*val1)+(val16*val3)+(val27*val5)+(val18*val7)+(val19*val9)+(val20*val11)+(val22*val13)+(val23*val15)+(val28*val25));
      acc0[2] = (acc0[2]+(val2*val1)+(val4*val3)+(val17*val5)+(val8*val7)+(val10*val9)+(val21*val11)+(val14*val13)+(val24*val15)+(val29*val25));
      acc0[3] = (acc0[3]+(val16*val1)+(val27*val3)+(val30*val5)+(val19*val7)+(val20*val9)+(val31*val11)+(val23*val13)+(val28*val15)+(val32*val25));
    }
  }
  var alu25 = (alu2+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<24u)));
  data0_117440512[alu25] = acc0[0];
  data0_117440512[(alu25+1)] = acc0[1];
  data0_117440512[(alu25+2)] = acc0[2];
  data0_117440512[(alu25+3)] = acc0[3];
}`;

const r_2_256_32_4_8_16_4_4_15_3_3_3n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_134217728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var alu0 = (gidx0>>2u);
  var cast1 = bitcast<u32>(alu0);
  var alu1 = (gidx0&3);
  var cast2 = bitcast<u32>(alu1);
  var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast2<<6u)));
  var alu3 = ((lidx0+bitcast<i32>((cast1<<3u)))<255);
  var alu4 = ((lidx1+bitcast<i32>((cast2<<4u)))<63);
  var alu5 = (0<(lidx0+alu0));
  var alu6 = (0<(lidx1+alu1));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var alu23 = (gidx1+Ridx1);
      var alu24 = (alu2+cast0+bitcast<i32>((bitcast<u32>(Ridx1)<<16u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu25 = ((0<alu23)&(alu23<257));
      var val0 = select(0.0f, data1_251658240[(alu24+-65793)], (alu6&alu5&alu25));
      var alu26 = ((Ridx0*27)+(Ridx1*9)+(gidx2*1620));
      var val1 = data2_6075[(alu26+1)];
      var val2 = data2_6075[(alu26+2)];
      var val3 = data2_6075[alu26];
      var alu27 = (alu5&alu25);
      var val4 = select(0.0f, data1_251658240[(alu24+-65792)], alu27);
      var val5 = select(0.0f, data1_251658240[(alu24+-65791)], alu27);
      var val6 = select(0.0f, data1_251658240[(alu24+-65537)], (alu6&alu25));
      var val7 = data2_6075[(alu26+3)];
      var val8 = select(0.0f, data1_251658240[(alu24+-65536)], alu25);
      var val9 = data2_6075[(alu26+4)];
      var val10 = select(0.0f, data1_251658240[(alu24+-65535)], alu25);
      var val11 = data2_6075[(alu26+5)];
      var val12 = select(0.0f, data1_251658240[(alu24+-65281)], (alu6&alu3&alu25));
      var val13 = data2_6075[(alu26+6)];
      var alu28 = (alu3&alu25);
      var val14 = select(0.0f, data1_251658240[(alu24+-65280)], alu28);
      var val15 = data2_6075[(alu26+7)];
      var val16 = select(0.0f, data1_251658240[(alu24+-65279)], alu28);
      var val17 = data2_6075[(alu26+8)];
      var val18 = data2_6075[(alu26+405)];
      var val19 = data2_6075[(alu26+406)];
      var val20 = data2_6075[(alu26+407)];
      var val21 = data2_6075[(alu26+408)];
      var val22 = data2_6075[(alu26+409)];
      var val23 = data2_6075[(alu26+410)];
      var val24 = data2_6075[(alu26+411)];
      var val25 = data2_6075[(alu26+412)];
      var val26 = data2_6075[(alu26+413)];
      var val27 = data2_6075[(alu26+810)];
      var val28 = data2_6075[(alu26+811)];
      var val29 = data2_6075[(alu26+812)];
      var val30 = data2_6075[(alu26+813)];
      var val31 = data2_6075[(alu26+814)];
      var val32 = data2_6075[(alu26+815)];
      var val33 = data2_6075[(alu26+816)];
      var val34 = data2_6075[(alu26+817)];
      var val35 = data2_6075[(alu26+818)];
      var val36 = data2_6075[(alu26+1215)];
      var val37 = data2_6075[(alu26+1216)];
      var val38 = data2_6075[(alu26+1217)];
      var val39 = data2_6075[(alu26+1218)];
      var val40 = data2_6075[(alu26+1219)];
      var val41 = data2_6075[(alu26+1220)];
      var val42 = data2_6075[(alu26+1221)];
      var val43 = data2_6075[(alu26+1222)];
      var val44 = data2_6075[(alu26+1223)];
      var val45 = select(0.0f, data1_251658240[(alu24+-65790)], alu27);
      var val46 = select(0.0f, data1_251658240[(alu24+-65534)], alu25);
      var val47 = select(0.0f, data1_251658240[(alu24+-65278)], alu28);
      var val48 = select(0.0f, data1_251658240[(alu24+-65789)], alu27);
      var val49 = select(0.0f, data1_251658240[(alu24+-65533)], alu25);
      var val50 = select(0.0f, data1_251658240[(alu24+-65277)], alu28);
      var val51 = select(0.0f, data1_251658240[(alu24+-65788)], (alu4&alu5&alu25));
      var val52 = select(0.0f, data1_251658240[(alu24+-65532)], (alu4&alu25));
      var val53 = select(0.0f, data1_251658240[(alu24+-65276)], (alu4&alu3&alu25));
      acc0[0] = (acc0[0]+(val0*val3)+(val4*val1)+(val5*val2)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val0*val18)+(val4*val19)+(val5*val20)+(val6*val21)+(val8*val22)+(val10*val23)+(val12*val24)+(val14*val25)+(val16*val26));
      acc0[2] = (acc0[2]+(val0*val27)+(val4*val28)+(val5*val29)+(val6*val30)+(val8*val31)+(val10*val32)+(val12*val33)+(val14*val34)+(val16*val35));
      acc0[3] = (acc0[3]+(val0*val36)+(val4*val37)+(val5*val38)+(val6*val39)+(val8*val40)+(val10*val41)+(val12*val42)+(val14*val43)+(val16*val44));
      acc0[4] = (acc0[4]+(val4*val3)+(val5*val1)+(val45*val2)+(val8*val7)+(val10*val9)+(val46*val11)+(val14*val13)+(val16*val15)+(val47*val17));
      acc0[5] = (acc0[5]+(val4*val18)+(val5*val19)+(val45*val20)+(val8*val21)+(val10*val22)+(val46*val23)+(val14*val24)+(val16*val25)+(val47*val26));
      acc0[6] = (acc0[6]+(val4*val27)+(val5*val28)+(val45*val29)+(val8*val30)+(val10*val31)+(val46*val32)+(val14*val33)+(val16*val34)+(val47*val35));
      acc0[7] = (acc0[7]+(val4*val36)+(val5*val37)+(val45*val38)+(val8*val39)+(val10*val40)+(val46*val41)+(val14*val42)+(val16*val43)+(val47*val44));
      acc0[8] = (acc0[8]+(val5*val3)+(val45*val1)+(val48*val2)+(val10*val7)+(val46*val9)+(val49*val11)+(val16*val13)+(val47*val15)+(val50*val17));
      acc0[9] = (acc0[9]+(val5*val18)+(val45*val19)+(val48*val20)+(val10*val21)+(val46*val22)+(val49*val23)+(val16*val24)+(val47*val25)+(val50*val26));
      acc0[10] = (acc0[10]+(val5*val27)+(val45*val28)+(val48*val29)+(val10*val30)+(val46*val31)+(val49*val32)+(val16*val33)+(val47*val34)+(val50*val35));
      acc0[11] = (acc0[11]+(val5*val36)+(val45*val37)+(val48*val38)+(val10*val39)+(val46*val40)+(val49*val41)+(val16*val42)+(val47*val43)+(val50*val44));
      acc0[12] = (acc0[12]+(val45*val3)+(val48*val1)+(val51*val2)+(val46*val7)+(val49*val9)+(val52*val11)+(val47*val13)+(val50*val15)+(val53*val17));
      acc0[13] = (acc0[13]+(val45*val18)+(val48*val19)+(val51*val20)+(val46*val21)+(val49*val22)+(val52*val23)+(val47*val24)+(val50*val25)+(val53*val26));
      acc0[14] = (acc0[14]+(val45*val27)+(val48*val28)+(val51*val29)+(val46*val30)+(val49*val31)+(val52*val32)+(val47*val33)+(val50*val34)+(val53*val35));
      acc0[15] = (acc0[15]+(val45*val36)+(val48*val37)+(val51*val38)+(val46*val39)+(val49*val40)+(val52*val41)+(val47*val42)+(val50*val43)+(val53*val44));
    }
  }
  var alu47 = (alu2+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<26u)));
  data0_134217728[alu47] = acc0[0];
  data0_134217728[(alu47+1)] = acc0[4];
  data0_134217728[(alu47+2)] = acc0[8];
  data0_134217728[(alu47+3)] = acc0[12];
  data0_134217728[(alu47+16777216)] = acc0[1];
  data0_134217728[(alu47+16777217)] = acc0[5];
  data0_134217728[(alu47+16777218)] = acc0[9];
  data0_134217728[(alu47+16777219)] = acc0[13];
  data0_134217728[(alu47+33554432)] = acc0[2];
  data0_134217728[(alu47+33554433)] = acc0[6];
  data0_134217728[(alu47+33554434)] = acc0[10];
  data0_134217728[(alu47+33554435)] = acc0[14];
  data0_134217728[(alu47+50331648)] = acc0[3];
  data0_134217728[(alu47+50331649)] = acc0[7];
  data0_134217728[(alu47+50331650)] = acc0[11];
  data0_134217728[(alu47+50331651)] = acc0[15];
}`;

const r_7_256_32_4_8_16_4_15_3_3_3n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_117440512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var alu0 = (gidx0>>2u);
  var cast1 = bitcast<u32>(alu0);
  var alu1 = (gidx0&3);
  var cast2 = bitcast<u32>(alu1);
  var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast2<<6u)));
  var alu3 = ((lidx0+bitcast<i32>((cast1<<3u)))<255);
  var alu4 = ((lidx1+bitcast<i32>((cast2<<4u)))<63);
  var alu5 = (0<(lidx0+alu0));
  var alu6 = (0<(lidx1+alu1));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var alu11 = (gidx1+Ridx1);
      var alu12 = (alu2+cast0+bitcast<i32>((bitcast<u32>(Ridx1)<<16u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu13 = ((0<alu11)&(alu11<257));
      var val0 = select(0.0f, data1_251658240[(alu12+-65793)], (alu6&alu5&alu13));
      var alu14 = ((Ridx0*27)+(Ridx1*9)+(gidx2*405));
      var val1 = data2_6075[(alu14+3240)];
      var alu15 = (alu5&alu13);
      var val2 = select(0.0f, data1_251658240[(alu12+-65792)], alu15);
      var val3 = data2_6075[(alu14+3241)];
      var val4 = select(0.0f, data1_251658240[(alu12+-65791)], alu15);
      var val5 = data2_6075[(alu14+3242)];
      var val6 = select(0.0f, data1_251658240[(alu12+-65537)], (alu6&alu13));
      var val7 = data2_6075[(alu14+3243)];
      var val8 = select(0.0f, data1_251658240[(alu12+-65536)], alu13);
      var val9 = data2_6075[(alu14+3244)];
      var val10 = select(0.0f, data1_251658240[(alu12+-65535)], alu13);
      var val11 = data2_6075[(alu14+3245)];
      var val12 = select(0.0f, data1_251658240[(alu12+-65281)], (alu6&alu3&alu13));
      var val13 = data2_6075[(alu14+3246)];
      var val14 = select(0.0f, data1_251658240[(alu12+-65789)], alu15);
      var val15 = select(0.0f, data1_251658240[(alu12+-65534)], alu13);
      var alu16 = (alu3&alu13);
      var val16 = select(0.0f, data1_251658240[(alu12+-65280)], alu16);
      var val17 = data2_6075[(alu14+3247)];
      var val18 = select(0.0f, data1_251658240[(alu12+-65279)], alu16);
      var val19 = data2_6075[(alu14+3248)];
      var val20 = select(0.0f, data1_251658240[(alu12+-65790)], alu15);
      var val21 = select(0.0f, data1_251658240[(alu12+-65278)], alu16);
      var val22 = select(0.0f, data1_251658240[(alu12+-65533)], alu13);
      var val23 = select(0.0f, data1_251658240[(alu12+-65277)], alu16);
      var val24 = select(0.0f, data1_251658240[(alu12+-65788)], (alu4&alu5&alu13));
      var val25 = select(0.0f, data1_251658240[(alu12+-65532)], (alu4&alu13));
      var val26 = select(0.0f, data1_251658240[(alu12+-65276)], (alu4&alu3&alu13));
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val16*val17)+(val18*val19));
      acc0[1] = (acc0[1]+(val2*val1)+(val4*val3)+(val20*val5)+(val8*val7)+(val10*val9)+(val15*val11)+(val16*val13)+(val18*val17)+(val21*val19));
      acc0[2] = (acc0[2]+(val4*val1)+(val20*val3)+(val14*val5)+(val10*val7)+(val15*val9)+(val22*val11)+(val18*val13)+(val21*val17)+(val23*val19));
      acc0[3] = (acc0[3]+(val20*val1)+(val14*val3)+(val24*val5)+(val15*val7)+(val22*val9)+(val25*val11)+(val21*val13)+(val23*val17)+(val26*val19));
    }
  }
  var alu23 = (alu2+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<24u)));
  data0_117440512[alu23] = acc0[0];
  data0_117440512[(alu23+1)] = acc0[1];
  data0_117440512[(alu23+2)] = acc0[2];
  data0_117440512[(alu23+3)] = acc0[3];
}`;

const r_2_256_32_4_8_16_4_4_15_3_3_3n4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_134217728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx1+bitcast<i32>((cast1<<4u)));
  var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>((gidx0>>2u))<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast1<<6u)));
  var alu2 = (gidx0<120);
  var alu3 = (alu0<60);
  var alu4 = (3<alu0);
  var alu5 = (7<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast2 = bitcast<u32>(Ridx1);
      var alu22 = (gidx1+bitcast<i32>((cast2<<4u)));
      var alu23 = (alu1+cast0+bitcast<i32>((cast2<<20u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu24 = ((15<alu22)&(alu22<272));
      var alu25 = (alu4&alu5&alu24);
      var val0 = select(0.0f, data1_251658240[(alu23+-1052688)], alu25);
      var alu26 = ((Ridx0*27)+(Ridx1*9)+(gidx2*1620));
      var val1 = data2_6075[alu26];
      var alu27 = (alu5&alu24);
      var val2 = select(0.0f, data1_251658240[(alu23+-1052672)], alu27);
      var val3 = data2_6075[(alu26+1)];
      var alu28 = (alu3&alu5&alu24);
      var val4 = select(0.0f, data1_251658240[(alu23+-1052656)], alu28);
      var val5 = data2_6075[(alu26+2)];
      var alu29 = (alu4&alu24);
      var val6 = select(0.0f, data1_251658240[(alu23+-1048592)], alu29);
      var val7 = data2_6075[(alu26+3)];
      var val8 = select(0.0f, data1_251658240[(alu23+-1048576)], alu24);
      var val9 = data2_6075[(alu26+4)];
      var alu30 = (alu3&alu24);
      var val10 = select(0.0f, data1_251658240[(alu23+-1048560)], alu30);
      var val11 = data2_6075[(alu26+5)];
      var alu31 = (alu4&alu2&alu24);
      var val12 = select(0.0f, data1_251658240[(alu23+-1044496)], alu31);
      var val13 = data2_6075[(alu26+6)];
      var alu32 = (alu2&alu24);
      var val14 = select(0.0f, data1_251658240[(alu23+-1044480)], alu32);
      var val15 = data2_6075[(alu26+7)];
      var alu33 = (alu3&alu2&alu24);
      var val16 = select(0.0f, data1_251658240[(alu23+-1044464)], alu33);
      var val17 = data2_6075[(alu26+8)];
      var val18 = data2_6075[(alu26+405)];
      var val19 = data2_6075[(alu26+406)];
      var val20 = data2_6075[(alu26+407)];
      var val21 = data2_6075[(alu26+408)];
      var val22 = data2_6075[(alu26+409)];
      var val23 = data2_6075[(alu26+410)];
      var val24 = data2_6075[(alu26+411)];
      var val25 = data2_6075[(alu26+412)];
      var val26 = data2_6075[(alu26+413)];
      var val27 = data2_6075[(alu26+810)];
      var val28 = data2_6075[(alu26+811)];
      var val29 = data2_6075[(alu26+812)];
      var val30 = data2_6075[(alu26+813)];
      var val31 = data2_6075[(alu26+814)];
      var val32 = data2_6075[(alu26+815)];
      var val33 = data2_6075[(alu26+816)];
      var val34 = data2_6075[(alu26+817)];
      var val35 = data2_6075[(alu26+818)];
      var val36 = data2_6075[(alu26+1215)];
      var val37 = data2_6075[(alu26+1216)];
      var val38 = data2_6075[(alu26+1217)];
      var val39 = data2_6075[(alu26+1218)];
      var val40 = data2_6075[(alu26+1219)];
      var val41 = data2_6075[(alu26+1220)];
      var val42 = data2_6075[(alu26+1221)];
      var val43 = data2_6075[(alu26+1222)];
      var val44 = data2_6075[(alu26+1223)];
      var val45 = select(0.0f, data1_251658240[(alu23+-1052687)], alu25);
      var val46 = select(0.0f, data1_251658240[(alu23+-1048559)], alu30);
      var val47 = select(0.0f, data1_251658240[(alu23+-1044495)], alu31);
      var val48 = select(0.0f, data1_251658240[(alu23+-1044479)], alu32);
      var val49 = select(0.0f, data1_251658240[(alu23+-1044463)], alu33);
      var val50 = select(0.0f, data1_251658240[(alu23+-1052686)], alu25);
      var val51 = select(0.0f, data1_251658240[(alu23+-1048590)], alu29);
      var val52 = select(0.0f, data1_251658240[(alu23+-1048574)], alu24);
      var val53 = select(0.0f, data1_251658240[(alu23+-1048558)], alu30);
      var val54 = select(0.0f, data1_251658240[(alu23+-1044494)], alu31);
      var val55 = select(0.0f, data1_251658240[(alu23+-1044478)], alu32);
      var val56 = select(0.0f, data1_251658240[(alu23+-1044462)], alu33);
      var val57 = select(0.0f, data1_251658240[(alu23+-1052685)], alu25);
      var val58 = select(0.0f, data1_251658240[(alu23+-1052671)], alu27);
      var val59 = select(0.0f, data1_251658240[(alu23+-1052670)], alu27);
      var val60 = select(0.0f, data1_251658240[(alu23+-1052669)], alu27);
      var val61 = select(0.0f, data1_251658240[(alu23+-1052655)], alu28);
      var val62 = select(0.0f, data1_251658240[(alu23+-1052654)], alu28);
      var val63 = select(0.0f, data1_251658240[(alu23+-1052653)], alu28);
      var val64 = select(0.0f, data1_251658240[(alu23+-1048591)], alu29);
      var val65 = select(0.0f, data1_251658240[(alu23+-1048589)], alu29);
      var val66 = select(0.0f, data1_251658240[(alu23+-1048575)], alu24);
      var val67 = select(0.0f, data1_251658240[(alu23+-1048573)], alu24);
      var val68 = select(0.0f, data1_251658240[(alu23+-1048557)], alu30);
      var val69 = select(0.0f, data1_251658240[(alu23+-1044493)], alu31);
      var val70 = select(0.0f, data1_251658240[(alu23+-1044477)], alu32);
      var val71 = select(0.0f, data1_251658240[(alu23+-1044461)], alu33);
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val0*val18)+(val2*val19)+(val4*val20)+(val6*val21)+(val8*val22)+(val10*val23)+(val12*val24)+(val14*val25)+(val16*val26));
      acc0[2] = (acc0[2]+(val0*val27)+(val2*val28)+(val4*val29)+(val6*val30)+(val8*val31)+(val10*val32)+(val12*val33)+(val14*val34)+(val16*val35));
      acc0[3] = (acc0[3]+(val0*val36)+(val2*val37)+(val4*val38)+(val6*val39)+(val8*val40)+(val10*val41)+(val12*val42)+(val14*val43)+(val16*val44));
      acc0[4] = (acc0[4]+(val45*val1)+(val58*val3)+(val61*val5)+(val64*val7)+(val66*val9)+(val46*val11)+(val47*val13)+(val48*val15)+(val49*val17));
      acc0[5] = (acc0[5]+(val45*val18)+(val58*val19)+(val61*val20)+(val64*val21)+(val66*val22)+(val46*val23)+(val47*val24)+(val48*val25)+(val49*val26));
      acc0[6] = (acc0[6]+(val45*val27)+(val58*val28)+(val61*val29)+(val64*val30)+(val66*val31)+(val46*val32)+(val47*val33)+(val48*val34)+(val49*val35));
      acc0[7] = (acc0[7]+(val45*val36)+(val58*val37)+(val61*val38)+(val64*val39)+(val66*val40)+(val46*val41)+(val47*val42)+(val48*val43)+(val49*val44));
      acc0[8] = (acc0[8]+(val50*val1)+(val59*val3)+(val62*val5)+(val51*val7)+(val52*val9)+(val53*val11)+(val54*val13)+(val55*val15)+(val56*val17));
      acc0[9] = (acc0[9]+(val50*val18)+(val59*val19)+(val62*val20)+(val51*val21)+(val52*val22)+(val53*val23)+(val54*val24)+(val55*val25)+(val56*val26));
      acc0[10] = (acc0[10]+(val50*val27)+(val59*val28)+(val62*val29)+(val51*val30)+(val52*val31)+(val53*val32)+(val54*val33)+(val55*val34)+(val56*val35));
      acc0[11] = (acc0[11]+(val50*val36)+(val59*val37)+(val62*val38)+(val51*val39)+(val52*val40)+(val53*val41)+(val54*val42)+(val55*val43)+(val56*val44));
      acc0[12] = (acc0[12]+(val57*val1)+(val60*val3)+(val63*val5)+(val65*val7)+(val67*val9)+(val68*val11)+(val69*val13)+(val70*val15)+(val71*val17));
      acc0[13] = (acc0[13]+(val57*val18)+(val60*val19)+(val63*val20)+(val65*val21)+(val67*val22)+(val68*val23)+(val69*val24)+(val70*val25)+(val71*val26));
      acc0[14] = (acc0[14]+(val57*val27)+(val60*val28)+(val63*val29)+(val65*val30)+(val67*val31)+(val68*val32)+(val69*val33)+(val70*val34)+(val71*val35));
      acc0[15] = (acc0[15]+(val57*val36)+(val60*val37)+(val63*val38)+(val65*val39)+(val67*val40)+(val68*val41)+(val69*val42)+(val70*val43)+(val71*val44));
    }
  }
  var alu52 = (alu1+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<26u)));
  data0_134217728[alu52] = acc0[0];
  data0_134217728[(alu52+1)] = acc0[4];
  data0_134217728[(alu52+2)] = acc0[8];
  data0_134217728[(alu52+3)] = acc0[12];
  data0_134217728[(alu52+16777216)] = acc0[1];
  data0_134217728[(alu52+16777217)] = acc0[5];
  data0_134217728[(alu52+16777218)] = acc0[9];
  data0_134217728[(alu52+16777219)] = acc0[13];
  data0_134217728[(alu52+33554432)] = acc0[2];
  data0_134217728[(alu52+33554433)] = acc0[6];
  data0_134217728[(alu52+33554434)] = acc0[10];
  data0_134217728[(alu52+33554435)] = acc0[14];
  data0_134217728[(alu52+50331648)] = acc0[3];
  data0_134217728[(alu52+50331649)] = acc0[7];
  data0_134217728[(alu52+50331650)] = acc0[11];
  data0_134217728[(alu52+50331651)] = acc0[15];
}`;

const r_7_256_32_4_8_16_4_15_3_3_3n4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_117440512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6075:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 256 */
  var gidx2 = i32(gindex.z); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<16u));
  var cast1 = bitcast<u32>((gidx0&3));
  var alu0 = (lidx1+bitcast<i32>((cast1<<4u)));
  var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>((gidx0>>2u))<<11u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u))+bitcast<i32>((cast1<<6u)));
  var alu2 = (gidx0<120);
  var alu3 = (alu0<60);
  var alu4 = (3<alu0);
  var alu5 = (7<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 15; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var cast2 = bitcast<u32>(Ridx1);
      var alu10 = (gidx1+bitcast<i32>((cast2<<4u)));
      var alu11 = (alu1+cast0+bitcast<i32>((cast2<<20u))+bitcast<i32>((bitcast<u32>(Ridx0)<<24u)));
      var alu12 = ((15<alu10)&(alu10<272));
      var alu13 = (alu4&alu5&alu12);
      var val0 = select(0.0f, data1_251658240[(alu11+-1052688)], alu13);
      var alu14 = ((Ridx0*27)+(Ridx1*9)+(gidx2*405));
      var val1 = data2_6075[(alu14+3240)];
      var alu15 = (alu5&alu12);
      var val2 = select(0.0f, data1_251658240[(alu11+-1052672)], alu15);
      var val3 = data2_6075[(alu14+3241)];
      var alu16 = (alu3&alu5&alu12);
      var val4 = select(0.0f, data1_251658240[(alu11+-1052656)], alu16);
      var val5 = data2_6075[(alu14+3242)];
      var alu17 = (alu4&alu12);
      var val6 = select(0.0f, data1_251658240[(alu11+-1048592)], alu17);
      var val7 = data2_6075[(alu14+3243)];
      var val8 = select(0.0f, data1_251658240[(alu11+-1048576)], alu12);
      var val9 = data2_6075[(alu14+3244)];
      var alu18 = (alu3&alu12);
      var val10 = select(0.0f, data1_251658240[(alu11+-1048560)], alu18);
      var val11 = data2_6075[(alu14+3245)];
      var alu19 = (alu4&alu2&alu12);
      var val12 = select(0.0f, data1_251658240[(alu11+-1044496)], alu19);
      var val13 = data2_6075[(alu14+3246)];
      var alu20 = (alu2&alu12);
      var val14 = select(0.0f, data1_251658240[(alu11+-1044480)], alu20);
      var val15 = data2_6075[(alu14+3247)];
      var alu21 = (alu3&alu2&alu12);
      var val16 = select(0.0f, data1_251658240[(alu11+-1044464)], alu21);
      var val17 = data2_6075[(alu14+3248)];
      var val18 = select(0.0f, data1_251658240[(alu11+-1052687)], alu13);
      var val19 = select(0.0f, data1_251658240[(alu11+-1052686)], alu13);
      var val20 = select(0.0f, data1_251658240[(alu11+-1052671)], alu15);
      var val21 = select(0.0f, data1_251658240[(alu11+-1052670)], alu15);
      var val22 = select(0.0f, data1_251658240[(alu11+-1052655)], alu16);
      var val23 = select(0.0f, data1_251658240[(alu11+-1052654)], alu16);
      var val24 = select(0.0f, data1_251658240[(alu11+-1048591)], alu17);
      var val25 = select(0.0f, data1_251658240[(alu11+-1048590)], alu17);
      var val26 = select(0.0f, data1_251658240[(alu11+-1048575)], alu12);
      var val27 = select(0.0f, data1_251658240[(alu11+-1048559)], alu18);
      var val28 = select(0.0f, data1_251658240[(alu11+-1048558)], alu18);
      var val29 = select(0.0f, data1_251658240[(alu11+-1044495)], alu19);
      var val30 = select(0.0f, data1_251658240[(alu11+-1044479)], alu20);
      var val31 = select(0.0f, data1_251658240[(alu11+-1044463)], alu21);
      var val32 = select(0.0f, data1_251658240[(alu11+-1052685)], alu13);
      var val33 = select(0.0f, data1_251658240[(alu11+-1052669)], alu15);
      var val34 = select(0.0f, data1_251658240[(alu11+-1052653)], alu16);
      var val35 = select(0.0f, data1_251658240[(alu11+-1048589)], alu17);
      var val36 = select(0.0f, data1_251658240[(alu11+-1048574)], alu12);
      var val37 = select(0.0f, data1_251658240[(alu11+-1048573)], alu12);
      var val38 = select(0.0f, data1_251658240[(alu11+-1044494)], alu19);
      var val39 = select(0.0f, data1_251658240[(alu11+-1044478)], alu20);
      var val40 = select(0.0f, data1_251658240[(alu11+-1044462)], alu21);
      var val41 = select(0.0f, data1_251658240[(alu11+-1048557)], alu18);
      var val42 = select(0.0f, data1_251658240[(alu11+-1044493)], alu19);
      var val43 = select(0.0f, data1_251658240[(alu11+-1044477)], alu20);
      var val44 = select(0.0f, data1_251658240[(alu11+-1044461)], alu21);
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17));
      acc0[1] = (acc0[1]+(val18*val1)+(val20*val3)+(val22*val5)+(val24*val7)+(val26*val9)+(val27*val11)+(val29*val13)+(val30*val15)+(val31*val17));
      acc0[2] = (acc0[2]+(val19*val1)+(val21*val3)+(val23*val5)+(val25*val7)+(val36*val9)+(val28*val11)+(val38*val13)+(val39*val15)+(val40*val17));
      acc0[3] = (acc0[3]+(val32*val1)+(val33*val3)+(val34*val5)+(val35*val7)+(val37*val9)+(val41*val11)+(val42*val13)+(val43*val15)+(val44*val17));
    }
  }
  var alu28 = (alu1+cast0+bitcast<i32>((bitcast<u32>(gidx2)<<24u)));
  data0_117440512[alu28] = acc0[0];
  data0_117440512[(alu28+1)] = acc0[1];
  data0_117440512[(alu28+2)] = acc0[2];
  data0_117440512[(alu28+3)] = acc0[3];
}`;

const r_262144_2_16_4_15 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_33554432:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_251658240:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_30:array<f32>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32768 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<9u))+bitcast<i32>((bitcast<u32>(gidx1)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var val0 = data1_251658240[alu0];
  var lidx0 = i32(lindex.x); /* 2 */
  var alu1 = (lidx0*15);
  var val1 = data2_30[alu1];
  var val2 = data1_251658240[(alu0+2)];
  var val3 = data1_251658240[(alu0+3)];
  var val4 = data1_251658240[(alu0+16777216)];
  var val5 = data2_30[(alu1+1)];
  var val6 = data1_251658240[(alu0+16777218)];
  var val7 = data1_251658240[(alu0+16777219)];
  var val8 = data1_251658240[(alu0+33554432)];
  var val9 = data2_30[(alu1+2)];
  var val10 = data1_251658240[(alu0+33554434)];
  var val11 = data1_251658240[(alu0+33554435)];
  var val12 = data1_251658240[(alu0+50331648)];
  var val13 = data2_30[(alu1+3)];
  var val14 = data1_251658240[(alu0+50331650)];
  var val15 = data1_251658240[(alu0+50331651)];
  var val16 = data1_251658240[(alu0+67108864)];
  var val17 = data2_30[(alu1+4)];
  var val18 = data1_251658240[(alu0+67108866)];
  var val19 = data1_251658240[(alu0+67108867)];
  var val20 = data1_251658240[(alu0+83886080)];
  var val21 = data2_30[(alu1+5)];
  var val22 = data1_251658240[(alu0+83886082)];
  var val23 = data1_251658240[(alu0+83886083)];
  var val24 = data1_251658240[(alu0+100663296)];
  var val25 = data2_30[(alu1+6)];
  var val26 = data1_251658240[(alu0+100663298)];
  var val27 = data1_251658240[(alu0+100663299)];
  var val28 = data1_251658240[(alu0+117440512)];
  var val29 = data2_30[(alu1+7)];
  var val30 = data1_251658240[(alu0+117440514)];
  var val31 = data1_251658240[(alu0+117440515)];
  var val32 = data1_251658240[(alu0+134217728)];
  var val33 = data2_30[(alu1+8)];
  var val34 = data1_251658240[(alu0+134217729)];
  var val35 = data1_251658240[(alu0+134217730)];
  var val36 = data1_251658240[(alu0+134217731)];
  var val37 = data1_251658240[(alu0+150994944)];
  var val38 = data2_30[(alu1+9)];
  var val39 = data1_251658240[(alu0+150994945)];
  var val40 = data1_251658240[(alu0+150994946)];
  var val41 = data1_251658240[(alu0+150994947)];
  var val42 = data1_251658240[(alu0+167772160)];
  var val43 = data2_30[(alu1+10)];
  var val44 = data1_251658240[(alu0+167772161)];
  var val45 = data1_251658240[(alu0+167772162)];
  var val46 = data1_251658240[(alu0+167772163)];
  var val47 = data1_251658240[(alu0+184549376)];
  var val48 = data2_30[(alu1+11)];
  var val49 = data1_251658240[(alu0+184549377)];
  var val50 = data1_251658240[(alu0+184549378)];
  var val51 = data1_251658240[(alu0+184549379)];
  var val52 = data1_251658240[(alu0+201326592)];
  var val53 = data2_30[(alu1+12)];
  var val54 = data1_251658240[(alu0+201326593)];
  var val55 = data1_251658240[(alu0+201326594)];
  var val56 = data1_251658240[(alu0+201326595)];
  var val57 = data1_251658240[(alu0+218103808)];
  var val58 = data2_30[(alu1+13)];
  var val59 = data1_251658240[(alu0+218103809)];
  var val60 = data1_251658240[(alu0+218103810)];
  var val61 = data1_251658240[(alu0+218103811)];
  var val62 = data1_251658240[(alu0+234881024)];
  var val63 = data2_30[(alu1+14)];
  var val64 = data1_251658240[(alu0+1)];
  var val65 = data1_251658240[(alu0+16777217)];
  var val66 = data1_251658240[(alu0+33554433)];
  var val67 = data1_251658240[(alu0+50331649)];
  var val68 = data1_251658240[(alu0+67108865)];
  var val69 = data1_251658240[(alu0+83886081)];
  var val70 = data1_251658240[(alu0+100663297)];
  var val71 = data1_251658240[(alu0+117440513)];
  var val72 = data1_251658240[(alu0+234881025)];
  var val73 = data1_251658240[(alu0+234881026)];
  var val74 = data1_251658240[(alu0+234881027)];
  var alu2 = (alu0+bitcast<i32>((bitcast<u32>(lidx0)<<24u)));
  data0_33554432[alu2] = ((val0*val1)+(val4*val5)+(val8*val9)+(val12*val13)+(val16*val17)+(val20*val21)+(val24*val25)+(val28*val29)+(val32*val33)+(val37*val38)+(val42*val43)+(val47*val48)+(val52*val53)+(val57*val58)+(val62*val63));
  data0_33554432[(alu2+1)] = ((val64*val1)+(val65*val5)+(val66*val9)+(val67*val13)+(val68*val17)+(val69*val21)+(val70*val25)+(val71*val29)+(val34*val33)+(val39*val38)+(val44*val43)+(val49*val48)+(val54*val53)+(val59*val58)+(val72*val63));
  data0_33554432[(alu2+2)] = ((val2*val1)+(val6*val5)+(val10*val9)+(val14*val13)+(val18*val17)+(val22*val21)+(val26*val25)+(val30*val29)+(val35*val33)+(val40*val38)+(val45*val43)+(val50*val48)+(val55*val53)+(val60*val58)+(val73*val63));
  data0_33554432[(alu2+3)] = ((val3*val1)+(val7*val5)+(val11*val9)+(val15*val13)+(val19*val17)+(val23*val21)+(val27*val25)+(val31*val29)+(val36*val33)+(val41*val38)+(val46*val43)+(val51*val48)+(val56*val53)+(val61*val58)+(val74*val63));
}`;

const E_131072_32_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_16777216:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_33554432:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32768 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<9u))+bitcast<i32>((bitcast<u32>(gidx1)<<7u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  var val0 = data1_33554432[alu0];
  var alu1 = (alu0+1);
  var val1 = data1_33554432[alu1];
  var alu2 = (alu0+2);
  var val2 = data1_33554432[alu2];
  var alu3 = (alu0+3);
  var val3 = data1_33554432[alu3];
  var val4 = data1_33554432[(alu0+16777216)];
  var val5 = data1_33554432[(alu0+16777217)];
  var val6 = data1_33554432[(alu0+16777218)];
  var val7 = data1_33554432[(alu0+16777219)];
  var alu4 = select(0.0f,1.0f,(val0<val4));
  var alu5 = select(0.0f,1.0f,(val1<val5));
  var alu6 = select(0.0f,1.0f,(val2<val6));
  var alu7 = select(0.0f,1.0f,(val3<val7));
  data0_16777216[alu0] = alu4;
  data0_16777216[alu1] = alu5;
  data0_16777216[alu2] = alu6;
  data0_16777216[alu3] = alu7;
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 536870912);;
    const input0 = createEmptyBuf(device, 67108864);;
    const buf_1 = createWeightBuf(device, 1620, getTensorBuffer(safetensor, metadata['model.0.weight']));
    const buf_2 = createEmptyBuf(device, 469762048);;
    const buf_3 = createEmptyBuf(device, 3932160);;
    const buf_4 = createEmptyBuf(device, 15360);;
    const buf_5 = createEmptyBuf(device, 60);;
    const buf_6 = createEmptyBuf(device, 60);;
    const buf_7 = createEmptyBuf(device, 1006632960);;
    const buf_8 = createEmptyBuf(device, 536870912);;
    const buf_9 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.3.weight']));
    const buf_10 = createEmptyBuf(device, 469762048);;
    const buf_11 = createEmptyBuf(device, 536870912);;
    const buf_12 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.6.weight']));
    const buf_13 = createEmptyBuf(device, 469762048);;
    const buf_14 = createEmptyBuf(device, 536870912);;
    const buf_15 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.9.weight']));
    const buf_16 = createEmptyBuf(device, 469762048);;
    const buf_17 = createEmptyBuf(device, 536870912);;
    const buf_18 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.12.weight']));
    const buf_19 = createEmptyBuf(device, 469762048);;
    const buf_20 = createEmptyBuf(device, 536870912);;
    const buf_21 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.15.weight']));
    const buf_22 = createEmptyBuf(device, 469762048);;
    const buf_23 = createEmptyBuf(device, 536870912);;
    const buf_24 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.18.weight']));
    const buf_25 = createEmptyBuf(device, 469762048);;
    const buf_26 = createEmptyBuf(device, 536870912);;
    const buf_27 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.21.weight']));
    const buf_28 = createEmptyBuf(device, 469762048);;
    const buf_29 = createEmptyBuf(device, 536870912);;
    const buf_30 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.24.weight']));
    const buf_31 = createEmptyBuf(device, 469762048);;
    const buf_32 = createEmptyBuf(device, 536870912);;
    const buf_33 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.27.weight']));
    const buf_34 = createEmptyBuf(device, 469762048);;
    const buf_35 = createEmptyBuf(device, 536870912);;
    const buf_36 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.30.weight']));
    const buf_37 = createEmptyBuf(device, 469762048);;
    const buf_38 = createEmptyBuf(device, 536870912);;
    const buf_39 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.33.weight']));
    const buf_40 = createEmptyBuf(device, 469762048);;
    const buf_41 = createEmptyBuf(device, 536870912);;
    const buf_42 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.36.weight']));
    const buf_43 = createEmptyBuf(device, 469762048);;
    const buf_44 = createEmptyBuf(device, 536870912);;
    const buf_45 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.39.weight']));
    const buf_46 = createEmptyBuf(device, 469762048);;
    const buf_47 = createEmptyBuf(device, 536870912);;
    const buf_48 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.42.weight']));
    const buf_49 = createEmptyBuf(device, 469762048);;
    const buf_50 = createEmptyBuf(device, 536870912);;
    const buf_51 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.45.weight']));
    const buf_52 = createEmptyBuf(device, 469762048);;
    const buf_53 = createEmptyBuf(device, 536870912);;
    const buf_54 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.48.weight']));
    const buf_55 = createEmptyBuf(device, 469762048);;
    const buf_56 = createEmptyBuf(device, 536870912);;
    const buf_57 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.51.weight']));
    const buf_58 = createEmptyBuf(device, 469762048);;
    const buf_59 = createEmptyBuf(device, 536870912);;
    const buf_60 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.54.weight']));
    const buf_61 = createEmptyBuf(device, 469762048);;
    const buf_62 = createEmptyBuf(device, 536870912);;
    const buf_63 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.57.weight']));
    const buf_64 = createEmptyBuf(device, 469762048);;
    const buf_65 = createEmptyBuf(device, 536870912);;
    const buf_66 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.60.weight']));
    const buf_67 = createEmptyBuf(device, 469762048);;
    const buf_68 = createEmptyBuf(device, 536870912);;
    const buf_69 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.63.weight']));
    const buf_70 = createEmptyBuf(device, 469762048);;
    const buf_71 = createEmptyBuf(device, 536870912);;
    const buf_72 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.66.weight']));
    const buf_73 = createEmptyBuf(device, 469762048);;
    const buf_74 = createEmptyBuf(device, 536870912);;
    const buf_75 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.69.weight']));
    const buf_76 = createEmptyBuf(device, 469762048);;
    const buf_77 = createEmptyBuf(device, 536870912);;
    const buf_78 = createWeightBuf(device, 24300, getTensorBuffer(safetensor, metadata['model.72.weight']));
    const buf_79 = createEmptyBuf(device, 469762048);;
    const buf_80 = createEmptyBuf(device, 134217728);;
    const buf_81 = createWeightBuf(device, 120, getTensorBuffer(safetensor, metadata['model.75.weight']));
    const output0 = createEmptyBuf(device, 67108864);;

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_2_256_32_4_8_16_4_4_3_3_3, r_7_256_32_4_8_16_4_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n4, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n1, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n2, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_2_256_32_4_8_16_4_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_7_256_32_4_8_16_4_15_3_3_3n3, r_10240_32_3_64_4, r_40_32_3_64_4, r_15_16_16, r_5_1024_3_16_4_64_4, r_40_32_3_64_4, r_15_16_16n1, E_5_262144_3_16_4, r_262144_2_16_4_15, E_131072_32_4n1];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_2, input0, buf_1], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_3, buf_0, buf_2], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_3, buf_0, buf_2, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_7, buf_0, buf_2, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_8, buf_7, buf_9], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_3, buf_0, buf_2], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_3, buf_0, buf_2, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_7, buf_0, buf_2, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_10, buf_7, buf_9], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_3, buf_8, buf_10], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_3, buf_8, buf_10, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_7, buf_8, buf_10, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_11, buf_7, buf_12], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_3, buf_8, buf_10], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_3, buf_8, buf_10, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_7, buf_8, buf_10, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_13, buf_7, buf_12], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_3, buf_11, buf_13], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_3, buf_11, buf_13, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_7, buf_11, buf_13, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_14, buf_7, buf_15], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_3, buf_11, buf_13], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_3, buf_11, buf_13, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_7, buf_11, buf_13, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_16, buf_7, buf_15], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_3, buf_14, buf_16], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_3, buf_14, buf_16, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_7, buf_14, buf_16, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_17, buf_7, buf_18], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_3, buf_14, buf_16], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_3, buf_14, buf_16, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_7, buf_14, buf_16, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_19, buf_7, buf_18], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_3, buf_17, buf_19], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_3, buf_17, buf_19, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_7, buf_17, buf_19, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_20, buf_7, buf_21], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_3, buf_17, buf_19], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_3, buf_17, buf_19, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_7, buf_17, buf_19, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_22, buf_7, buf_21], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_3, buf_20, buf_22], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_3, buf_20, buf_22, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_7, buf_20, buf_22, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_23, buf_7, buf_24], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_3, buf_20, buf_22], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_3, buf_20, buf_22, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_7, buf_20, buf_22, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_25, buf_7, buf_24], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_3, buf_23, buf_25], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_3, buf_23, buf_25, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_7, buf_23, buf_25, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_26, buf_7, buf_27], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_3, buf_23, buf_25], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_3, buf_23, buf_25, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_7, buf_23, buf_25, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_28, buf_7, buf_27], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_3, buf_26, buf_28], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_3, buf_26, buf_28, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_7, buf_26, buf_28, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_29, buf_7, buf_30], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_3, buf_26, buf_28], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_3, buf_26, buf_28, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_7, buf_26, buf_28, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_31, buf_7, buf_30], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_3, buf_29, buf_31], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_3, buf_29, buf_31, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_7, buf_29, buf_31, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_32, buf_7, buf_33], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_3, buf_29, buf_31], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_3, buf_29, buf_31, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_7, buf_29, buf_31, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_34, buf_7, buf_33], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_3, buf_32, buf_34], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_3, buf_32, buf_34, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_7, buf_32, buf_34, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_35, buf_7, buf_36], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_3, buf_32, buf_34], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_3, buf_32, buf_34, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_7, buf_32, buf_34, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_37, buf_7, buf_36], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_3, buf_35, buf_37], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_3, buf_35, buf_37, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_7, buf_35, buf_37, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_38, buf_7, buf_39], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [buf_3, buf_35, buf_37], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[171], layouts[171], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[172], layouts[172], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[173], layouts[173], infinityBuf, [buf_3, buf_35, buf_37, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[174], layouts[174], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[175], layouts[175], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[176], layouts[176], infinityBuf, [buf_7, buf_35, buf_37, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[177], layouts[177], infinityBuf, [buf_40, buf_7, buf_39], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[178], layouts[178], infinityBuf, [buf_3, buf_38, buf_40], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[179], layouts[179], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[180], layouts[180], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[181], layouts[181], infinityBuf, [buf_3, buf_38, buf_40, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[182], layouts[182], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[183], layouts[183], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[184], layouts[184], infinityBuf, [buf_7, buf_38, buf_40, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[185], layouts[185], infinityBuf, [buf_41, buf_7, buf_42], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[186], layouts[186], infinityBuf, [buf_3, buf_38, buf_40], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[187], layouts[187], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[188], layouts[188], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[189], layouts[189], infinityBuf, [buf_3, buf_38, buf_40, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[190], layouts[190], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[191], layouts[191], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[192], layouts[192], infinityBuf, [buf_7, buf_38, buf_40, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[193], layouts[193], infinityBuf, [buf_43, buf_7, buf_42], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[194], layouts[194], infinityBuf, [buf_3, buf_41, buf_43], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[195], layouts[195], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[196], layouts[196], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[197], layouts[197], infinityBuf, [buf_3, buf_41, buf_43, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[198], layouts[198], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[199], layouts[199], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[200], layouts[200], infinityBuf, [buf_7, buf_41, buf_43, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[201], layouts[201], infinityBuf, [buf_44, buf_7, buf_45], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[202], layouts[202], infinityBuf, [buf_3, buf_41, buf_43], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[203], layouts[203], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[204], layouts[204], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[205], layouts[205], infinityBuf, [buf_3, buf_41, buf_43, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[206], layouts[206], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[207], layouts[207], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[208], layouts[208], infinityBuf, [buf_7, buf_41, buf_43, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[209], layouts[209], infinityBuf, [buf_46, buf_7, buf_45], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[210], layouts[210], infinityBuf, [buf_3, buf_44, buf_46], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[211], layouts[211], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[212], layouts[212], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[213], layouts[213], infinityBuf, [buf_3, buf_44, buf_46, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[214], layouts[214], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[215], layouts[215], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[216], layouts[216], infinityBuf, [buf_7, buf_44, buf_46, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[217], layouts[217], infinityBuf, [buf_47, buf_7, buf_48], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[218], layouts[218], infinityBuf, [buf_3, buf_44, buf_46], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[219], layouts[219], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[220], layouts[220], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[221], layouts[221], infinityBuf, [buf_3, buf_44, buf_46, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[222], layouts[222], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[223], layouts[223], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[224], layouts[224], infinityBuf, [buf_7, buf_44, buf_46, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[225], layouts[225], infinityBuf, [buf_49, buf_7, buf_48], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[226], layouts[226], infinityBuf, [buf_3, buf_47, buf_49], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[227], layouts[227], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[228], layouts[228], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[229], layouts[229], infinityBuf, [buf_3, buf_47, buf_49, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[230], layouts[230], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[231], layouts[231], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[232], layouts[232], infinityBuf, [buf_7, buf_47, buf_49, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[233], layouts[233], infinityBuf, [buf_50, buf_7, buf_51], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[234], layouts[234], infinityBuf, [buf_3, buf_47, buf_49], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[235], layouts[235], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[236], layouts[236], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[237], layouts[237], infinityBuf, [buf_3, buf_47, buf_49, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[238], layouts[238], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[239], layouts[239], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[240], layouts[240], infinityBuf, [buf_7, buf_47, buf_49, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[241], layouts[241], infinityBuf, [buf_52, buf_7, buf_51], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[242], layouts[242], infinityBuf, [buf_3, buf_50, buf_52], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[243], layouts[243], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[244], layouts[244], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[245], layouts[245], infinityBuf, [buf_3, buf_50, buf_52, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[246], layouts[246], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[247], layouts[247], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[248], layouts[248], infinityBuf, [buf_7, buf_50, buf_52, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[249], layouts[249], infinityBuf, [buf_53, buf_7, buf_54], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[250], layouts[250], infinityBuf, [buf_3, buf_50, buf_52], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[251], layouts[251], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[252], layouts[252], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[253], layouts[253], infinityBuf, [buf_3, buf_50, buf_52, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[254], layouts[254], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[255], layouts[255], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[256], layouts[256], infinityBuf, [buf_7, buf_50, buf_52, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[257], layouts[257], infinityBuf, [buf_55, buf_7, buf_54], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[258], layouts[258], infinityBuf, [buf_3, buf_53, buf_55], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[259], layouts[259], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[260], layouts[260], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[261], layouts[261], infinityBuf, [buf_3, buf_53, buf_55, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[262], layouts[262], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[263], layouts[263], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[264], layouts[264], infinityBuf, [buf_7, buf_53, buf_55, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[265], layouts[265], infinityBuf, [buf_56, buf_7, buf_57], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[266], layouts[266], infinityBuf, [buf_3, buf_53, buf_55], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[267], layouts[267], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[268], layouts[268], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[269], layouts[269], infinityBuf, [buf_3, buf_53, buf_55, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[270], layouts[270], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[271], layouts[271], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[272], layouts[272], infinityBuf, [buf_7, buf_53, buf_55, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[273], layouts[273], infinityBuf, [buf_58, buf_7, buf_57], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[274], layouts[274], infinityBuf, [buf_3, buf_56, buf_58], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[275], layouts[275], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[276], layouts[276], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[277], layouts[277], infinityBuf, [buf_3, buf_56, buf_58, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[278], layouts[278], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[279], layouts[279], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[280], layouts[280], infinityBuf, [buf_7, buf_56, buf_58, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[281], layouts[281], infinityBuf, [buf_59, buf_7, buf_60], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[282], layouts[282], infinityBuf, [buf_3, buf_56, buf_58], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[283], layouts[283], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[284], layouts[284], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[285], layouts[285], infinityBuf, [buf_3, buf_56, buf_58, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[286], layouts[286], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[287], layouts[287], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[288], layouts[288], infinityBuf, [buf_7, buf_56, buf_58, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[289], layouts[289], infinityBuf, [buf_61, buf_7, buf_60], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[290], layouts[290], infinityBuf, [buf_3, buf_59, buf_61], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[291], layouts[291], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[292], layouts[292], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[293], layouts[293], infinityBuf, [buf_3, buf_59, buf_61, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[294], layouts[294], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[295], layouts[295], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[296], layouts[296], infinityBuf, [buf_7, buf_59, buf_61, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[297], layouts[297], infinityBuf, [buf_62, buf_7, buf_63], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[298], layouts[298], infinityBuf, [buf_3, buf_59, buf_61], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[299], layouts[299], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[300], layouts[300], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[301], layouts[301], infinityBuf, [buf_3, buf_59, buf_61, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[302], layouts[302], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[303], layouts[303], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[304], layouts[304], infinityBuf, [buf_7, buf_59, buf_61, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[305], layouts[305], infinityBuf, [buf_64, buf_7, buf_63], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[306], layouts[306], infinityBuf, [buf_3, buf_62, buf_64], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[307], layouts[307], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[308], layouts[308], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[309], layouts[309], infinityBuf, [buf_3, buf_62, buf_64, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[310], layouts[310], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[311], layouts[311], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[312], layouts[312], infinityBuf, [buf_7, buf_62, buf_64, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[313], layouts[313], infinityBuf, [buf_65, buf_7, buf_66], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[314], layouts[314], infinityBuf, [buf_3, buf_62, buf_64], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[315], layouts[315], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[316], layouts[316], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[317], layouts[317], infinityBuf, [buf_3, buf_62, buf_64, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[318], layouts[318], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[319], layouts[319], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[320], layouts[320], infinityBuf, [buf_7, buf_62, buf_64, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[321], layouts[321], infinityBuf, [buf_67, buf_7, buf_66], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[322], layouts[322], infinityBuf, [buf_3, buf_65, buf_67], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[323], layouts[323], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[324], layouts[324], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[325], layouts[325], infinityBuf, [buf_3, buf_65, buf_67, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[326], layouts[326], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[327], layouts[327], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[328], layouts[328], infinityBuf, [buf_7, buf_65, buf_67, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[329], layouts[329], infinityBuf, [buf_68, buf_7, buf_69], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[330], layouts[330], infinityBuf, [buf_3, buf_65, buf_67], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[331], layouts[331], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[332], layouts[332], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[333], layouts[333], infinityBuf, [buf_3, buf_65, buf_67, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[334], layouts[334], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[335], layouts[335], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[336], layouts[336], infinityBuf, [buf_7, buf_65, buf_67, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[337], layouts[337], infinityBuf, [buf_70, buf_7, buf_69], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[338], layouts[338], infinityBuf, [buf_3, buf_68, buf_70], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[339], layouts[339], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[340], layouts[340], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[341], layouts[341], infinityBuf, [buf_3, buf_68, buf_70, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[342], layouts[342], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[343], layouts[343], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[344], layouts[344], infinityBuf, [buf_7, buf_68, buf_70, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[345], layouts[345], infinityBuf, [buf_71, buf_7, buf_72], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[346], layouts[346], infinityBuf, [buf_3, buf_68, buf_70], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[347], layouts[347], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[348], layouts[348], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[349], layouts[349], infinityBuf, [buf_3, buf_68, buf_70, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[350], layouts[350], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[351], layouts[351], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[352], layouts[352], infinityBuf, [buf_7, buf_68, buf_70, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[353], layouts[353], infinityBuf, [buf_73, buf_7, buf_72], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[354], layouts[354], infinityBuf, [buf_3, buf_71, buf_73], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[355], layouts[355], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[356], layouts[356], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[357], layouts[357], infinityBuf, [buf_3, buf_71, buf_73, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[358], layouts[358], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[359], layouts[359], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[360], layouts[360], infinityBuf, [buf_7, buf_71, buf_73, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[361], layouts[361], infinityBuf, [buf_74, buf_7, buf_75], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[362], layouts[362], infinityBuf, [buf_3, buf_71, buf_73], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[363], layouts[363], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[364], layouts[364], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[365], layouts[365], infinityBuf, [buf_3, buf_71, buf_73, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[366], layouts[366], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[367], layouts[367], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[368], layouts[368], infinityBuf, [buf_7, buf_71, buf_73, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[369], layouts[369], infinityBuf, [buf_76, buf_7, buf_75], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[370], layouts[370], infinityBuf, [buf_3, buf_74, buf_76], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[371], layouts[371], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[372], layouts[372], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[373], layouts[373], infinityBuf, [buf_3, buf_74, buf_76, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[374], layouts[374], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[375], layouts[375], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[376], layouts[376], infinityBuf, [buf_7, buf_74, buf_76, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[377], layouts[377], infinityBuf, [buf_77, buf_7, buf_78], [128, 256, 2]);
        addComputePass(device, commandEncoder, pipelines[378], layouts[378], infinityBuf, [buf_3, buf_74, buf_76], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[379], layouts[379], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[380], layouts[380], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[381], layouts[381], infinityBuf, [buf_3, buf_74, buf_76, buf_6], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[382], layouts[382], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[383], layouts[383], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[384], layouts[384], infinityBuf, [buf_7, buf_74, buf_76, buf_6, buf_5], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[385], layouts[385], infinityBuf, [buf_79, buf_7, buf_78], [128, 256, 7]);
        addComputePass(device, commandEncoder, pipelines[386], layouts[386], infinityBuf, [buf_3, buf_77, buf_79], [10240, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[387], layouts[387], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[388], layouts[388], infinityBuf, [buf_5, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[389], layouts[389], infinityBuf, [buf_3, buf_77, buf_79, buf_5], [1024, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[390], layouts[390], infinityBuf, [buf_4, buf_3], [40, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[391], layouts[391], infinityBuf, [buf_6, buf_4], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[392], layouts[392], infinityBuf, [buf_7, buf_77, buf_79, buf_5, buf_6], [32768, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[393], layouts[393], infinityBuf, [buf_80, buf_7, buf_81], [32768, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[394], layouts[394], infinityBuf, [output0, buf_80], [32768, 4, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mindgrab_128MB;
