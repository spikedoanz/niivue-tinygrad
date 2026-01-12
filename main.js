import { Niivue } from '@niivue/niivue'
// IMPORTANT: we need to import this specific file.
import subcortical from "./net_subcortical.js"
import subcortical30chan from "./net_subcortical30chan.js"
import tissue_fast from "./net_tissue_fast.js"
import tissue_fast_tta from "./net_tissue_fast_tta.js"
import robust_tissue from "./net_robust_tissue.js"
import mindgrab from "./net_mindgrab.js"
import t2 from "./net_t2.js"
import DKatlas from "./net_DKatlas.js"
import aparc50 from "./net_aparc50.js"

const models = {
  "subcortical": {
    "net": subcortical,
    "weightPath": "./net_subcortical.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "min-max"
  },
  "subcortical30chan": {
    "net": subcortical30chan,
    "weightPath": "./net_subcortical30chan.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "min-max"
  },
  "tissue_fast": {
    "net": tissue_fast,
    "weightPath":
    "./net_tissue_fast.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "min-max"
  },
  "tissue_fast_tta": {
    "net": tissue_fast_tta,
    "weightPath": "./net_tissue_fast_tta.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "min-max"
  },
  "robust_tissue": {
    "net": robust_tissue,
    "weightPath": "./net_robust_tissue.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "qnormalize"
  },
  "DKatlas": { // this is in float16
    "net": DKatlas,
    "weightPath": "./net_DKatlas.safetensors",
    "colormap": "./colormap_DKatlas.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "min-max",
    "fp16": true  // requires Float16Array input
  },
  "mindgrab": {
    "net": mindgrab,
    "weightPath": "./net_mindgrab.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "qnormalize"
  },
  "t2": {
    "net": t2,
    "weightPath":
    "./net_t2.safetensors",
    "colormap": "./colormap_t2.json",
    "volume": "./M2265_T2w.nii.gz",
    "normalization": "min-max"
  },
  "aparc50": {
    "net": aparc50,
    "weightPath": "./net_aparc50.safetensors",
    "colormap": "./colormap_aparc50.json",
    "volume": "./t1_crop.nii.gz",
    "normalization": "qnormalize",
    "fp16": true
  }
}

let selectedModel = models[document.getElementById("segmentationDropdown").value]

function qnormalize(img32, qmin = 0.02, qmax = 0.98, eps = 1e-3) {
  // Create sorted copy to find quantiles
  const sorted = [...img32].sort((a, b) => a - b);
  // Calculate quantile indices
  const n = sorted.length;
  const qminIndex = Math.floor(qmin * (n - 1));
  const qmaxIndex = Math.floor(qmax * (n - 1));
  // Linear interpolation for accurate quantiles
  const qminFrac = qmin * (n - 1) - qminIndex;
  const qmaxFrac = qmax * (n - 1) - qmaxIndex;
  let qlow = sorted[qminIndex];
  if (qminIndex < n - 1) {
    qlow += qminFrac * (sorted[qminIndex + 1] - sorted[qminIndex]);
  }
  let qhigh = sorted[qmaxIndex];
  if (qmaxIndex < n - 1) {
    qhigh += qmaxFrac * (sorted[qmaxIndex + 1] - sorted[qmaxIndex]);
  }
  // Normalize and clip in-place
  const scale = 1 / (qhigh - qlow + eps);
  for (let i = 0; i < img32.length; i++) {
    img32[i] = Math.max(0, Math.min(1, (img32[i] - qlow) * scale));
  }
}

async function main() {
  clipCheck.onchange = function () {
    if (clipCheck.checked) {
      nv1.setClipPlane([0, 0, 90])
    } else {
      nv1.setClipPlane([2, 0, 90])
    }
  }
  opacitySlider0.oninput = function () {
    nv1.setOpacity(0, opacitySlider0.value / 255)
    nv1.updateGLVolume()
  }
  opacitySlider1.oninput = function () {
    nv1.setOpacity(1, opacitySlider1.value / 255)
  }
  function doLoadImage() {
    opacitySlider0.oninput()
  }
  async function fetchJSON(fnm) {
    const response = await fetch(fnm)
    const js = await response.json()
    return js
  }
  saveImgBtn.onclick = function () {
    nv1.volumes[1].saveToDisk('Custom.nii')
  }
  aboutBtn.onclick = function () {
    const url = "https://github.com/niivue/niivue-tinygrad"
    window.open(url, "_blank")
  }
  async function ensureConformed() {
    const nii = nv1.volumes[0]
    let isConformed = nii.dims[1] === 256 && nii.dims[2] === 256 && nii.dims[3] === 256
    if (nii.permRAS[0] !== -1 || nii.permRAS[1] !== 3 || nii.permRAS[2] !== -2) {
      isConformed = false
    }
    if (isConformed) {
      return
    }
    const nii2 = await nv1.conform(nii, false)
    nv1.removeVolume(nv1.volumes[0])
    nv1.addVolume(nii2)
  }
  async function closeAllOverlays() {
    while (nv1.volumes.length > 1) {
      nv1.removeVolume(nv1.volumes[1])
    }
  }
  const getDevice = async () => {
    if (!navigator.gpu) return false;
    const requiredLimits = {};
    const maxBufferSize = 4294967200; // 4gb required for DKatlas
    //const maxBufferSize = 1294967200; // 4gb required for DKatlas
    requiredLimits.maxStorageBufferBindingSize = maxBufferSize;
    requiredLimits.maxBufferSize = maxBufferSize;
    const adapter = await navigator.gpu.requestAdapter();
    console.log('Adapter limits:', adapter.limits);
    console.log('Max buffer size:', adapter.limits.maxBufferSize);
    console.log('Max storage buffer binding size:', adapter.limits.maxStorageBufferBindingSize);
    return await adapter.requestDevice({
        requiredLimits: requiredLimits,
        requiredFeatures: ["shader-f16"]
    });
  };

  const device = await getDevice();

  function convertInMemoryOrder(inverse, size, data) {
    let output = new Float32Array(data.length)
    let it = 0;

    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        for (let z = 0; z < size; z++) {
          let idx = x + y * size + z * size * size;
          if (inverse) {
            output[idx] = data[it++];
          } else {
            output[it++] = data[idx];
          }
        }
      }
    }

    return output
  }

  segmentBtn.onclick = async function () {
    if (nv1.volumes.length < 1) {
      window.alert('Please open a voxel-based image')
      return
    }
    const startTime = Date.now();
    console.log('[Model] Starting segmentation...')
    loadingCircle.classList.remove('hidden')

    console.log('[Model] Phase 1: Preparing volume (closing overlays, conforming)...')
    await closeAllOverlays()
    await ensureConformed()
    console.log('[Model] Phase 1: Volume preparation complete')

    console.log('[Model] Phase 2: Normalizing input data...')
    let img32 = convertInMemoryOrder(/*inverse*/ false, 256, new Float32Array(nv1.volumes[0].img))

    console.log(selectedModel)
    if (selectedModel['normalization'] === 'min-max') {
      console.log('[Model] Phase 2: Using min-max normalization')
      // normalize input data to range 0..1
      let mx = img32[0]
      let mn = mx
      for (let i = 0; i < img32.length; i++) {
        mx = Math.max(mx, img32[i])
        mn = Math.min(mn, img32[i])
      }
      let scale32 = 1 / (mx - mn)
      for (let i = 0; i < img32.length; i++) {
        img32[i] = (img32[i] - mn) * scale32
      }
    } else {
      console.log('[Model] Phase 2: Using quantile normalization')
      qnormalize(img32);
    }
    console.log('[Model] Phase 2: Normalization complete')

    console.log('[Model] Phase 3: Loading model weights...')
    const session = await selectedModel["net"].load(device, selectedModel["weightPath"]);
    console.log('[Model] Phase 3: Model weights loaded')

    const shape = [1, 1, 256, 256, 256]
    const nvox = shape.reduce((a, b) => a * b)
    if (img32.length !== nvox) {
      throw new Error(`img32 length (${img32.length}) does not match expected tensor length (${expectedLength})`)
    }

    console.log('[Model] Phase 4: Running inference...')
    // Convert to Float16Array if model requires fp16 input
    let inputData = img32;
    if (selectedModel['fp16']) {
      console.log('[Model] Phase 4: Converting to Float16Array for fp16 model')
      inputData = new Float16Array(img32);
    }
    const results = await session(inputData);
    console.log('[Model] Phase 4: Inference complete')

    // Log label distribution in output cube
    const outputData = results[0];
    const totalVoxels = outputData.length;
    const labelCounts = {};
    for (let i = 0; i < totalVoxels; i++) {
      const label = Math.round(outputData[i]);
      labelCounts[label] = (labelCounts[label] || 0) + 1;
    }
    console.log('[Model] Label distribution in output cube:');
    const sortedLabels = Object.keys(labelCounts).sort((a, b) => a - b);
    for (const label of sortedLabels) {
      const count = labelCounts[label];
      const ratio = (count / totalVoxels * 100).toFixed(2);
      console.log(`  Label ${label}: ${count} voxels (${ratio}%)`);
    }

    console.log('[Model] Phase 5: Post-processing results...')
    let segmentImg = nv1.cloneVolume(0)
    segmentImg.img = convertInMemoryOrder(/*inverse*/ true, 256, results[0])
    segmentImg.hdr.datatypeCode = 16
    segmentImg.hdr.dims[4] = 1
    segmentImg.trustCalMinMax = false

    // Add the output to niivue
    const cmap = await fetchJSON(selectedModel["colormap"])
    segmentImg.setColormapLabel(cmap)
    segmentImg.opacity = opacitySlider1.value / 255
    nv1.addVolume(segmentImg)
    console.log('[Model] Phase 5: Post-processing complete')

    loadingCircle.classList.add('hidden')
    const elapsedTime = Date.now() - startTime
    console.log(`[Model] Segmentation complete in ${elapsedTime}ms`)
    document.getElementById("intensity").innerHTML = `${elapsedTime}ms to segment`
  }
  function handleLocationChange(data) {
    document.getElementById("intensity").innerHTML = data.string
  }
  const defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    onLocationChange: handleLocationChange,
  }
  const nv1 = new Niivue(defaults)
  nv1.attachToCanvas(gl1)
  nv1.opts.multiplanarForceRender = true
  nv1.opts.yoke3Dto2DZoom = true
  nv1.opts.crosshairGap = 11
  nv1.setInterpolation(true)
  nv1.onImageLoaded = doLoadImage
  await nv1.loadVolumes([{ url: selectedModel["volume"] }])
  segmentBtn.onclick()

  document.getElementById("segmentationDropdown").addEventListener("change", async function () {
    selectedModel = models[this.value]
    if (nv1.volumes[0].url != selectedModel["volume"]) {
      nv1.removeVolumeByIndex(0)
      await nv1.loadVolumes([{ url: selectedModel["volume"] }])
    }
    segmentBtn.onclick()
  });
}

main()
