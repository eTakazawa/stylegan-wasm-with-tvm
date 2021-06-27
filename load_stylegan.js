const fs = require("fs");
const tvmjs = require("./dist");
const EmccWASI = require("./dist/wasm/tvmjs_runtime.wasi.js");

function getRandomArray(length) {
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    data[i] = Math.random();
  }
  return data;
}

function transformForCanvasImage(result) {
  let arr = new Array(result.length / 3 * 4);
  for (let c = 0; c < 3; c++) {
    for (let h = 0; h < 512; h++) {
      for (let w = 0; w < 512; w++) {
        let pos = h * 512 * 4 + w * 4 + c;
        let val = result[c * 512 * 512 + h * 512 + w];
        arr[pos] = (val + 1) / 2 * 255;
      }
    }
  }
  for (let h = 0; h < 512; h++) {
    for (let w = 0; w < 512; w++) {
      let pos = h * 512 * 4 + w * 4 + 3;
      arr[pos] = 255;
    }
  }
  return arr;
}

function customLog(message) {
  console.log(message);
}

// Load net.wasm
classify = null;
async function initModel() {
  console.log('Start model initialization.')
  console.time('Initialization time');
  const wasmSource = fs.readFileSync("stylegan.wasm");
  // const wasmModule = await WebAssembly.compile(wasmSource);
  const tvm = await tvmjs.instantiate(
    new Uint8Array(wasmSource),
    new EmccWASI(),
    customLog
  );
  // Load system library
  const syslib = tvm.systemLib();

  // grab pre-loaded function
  ctx = tvm.cpu();
  const graphJson = fs.readFileSync("stylegan.json");
  const paramsBinary = fs.readFileSync("stylegan.params");
  const executor = tvm.createGraphExecutor(graphJson, syslib, ctx);
  executor.loadParams(paramsBinary);

  // prepare input data
  const inputData = tvm.empty([1, 512], "float32", ctx);
  const labelData = tvm.empty([1], "float32", ctx);
  const outputData = tvm.empty([1, 3, 512, 512], "float32", ctx);

  // run the first time to make sure all weights are populated.
  executor.run();
  await ctx.sync();
  console.timeEnd('Initialization time');

  classify = async(inputArray) => {
    // run
    console.log('Start inference.')
    console.time('Inference time');
    inputData.copyFrom(inputArray);
    labelData.copyFrom(new Float32Array(1));
    executor.setInput("input", inputData);
    executor.setInput("label", labelData);
    executor.run();
    outputData.copyFrom(executor.getOutput(0));
    await ctx.sync();
    console.timeEnd('Inference time');

    // report the result.
    result = Array.from(outputData.toArray());
    return result;
  }
}

async function Run(inputArray) {
  if (classify == null) {
    await initModel();
  }
  return await classify(inputArray);
}

async function test() {
  await initModel();
  let result = await classify(getRandomArray(512));
  for (let i = 0; i < result.length; i++) {
    console.log(result[i]);
  }
}

exports.test = test;
exports.Run = Run;
exports.getRandomArray = getRandomArray;
exports.transformForCanvasImage = transformForCanvasImage;
