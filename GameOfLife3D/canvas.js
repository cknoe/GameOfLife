import { computePass, renderPass} from './pipelines.js'

const GRID_SIZE = 300;
const UPDATE_INTERVAL = 16;
let step = 0;
const canvas = document.querySelector("canvas");
// many hardaware should work with a workgroup size of 64 8*8
const WORKGROUP_SIZE = 8;

// ---------------- Set up WebGPU
// Check if browser is compatible with webgpu
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}
// Check if GPUAdapter exits (relative to hardware)
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdaptater found.");
}
//GPUDevice
const device = await adapter.requestDevice();
console.log(device.limits);

//----------------- Configure Canvas
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat, //texture format
});

//--------------- Preparing Buffer data to draw a square
// preparing a square, as two triangle (primitive), in canvas space (-1 to 1 2D plane)
const vertices = new Float32Array([
    // X,  Y
    -0.8, -0.8,
    -0.8, 0.8, 
    0.8, 0.8,

    -0.8, -0.8,
    0.8, -0.8, 
    0.8, 0.8,
]);
// creating GPUBuffer object
const vertexBuffer = device.createBuffer({
    label: "Cell vertices", //Optional
    size: vertices.byteLength, // give length in byte
    // Buffer will be used for vertex data and we want to copy data
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
// copying vertices array in its memory
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);
// Defining buffer data structure with a GPUVertexBufferLayout
const vertexBufferLayout = {
    arrayStride: 8, // Number of bytes between 2 vertices (vertex is defined as two Float32)
    attributes: [{
        format: "float32x2", // GPUVertexFormat
        offset: 0, // Number of bytes before this attribute (only one attribute its 0) 
        shaderLocation: 0,
    }]
};

//-------------- Creating the grid size uniform buffer
// Create a uniform buffer that describes the grid
/* Uniform buffer are used to pass data that stay identical each time they're called
(compared to Vertex buffer which pass different value from its data each time its called)*/
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

//-------------- Creating Grid state storage buffer
// create a storage buffer representing grid (to activate cells)
const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateStorage = [
    device.createBuffer({
    label: "Cell State A",
    size: cellStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];
for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

//--------------- Creating Bind Group
// Bind groups are ressources that we want topass to our shader like different buffers
// Create a GPUBindGroupLayout wich will allow to share bind groups between pipelines
const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, // where we want to use our buffer
      buffer: {} // Grid uniform buffer (uniform buffer are default type)
    }, {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage"} // Cell state input buffer
    }, {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage"} // Cell state output buffer
    }]
});

// GPUBindGroups are immutable, you can't change pointer to resources but can change data
const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
        // Pipeline layout is "auto", bind group layout from shader are created
        // @group(0) from shader
        layout: bindGroupLayout,
        entries: [{
                // @binding(0) from shader
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[0] }
            },
            {
                binding: 2,
                resource: { buffer: cellStateStorage[1] }
            }],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
        // Pipeline layout is "auto", bind group layout from shader are created
        // @group(0) from shader
        layout: bindGroupLayout,
        entries: [{
                // @binding(0) from shader
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[1] }
            },
            {
                binding: 2,
                resource: { buffer: cellStateStorage[0] }
            }],
    }),
];

// GPUPipelineLayout to use our binding group layout
const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [ bindGroupLayout ], //at index 0 bindGroupLayout is the @group(0) from shader
});

//------------------- Render & Compute pass
function updateGrid() {
    // interface to save gpu commands
    const encoder = device.createCommandEncoder();

    computePass(device, encoder, step, pipelineLayout, bindGroups, WORKGROUP_SIZE, GRID_SIZE)

    step++;

    renderPass(device,
        encoder,
        step,
        context,
        pipelineLayout,
        bindGroups,
        vertexBufferLayout,
        vertexBuffer,
        canvasFormat,
        GRID_SIZE,
        vertices)

    // creating pass does nothing, it needs to be called as a GPUCommandBuffer by the device queue
    /*const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);*/
    // Once called command buffer cant be re-used, no need to keep it. this can be used instead :
    device.queue.submit([encoder.finish()]);
}

updateGrid(); // not waiting interval
setInterval(updateGrid, UPDATE_INTERVAL);
