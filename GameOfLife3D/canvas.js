import { computePass, renderPass} from './pipelines.js'

const GRID_SIZE = 32;
const UPDATE_INTERVAL = 1000;
let step = 0;
const canvas = document.querySelector("canvas");
const WORKGROUP_SIZE = 8;

// ---------------- GPU Device
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdaptater found.");
}
const device = await adapter.requestDevice();
console.log(device.limits);

//----------------- Canvas
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat, //texture format
});

//--------------- Preparing Buffer data to draw a square
// preparing a square, as two triangle (primitive), in canvas space (-1 to 1 2D plane)
const vertices = new Float32Array([
    // X,  Y,  Z
    -0.8, -0.8, 0,
    -0.8, 0.8, 0,
    0.8, 0.8, 0,

    -0.8, -0.8, 0,
    0.8, -0.8, 0,
    0.8, 0.8, 0,

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
    arrayStride: 12, // Number of bytes between 2 vertices (vertex is defined as 3 Float32)
    attributes: [{
        format: "float32x3", // GPUVertexFormat
        offset: 0, // Number of bytes before this attribute (only one attribute its 0) 
        shaderLocation: 0,
    }]
};

/*Scalars (float, int, uint) require 4-byte alignment.
vec2 requires 8-byte alignment.
vec3 and vec4 require 16-byte alignment.
mat3x3 is treated as three vec3s, each padded to 16 bytes.*/
// meaning I need a 4*3 matrix with zeroes in the last column to represent 3*3 matrix
const identityMatrix = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
]);
const rotationRadians = Math.PI / 4;
const rotationYMatrix = new Float32Array([
    Math.cos(rotationRadians),0,Math.sin(rotationRadians), 0,
    0, 1, 0, 0,
    -Math.sin(rotationRadians),0,Math.cos(rotationRadians), 0,
]);

const matrixSize = identityMatrix.byteLength + rotationYMatrix.byteLength;
const matrixBuffer = device.createBuffer({
    label: "Rotation Buffer",
    size: matrixSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(matrixBuffer, 0, identityMatrix);
device.queue.writeBuffer(matrixBuffer, identityMatrix.byteLength, rotationYMatrix);




//---------- Some Buffer Logging
const readBackBuffer = device.createBuffer({
    size: matrixSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const commandEncoder = device.createCommandEncoder();
commandEncoder.copyBufferToBuffer(matrixBuffer, 0, readBackBuffer, 0, matrixSize);
const commands = commandEncoder.finish();
device.queue.submit([commands]);

await readBackBuffer.mapAsync(GPUMapMode.READ);
const copyArrayBuffer = readBackBuffer.getMappedRange();
const data = new Float32Array(copyArrayBuffer);
console.log("Buffer data :")
console.log(data);
readBackBuffer.unmap();
//------------------------------------





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
    },
    {
        binding: 3,
        visibility: GPUShaderStage.VERTEX,
        buffer: {} // rotation buffer
    },]
});

// GPUBindGroups are immutable, you can't change pointer to resources but can change data
const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
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
            },
            {
                binding: 3,
                resource: { buffer: matrixBuffer }
            }],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
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
            },
            {
                binding: 3,
                resource: { buffer: matrixBuffer }
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
