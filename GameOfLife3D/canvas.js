import { computePass, renderPass} from './pipelines.js'

const GRID_SIZE = 16;
const UPDATE_INTERVAL = 20;
let step = 0;
let RUN_SIMULATION = 1;
let computePhase = 1;
const canvas = document.querySelector("canvas");
const WORKGROUP_SIZE = 8;

// ------------------------------------------------ GPU Device ------------------------------------------------
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdaptater found.");
}
const device = await adapter.requestDevice();
console.log(device.limits);

// ------------------------------------------------ Canvas ------------------------------------------------
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat, //texture format
});
canvas.addEventListener("click", async () => {
    if (!document.pointerLockElement) {
      await canvas.requestPointerLock({
        unadjustedMovement: true,
      });
    }
});
document.addEventListener("pointerlockchange", lockChangeAlert, false);
function lockChangeAlert() {
    if (document.pointerLockElement === canvas) {
      document.addEventListener("mousemove", updatePosition, false);
    } else {
      document.removeEventListener("mousemove", updatePosition, false);
    }
  }
let mouseDeltaX = 0;
let mouseDeltaY = 0;
function updatePosition(e) {
    mouseDeltaX = mouseDeltaX + e.movementX;
    mouseDeltaY = mouseDeltaY + e.movementY;
}

// ------------------------------------------------ Vertices ------------------------------------------------
// preparing a square, as two triangle (primitive), in canvas space (-1 to 1 2D plane)
const vertices = new Float32Array([
    // X,  Y,  Z, u, v
    -1, -1, -1, 0, 1,
    -1, 1, -1, 0, 0,
    1, 1, -1, 1, 0,

    -1, -1, -1, 0, 1,
    1, -1, -1, 1, 1,
    1, 1, -1, 1, 0,
    // 2
    -1, -1, -1, 0, 0,
    -1, -1, 1, 0, 1,
    1, -1,  1, 1, 1,

    -1, -1, -1, 0, 0,
    1, -1, -1, 1, 0,
    1, -1, 1, 1, 1,
    // 3
    -1, -1, -1, 0, 0,
    -1, 1, -1, 0, 1,
    -1, 1, 1, 1, 1,

    -1, -1, -1, 0, 0,
    -1, -1, 1, 1, 0,
    -1, 1, 1, 1, 1,
    // 4
    1, -1, -1, 0, 0,
    1, 1, -1, 0, 1,
    1, 1, 1, 1, 1,

    1, -1, -1, 0, 0,
    1, -1, 1, 1, 0,
    1, 1, 1, 1, 1,
    // 5
    -1, 1, -1, 0, 0,
    -1, 1, 1, 0, 1,
    1, 1, 1, 1, 1,

    -1, 1, -1, 0, 0,
    1, 1, -1, 1, 0,
    1, 1, 1, 1, 1,
    // 6
    -1, -1, 1, 0, 0,
    -1, 1, 1, 0, 1,
    1, 1, 1, 1, 1,

    -1, -1, 1, 0, 0,
    1, -1, 1, 1, 0,
    1, 1, 1, 1, 1,
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
    arrayStride: 20, // Number of bytes between 2 vertices (vertex is defined as 3 Float32)
    attributes: [{
        format: "float32x3", // GPUVertexFormat
        offset: 0, // Number of bytes before this attribute (only one attribute its 0) 
        shaderLocation: 0,
    },
    {
        format: "float32x2", // GPUVertexFormat
        offset: 12, // Number of bytes before this attribute
        shaderLocation: 1,
    }
]
};


// ------------------------------------------------ Texture ------------------------------------------------
const imgOnResponse = await fetch('./faceon.png');
const imageOnBitmap = await createImageBitmap(await imgOnResponse.blob());

const textureOn = device.createTexture({
    size: [imageOnBitmap.width, imageOnBitmap.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
});

device.queue.copyExternalImageToTexture(
    { source: imageOnBitmap },
    { texture: textureOn },
    [imageOnBitmap.width, imageOnBitmap.height]
);

const imgOffResponse = await fetch('./faceoff.png');
const imageOffBitmap = await createImageBitmap(await imgOffResponse.blob());

const textureOff = device.createTexture({
    size: [imageOnBitmap.width, imageOnBitmap.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
});

device.queue.copyExternalImageToTexture(
    { source: imageOffBitmap },
    { texture: textureOff },
    [imageOffBitmap.width, imageOffBitmap.height]
);

const sampler = device.createSampler({
magFilter: 'linear',
minFilter: 'linear',
});

// ------------------------------------------------ Depth Stencil ------------------------------------------------

let depthStencilState = { //GPUDepthStenciState
    format: "depth24plus-stencil8",
    depthWriteEnabled: true,
    depthCompare: "less-equal",
};

// create depth texture
const depthTextureSize = { //GPUExtent3D
    width: canvas.width,
    height: canvas.height,
    depthOrArrayLayers: 1,
};
const depthBufferDescriptor = { //GPUTextureDescriptor
    size: depthTextureSize,
    format: "depth24plus-stencil8",
    usage:GPUTextureUsage.RENDER_ATTACHMENT,
};
const depthStencilBuffer = device.createTexture(depthBufferDescriptor);

// Create depth stencil view
const viewDescriptor = { //GPUTextureViewDescriptor
    format: "depth24plus-stencil8",
    dimension: "2d",
    aspect: "all"
};
const depthStencilView = depthStencilBuffer.createView(viewDescriptor);

const depthStencilAttachment = {
    view: depthStencilView,
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
    stencilLoadOp: "clear",
    stencilStoreOp: "discard",// not using this value
};
// ------------------------------------------------ Matrices ------------------------------------------------
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
const rotationRadians = Math.PI;
function rotationYMatrixFunction(theta) {
    return new Float32Array([
        Math.cos(theta), 0, Math.sin(theta), 0,
        0, 1, 0, 0,
        -Math.sin(theta), 0, Math.cos(theta), 0,
        0, 0, 0, 1,
    ]);
}
var rotationYMatrix = rotationYMatrixFunction(rotationRadians);

function rotationXMatrixFunction(theta) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(theta), Math.sin(theta), 0,
        0, -Math.sin(theta), Math.cos(theta), 0,
        0, 0, 0, 1,
    ]);
}
var rotationXMatrix = rotationXMatrixFunction(rotationRadians);

function orthographicMatrixFunction(left, right, bottom, top, near, far) {
    return new Float32Array([
        2 / (right - left), 0, 0, 0,
        0, 2 / (top - bottom), 0, 0,
        0, 0, -2 / (far - near), 0,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1
    ]);
}
const orthographicMatrix = orthographicMatrixFunction(-1, 1, -1, 1, 0.1, 20);

const matrixSize = identityMatrix.byteLength + rotationYMatrix.byteLength + rotationXMatrix.byteLength + orthographicMatrix.byteLength;
const matrixBuffer = device.createBuffer({
    label: "Matrix Buffer",
    size: matrixSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(matrixBuffer, 0, identityMatrix);
device.queue.writeBuffer(matrixBuffer, identityMatrix.byteLength, orthographicMatrix);



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





// ------------------------------------------------ Creating the grid size uniform buffer ------------------------------------------------
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

// ------------------------------------------------ Creating Grid state storage buffer ------------------------------------------------
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
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
    //cellStateArray[i] = 1;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

// ------------------------------------------------ Creating Bind Group ------------------------------------------------
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
        visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
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
    },
    {
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {} // sampler
    },
    {
        binding: 5,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {} // texture
    },
    {
        binding: 6,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {} // texture
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
            },
            {
                binding: 4,
                resource: sampler 
            },
            {
                binding: 5,
                resource: textureOn.createView()
            },
            {
                binding: 6,
                resource: textureOff.createView()
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
            },
            {
                binding: 4,
                resource: sampler
            },
            {
                binding: 5,
                resource: textureOn.createView()
            },
            {
                binding: 6,
                resource: textureOff.createView()
            }],
    }),
];

// GPUPipelineLayout to use our binding group layout
const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [ bindGroupLayout ], //at index 0 bindGroupLayout is the @group(0) from shader
});

// ------------------------------------------------ Render & Compute pass ------------------------------------------------

function updateGrid() {
    // interface to save gpu commands
    const encoder = device.createCommandEncoder();

    if ((step % 40 == 0) && (RUN_SIMULATION)) {
        computePhase = (computePhase == 0) ? 1 : 0;
        computePass(device, encoder, computePhase, pipelineLayout, bindGroups, WORKGROUP_SIZE, GRID_SIZE)
    }


    step++;


    let rotationMatrix = rotationYMatrixFunction(((mouseDeltaX * 0.5) % 360) * (Math.PI/180));
    device.queue.writeBuffer(matrixBuffer, identityMatrix.byteLength + orthographicMatrix.byteLength, rotationMatrix);
    rotationMatrix = rotationXMatrixFunction(((mouseDeltaY * 0.5) % 360) * (Math.PI/180));
    device.queue.writeBuffer(matrixBuffer, identityMatrix.byteLength + orthographicMatrix.byteLength + rotationMatrix.byteLength, rotationMatrix);

    renderPass(device,
        encoder,
        computePhase,
        context,
        pipelineLayout,
        bindGroups,
        vertexBufferLayout,
        vertexBuffer,
        canvasFormat,
        GRID_SIZE,
        vertices,
        depthStencilState,
        depthStencilAttachment)

    // creating pass does nothing, it needs to be called as a GPUCommandBuffer by the device queue
    /*const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);*/
    // Once called command buffer cant be re-used, no need to keep it. this can be used instead :
    device.queue.submit([encoder.finish()]);
}

updateGrid(); // not waiting interval
setInterval(updateGrid, UPDATE_INTERVAL);
