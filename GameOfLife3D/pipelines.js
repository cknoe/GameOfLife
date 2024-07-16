import { cellShaderModule, simulationShaderModule} from './shaders.js'

// Compute pipeline
function computePipeline(device, pipelineLayout, WORKGROUP_SIZE) {
    return device.createComputePipeline({
        label: "Simulation pipeline",
        layout: pipelineLayout,
        compute: {
          module: simulationShaderModule(device, WORKGROUP_SIZE),
          entryPoint: "computeMain",
        }
    });
}

//Render pipeline
function renderPipeline(device, pipelineLayout, vertexBufferLayout, canvasFormat) {
    return device.createRenderPipeline({
        label: "Cell pipeline",
        layout: pipelineLayout,
        vertex: {
            module: cellShaderModule(device),
            entryPoint: "vertexMain",
            buffers: [vertexBufferLayout]
        },
        fragment: {
            module: cellShaderModule(device),
            entryPoint: "fragmentMain",
            targets: [{
                format: canvasFormat //needs to match color attachments of render pass
            }]
        }
    });
}

// Compute pass
export function computePass(device, encoder, computePhase, pipelineLayout, bindGroups, WORKGROUP_SIZE, GRID_SIZE) {
     // Compute pass
     const computePass = encoder.beginComputePass();
     computePass.setPipeline(computePipeline(device, pipelineLayout, WORKGROUP_SIZE));
     computePass.setBindGroup(0, bindGroups[computePhase]);
     // to cover a 32*32 grid with worker sized at 8*8, i want to send 4*4 workgroup (4*8=32)
     const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
     computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
     computePass.end();
}

// Render pass
export function renderPass(device,
    encoder,
    computePhase,
    context,
    pipelineLayout,
    bindGroups,
    vertexBufferLayout,
    vertexBuffer,
    canvasFormat,
    GRID_SIZE, vertices) {
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(), // texture
            loadOp: "clear", //  clear canvas
            clearValue: { r: 0, g: 0.3, b: 0.2, a: 1 },
            storeOp: "store",
        }]
    });

    // call shader
    pass.setPipeline(renderPipeline(device, pipelineLayout, vertexBufferLayout, canvasFormat));
    // call vertexBuffer at location 0
    pass.setVertexBuffer(0, vertexBuffer);
    // set all @binding from @group(0) in shader are resources of corresponding bind group
    pass.setBindGroup(0, bindGroups[computePhase]);
    pass.draw(vertices.length / 3, GRID_SIZE * GRID_SIZE); // draw vertices, for number of instances
    pass.end();
}