// WebGPU doesn't expose hardware spec

// create a GPUShaderModule for the vertex shader and the fragment shader
export function cellShaderModule(device) {
    return device.createShaderModule({
        label: "Cell Shader",
        code:/* wgsl */`
            // structure to handle data input and output of shader
            struct VertexInput {
                // passing vertices positions as argument from vertexBufferLayout at location 0
                // since format is "float32x2" we pass a vec2f
                @location(0) pos: vec2f,
                // allow this shader to change behaviour depending on instance
                @builtin(instance_index) instance: u32,
            };
    
            struct VertexOutput {
                @builtin(position) pos: vec4f,
                // outputing cell treated by shader to transmit it to fragment shader
                @location(0) cell: vec2f,
            };
    
            // bind group as uniform variable describing data in the grid uniform buffer 
            @group(0) @binding(0) var<uniform> grid: vec2f;
            // bing group for cell state storage data
            @group(0) @binding(1) var<storage> cellState: array<u32>;
    
            // vertex shader is valid only when returning at least last vertex position
            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                
                // converting instance number as 32-bit floating point
                let instance = f32(input.instance);
                let cell  = vec2f(instance % grid.x, floor(instance / grid.x)); //targeting cell grid depending on instance
                let state = f32(cellState[input.instance]);
    
                let cellOffset = cell / grid * 2; //we only want to make the cell placed at 1/grid size of canvas (size 2 -1,1)
                // placing square center at the top right of the canvas (pos+1 means vertices are placed at their position + (1,1))
                // reducing its size to fit GRID_SIZE (/grid)
                // placing it bottom left of canvas (-1)
                // placing it at cell (relative to grid) with + cellOffset
                let squarePos = (input.pos * state + 1) / grid - 1 + cellOffset;
    
                output.pos = vec4f(squarePos, 0, 1);
                output.cell = cell;
                return output;
            }
            
            // fragment shader needs a location instead of a position as a fragment shader
            @fragment
            // retrieving output of vertex shader
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                // making red an green value depends on cell x and y
                let cellRedGreen = input.cell / grid;
                // calculating blue depending on red value
                return vec4f(cellRedGreen, 1 - cellRedGreen.x ,1);
            }
        `
    });
}

code:/* wgsl */`
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) cell: vec2f,
          };

          @group(0) @binding(0) var<uniform> grid: vec2f;
          @group(0) @binding(1) var<storage> cellState: array<u32>;

          @vertex
          fn vertexMain(@location(0) position: vec2f,
                        @builtin(instance_index) instance: u32) -> VertexOutput {
            var output: VertexOutput;

            let i = f32(instance);
            let cell = vec2f(i % grid.x, floor(i / grid.x));

            let scale = f32(cellState[instance]);
            let cellOffset = cell / grid * 2;
            let gridPos = (position*scale+1) / grid - 1 + cellOffset;

            output.position = vec4f(gridPos, 0, 1);
            output.cell = cell / grid;
            return output;
          }

          @fragment
          fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            return vec4f(input.cell, 1.0 - input.cell.x, 1);
          }
        `

// Create a compute shader to manage the simulation
// compute shaders don't have expected input or output
// meaning it needs to be called a specified number of times
// compared to vertex and fragment that iterate over passed data

export function simulationShaderModule(device, WORKGROUP_SIZE) {
    return device.createShaderModule({
        label: "Game simulation",
        code: /* wgsl */`
            // grid size
            @group(0) @binding(0) var<uniform> grid: vec2f;
            // for cell state we don't have access to vertex or fragment information so we store both the in and out state
            @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
            // we want to write in the buffer for output (and there is no write only)
            @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

            fn cellIndex(cell: vec2u) -> u32 {
                // returning cell pos as single int (row pos * number of column + column pos)
                // % grid make out of bound targeted cell to loop back at end/start of row/column
                return (cell.y  % u32(grid.y)) * u32(grid.x) + cell.x % u32(grid.x);
            }

            fn cellActive(x: u32, y: u32) -> u32 {
                return cellStateIn[cellIndex(vec2(x, y))];
            }

            @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
            // global_invocation_id is the position shader call grid
            fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
                let activeNeighbors =
                    cellActive(cell.x+1, cell.y+1) +
                    cellActive(cell.x, cell.y+1) +
                    cellActive(cell.x+1, cell.y) +
                    cellActive(cell.x-1, cell.y-1) +
                    cellActive(cell.x-1, cell.y) +
                    cellActive(cell.x, cell.y-1) +
                    cellActive(cell.x+1, cell.y-1) +
                    cellActive(cell.x-1, cell.y+1);
                
                let i = cellIndex(cell.xy);

                switch activeNeighbors {
                    case 2: {
                        cellStateOut[i] = cellStateIn[i];
                    }
                    case 3: {
                        cellStateOut[i] = 1;
                    }
                
                    default: {
                        cellStateOut[i] = 0;
                    }
                }
            }
        `
    })
}
