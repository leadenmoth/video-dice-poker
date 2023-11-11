const GRID_SIZE = 6;
const UPDATE_INTERVAL = 50; // Update every 200ms (5 times/sec)
const WORKGROUP_SIZE = 8;
const canvas = document.querySelector("canvas");

//Check the browser supports WebGPU
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}

//Get adapter if available, then get device
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}
const device = await adapter.requestDevice();

//Configure canvas for WebGPU
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

//TODO: function to split a square into 2 triangles
//or look into Index Buffers
// const vertices = new Float32Array([
//     //X,  Y,
//     -0.8, -0.8,
//     0.8, -0.8,
//     0.8, 0.8,
//     -0.8, 0.8,
// ]);
const vertices = new Float32Array([
    //   X,    Y,
    -0.8, -0.8, // Triangle 1 (Blue)
    0.8, -0.8,
    0.8, 0.8,

    -0.8, -0.8, // Triangle 2 (Red)
    0.8, 0.8,
    -0.8, 0.8,
]);

//Creates a GPU buffer, flags it for vertex usage and as a copy destination
//Then copies the vertices into the buffer
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

//Tells GPU how to read the buffer bytes
const vertexBufferLayout = {
    arrayStride: 8, // 2 floats per vertex, 4 bytes per float
    attributes: [{
        format: "float32x2", //2 floats per vertex: https://gpuweb.github.io/gpuweb/#enumdef-gpuvertexformat
        offset: 0, //there can be multiple attributes per element, their offsets are relative to the start of the element
        shaderLocation: 0, // Position, see vertex shader
    }],
};

// Create a uniform buffer that describes the grid; similar to vertex buffer
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// Create an array representing the active state of each cell.
const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);

// Create two storage buffers to hold the cell state.
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

// Enable one random cell in each column, then copy the JavaScript array 
// into the storage buffer.
for (let i = 0; i < GRID_SIZE; ++i) {
    cellStateArray[Math.floor(Math.random() * GRID_SIZE) * GRID_SIZE + i] = 1;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);


//Buffer for game controls
const controlArray = new Uint32Array(6);

const controlStorage = device.createBuffer({
    label: "Control Buffer",
    size: controlArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

controlArray[0] = 1;
document.addEventListener('keydown', function (event) {
    switch(event.key) {
        case " ":
            controlArray[0] = (controlArray[0] + 1) % 2;
            break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
            controlArray[event.key] = (controlArray[event.key] + 1) % 2;
            break;
    }
});

//Shaders are GPU code that defines how to process vertices
//takes shaderLocation 0 from vertexBefferLayout; float32x2=vec2f
//@vertex returns 4-dimensional vector, x y z w of the vertex
//pos can be used instead of pos.x, pos.y
//@fragment returns 4-dimensional vector, pixel color as r, g, b, a
const cellShaderModule = device.createShaderModule({
    label: "Cell shader",
    code: `
        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f,
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            let i = f32(input.instance); // Save the instance_index as a float
            let cell = vec2f(i % grid.x, floor(i / grid.x)); // Cell coordinates from bottom left
            let state = f32(cellState[input.instance]);

            let cellOffset = cell / grid * 2; //Canvas is actually 2x2, -1..+1
            let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

            var output: VertexOutput;
            output.pos = vec4f(gridPos, 0, 1);
            output.cell = cell;
            return output;
        }
        
        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            let c = input.cell / grid;
            return vec4f(c, 1 - c.x, 1);
        }
    `
});

// Create the compute shader that will process the simulation.
const simulationShaderModule = device.createShaderModule({
    label: "Game of Life simulation shader",
    code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;
        @group(0) @binding(3) var<storage> control: array<u32>;

        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) +
                    (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            // Check if cell above was active
            let isNext = cellActive(cell.x, cell.y+1);
            let i = cellIndex(cell.xy);

            if control[0] == 1 {
                switch isNext {
                    case 1: { // Cell with neighbor above becomes active.
                        cellStateOut[i] = 1;
                    }
                    default: { // Cells are inactive by default
                        cellStateOut[i] = 0;
                    }
                }
            } else {
                cellStateOut[i] = cellStateIn[i];
            }

        }`
});

// Create the bind group layout and pipeline layout.
const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {} // Grid uniform buffer
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Cell state input buffer
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Cell state output buffer
    }, {
        binding: 3,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Control buffer
    }]
});

const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});

//Render pipeline actually controls how the geometry is drawn, e.g. 
//which shaders to use and how to read buffers
const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout]
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

// Create a compute pipeline that updates the game state.
const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }, {
            binding: 1,
            resource: { buffer: cellStateStorage[0] }
        }, {
            binding: 2,
            resource: { buffer: cellStateStorage[1] }
        }, {
            binding: 3,
            resource: { buffer: controlStorage }
        }],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }, {
            binding: 1,
            resource: { buffer: cellStateStorage[1] }
        }, {
            binding: 2,
            resource: { buffer: cellStateStorage[0] }
        }, {
            binding: 3,
            resource: { buffer: controlStorage }
        }],
    })
];

let step = 0; // Track how many simulation steps have been run
function updateGrid() {
    device.queue.writeBuffer(controlStorage, 0, controlArray);
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    computePass.end();

    step++; // Increment the step count

    //Prepares a command for GPU: gets current canvas texture,
    //creates a view based on those dimensions,
    //sets clear on load and saves drawing results into the texture
    //at the end of render pass
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0.4, a: 1 }, //changes frame color
            storeOp: "store",
        }]
    });
    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices
    pass.end();

    //Creates buffer from finalized encoder
    //const commandBuffer = encoder.finish();

    //Sends an array of command buffers to the GPU for exectution
    //device.queue.submit([commandBuffer]);

    //Buffers are single-use, makes more sense to do in one-liner:
    device.queue.submit([encoder.finish()]);
}
// Schedule updateGrid() to run repeatedly
setInterval(updateGrid, UPDATE_INTERVAL);
