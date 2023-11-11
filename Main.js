const GRID_SIZE_X = 5;
const GRID_SIZE_Y = 6;
const UPDATE_INTERVAL = 10;
const TIMEOUT_INTERVAL = 1000;
const WORKGROUP_SIZE = 8;
let balance = 100;

const HandValues = {
    0: "Nothing",
    10: "Pair",
    15: "Two pairs",
    20: "Three of a kind",
    30: "Five-high straight",
    35: "Six-high straight",
    40: "Full house",
    50: "Four of a kind",
    60: "Five of a kind",
}

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
const uniformArray = new Float32Array([GRID_SIZE_X, GRID_SIZE_Y]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// Create an array representing the active state of each cell.
const cellStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);

// Create two storage buffers to hold the cell state.
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
];

// An extra buffer to read out the cell state for game logic
const cellStateReadStorage =
    device.createBuffer({
        label: "Cell State Read",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

// Enable one random cell in each column, then copy the JavaScript array 
// into the storage buffer.
for (let i = 0; i < GRID_SIZE_X; ++i) {
    cellStateArray[Math.floor(Math.random() * GRID_SIZE_Y) * GRID_SIZE_X + i] = 1;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);


//Buffer for game controls
const controlArray = new Uint32Array([0, 0, 0, 0, 0, 0]);

const controlStorage = device.createBuffer({
    label: "Control Buffer",
    size: controlArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
    label: "Logic shader",
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

            if (control[0] == 3) & (control[cell.x + 1] == 1) {
                cellStateOut[i] = cellStateIn[i];
            } else {
                switch isNext {
                    case 1: { // Cell with neighbor above becomes active.
                        cellStateOut[i] = 1;
                    }
                    default: { // Cells are inactive by default
                        cellStateOut[i] = 0;
                    }
                }
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

//#region Dice Pipeline
const DICE_GRID_SIZE_X = 25;
const DICE_GRID_SIZE_Y = 30;

// Create a uniform buffer that describes the grid; similar to vertex buffer
const diceUniformArray = new Float32Array([DICE_GRID_SIZE_X, DICE_GRID_SIZE_Y]);
const diceUniformBuffer = device.createBuffer({
    label: "Dice grid uniforms",
    size: diceUniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(diceUniformBuffer, 0, diceUniformArray);

// Create an array representing the active state of each die.
const diceStateArray = new Uint32Array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]);
console.log(diceStateArray.length);

// Create two storage buffers to hold the dice state.
const diceStateStorage =
    device.createBuffer({
        label: "Dice state",
        size: diceStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

// Randomly enable dice for testing
// for (let i = 0; i < diceStateArray.length; ++i) {
//     diceStateArray[i] = Math.random() < 0.6 ? 0 : 1;
// }
device.queue.writeBuffer(diceStateStorage, 0, diceStateArray);

const diceBindGroupLayout = device.createBindGroupLayout({
    label: "Dice Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {} // Grid uniform buffer
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Dice state input buffer
    }]
});

const diceBindGroup =
    device.createBindGroup({
        label: "Dice renderer bind group",
        layout: diceBindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: diceUniformBuffer }
        }, {
            binding: 1,
            resource: { buffer: diceStateStorage }
        }],
    });

const dicePipelineLayout = device.createPipelineLayout({
    label: "Dice Pipeline Layout",
    bindGroupLayouts: [diceBindGroupLayout],
});


const diceShaderModule = device.createShaderModule({
    label: "Dice shader",
    code: `
        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) dice: vec2f,
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> diceState: array<u32>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            let i = f32(input.instance); // Save the instance_index as a float
            let dice = vec2f(i % grid.x, floor(i / grid.x)); // dice coordinates from bottom left
            let state = f32(diceState[input.instance]);

            let diceOffset = dice / grid * 2; //Canvas is actually 2x2, -1..+1
            let gridPos = (input.pos * state + 1) / grid - 1 + diceOffset;

            var output: VertexOutput;
            output.pos = vec4f(gridPos, 0, 1);
            output.dice = dice;
            return output;
        }
        
        @fragment
        fn fragmentMain() -> @location(0) vec4f {
            return vec4f(0, 0, 0.4, 1);
        }
    `
});

const dicePipeline = device.createRenderPipeline({
    label: "Dice pipeline",
    layout: dicePipelineLayout,
    vertex: {
        module: diceShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout]
    },
    fragment: {
        module: diceShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
});
//#endregion

let step = 0; // Track how many simulation steps have been run
let countdown = 200;
function updateGrid() {
    device.queue.writeBuffer(controlStorage, 0, controlArray);
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    const workgroupCount = Math.ceil(GRID_SIZE_X * GRID_SIZE_Y / WORKGROUP_SIZE ** 2);
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
    pass.draw(vertices.length / 2, GRID_SIZE_X * GRID_SIZE_Y); // 6 vertices
    pass.setPipeline(dicePipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, diceBindGroup);
    pass.draw(vertices.length / 2, DICE_GRID_SIZE_X * DICE_GRID_SIZE_Y); // 6 vertices
    pass.end();

    encoder.copyBufferToBuffer(cellStateStorage[step % 2], 0, cellStateReadStorage, 0, cellStateArray.byteLength);
    //Buffers are single-use, makes more sense to do in one-liner:
    device.queue.submit([encoder.finish()]);
}

//Preview frame
updateGrid();

//Process user input
document.addEventListener('keydown', function (event) {
    switch (event.key) {
        case " ":
            //bet 10 if we are in a new round
            if (controlArray[0] == 0) {
                balance -= 10;
            }
            document.getElementById("balance").textContent = balance;

            if (controlArray[0] % 2 == 1) break; //Ignore spacebar if already running

            controlArray[0] += 1; //1 or 3 - running the first/second roll
            document.getElementById("round").textContent = "Rolling... ";

            // Schedule updateGrid() to run repeatedly
            const mainLoop = setInterval(updateGrid, UPDATE_INTERVAL)
            setTimeout(() => {
                clearInterval(mainLoop);
                controlArray[0] += 1; //2 or 4 - finished 1st/2nd roll

                if (controlArray[0] == 2) {
                    document.getElementById("round").textContent = "First roll - choose your holds!";
                }
                else if (controlArray[0] >= 4) {
                    getCellState().then(result => {
                        let winnings = evaluateHand(calculateHand(result));
                        balance += winnings;
                        document.getElementById("balance").textContent = balance;
                        document.getElementById("round").textContent = HandValues[winnings] + "! ";
                        controlArray.fill(0);
                        cellStateArray.fill(0);
                        for (let i = 0; i < GRID_SIZE_X; ++i) {
                            cellStateArray[Math.floor(Math.random() * GRID_SIZE_Y) * GRID_SIZE_X + i] = 1;
                        }
                        device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);
                        device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);
                    });

                };


            }, TIMEOUT_INTERVAL);

            break;
        //hold dice
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
            if (controlArray[0] == 2) //allow holding only after first roll
            {
                controlArray[event.key] = (controlArray[event.key] + 1) % 2;
            }
            break;
    }
    console.log(controlArray);
});


async function getCellState() {
    await cellStateReadStorage.mapAsync(
        GPUMapMode.READ,
        0,
        cellStateArray.byteLength
    );
    const copyArrayBuffer = cellStateReadStorage.getMappedRange(0, cellStateArray.byteLength);
    const data = copyArrayBuffer.slice();
    cellStateReadStorage.unmap();
    //console.log(new Uint32Array(data));
    return data;
}

function calculateHand(cellStates) {
    const cells = new Uint32Array(cellStates);
    const hand = Array(5);
    for (let i = 0; i < cells.length; i++) {
        if (cells[i] == 1) {
            hand[i % GRID_SIZE_X] = Math.floor(i / GRID_SIZE_X) + 1;

        };
    };
    return hand;
}

function evaluateHand(hand) {
    const counts = Array(6).fill(0);
    for (let i = 0; i < hand.length; i++) {
        counts[hand[i] - 1]++;
    }
    console.log(counts);
    if (counts.includes(5)) {
        return 60;
    }
    if (counts.includes(4)) {
        return 50;
    }
    if (counts.includes(3) && counts.includes(2)) {
        return 40;
    }
    if ((counts.filter(x => x == 1).length == 5) && (counts[0] == 0)) {
        return 35;
    }
    if ((counts.filter(x => x == 1).length == 5) && (counts[5] == 0)) {
        return 30;
    }
    if (counts.includes(3)) {
        return 20;
    }
    if (counts.filter(x => x == 2).length == 2) {
        return 15;
    }
    if (counts.includes(2)) {
        return 10;
    }
    return 0;
}

