// Vertex shader

struct Camera {
    view_pos: vec4<f32>, // Camera position (optional, if used for shading)
    view_proj: mat4x4<f32>, // View-Projection matrix (to transform vertices)
}

@group(1) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,  // Vertex position
    @location(1) tex_coords: vec2<f32>, // Texture coordinates
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>, // 1st row of model matrix
    @location(6) model_matrix_1: vec4<f32>, // 2nd row of model matrix
    @location(7) model_matrix_2: vec4<f32>, // 3rd row of model matrix
    @location(8) model_matrix_3: vec4<f32>, // 4th row of model matrix
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>, // Clip space position
    @location(0) tex_coords: vec2<f32>, // Texture coordinates passed to fragment shader
}

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // Construct model matrix from instance input
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    // Prepare output
    var out: VertexOutput;
    out.tex_coords = vertex.tex_coords; // Pass texture coordinates through
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0); // Transform vertex

    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>; // Diffuse texture
@group(0) @binding(1)
var s_diffuse: sampler; // Sampler

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the texture using texture coordinates from the vertex shader
    let color = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    // Return the sampled color
    return color;
    //return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}