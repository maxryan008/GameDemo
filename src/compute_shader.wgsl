struct TextureMapEntry {
    position: vec2<f32>,
    width: f32,
    height: f32,
};

struct PackedVec3 {
    x: f32,
    y: f32,
    z: f32,
};

struct PackedVec2 {
    x: f32,
    y: f32,
};

struct ChunkMeshVertex {
    position: PackedVec3,
    tex_coords: PackedVec2,
};

// Bindings for flat array inputs, texture atlas, and seed
@group(0) @binding(0) var<storage, read> vertices_flat: array<PackedVec3>;
@group(0) @binding(1) var<storage, read> indices_flat: array<u32>;
@group(0) @binding(2) var<storage, read> blocks_flat: array<u32>;
@group(0) @binding(3) var<storage, read> tints_flat: array<PackedVec3>;
@group(0) @binding(4) var<storage, read> texture_map: array<TextureMapEntry>;
@group(0) @binding(5) var atlas_texture: texture_2d<f32>;  // The atlas texture containing all block variants
@group(0) @binding(6) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<uniform> seed: u32; // Seed for random number generation

// Bindings for input of vec sizes in the flat arrays
@group(1) @binding(0) var<storage, read> rects_vertices_sizes: array<u32>;
@group(1) @binding(1) var<storage, read> rects_indices_sizes: array<u32>;
@group(1) @binding(2) var<storage, read> rects_blocks_sizes: array<u32>;
@group(1) @binding(3) var<storage, read> rects_tints_sizes: array<u32>;
@group(1) @binding(4) var<storage, read> variant_lengths: array<u32>;  // New buffer for variant lengths
// Buffers for output (vertex and index data)
@group(1) @binding(5) var<storage, read_write> vertex_output: array<ChunkMeshVertex>;
@group(1) @binding(6) var<storage, read_write> index_output: array<u32>;
@group(1) @binding(7) var<storage, read> rect_widths: array<u32>;

@group(2) @binding(0) var<storage, read> rect_positions: array<vec2<u32>>;

// Random number generation based on a seed
fn random(seed: u32, index: u32, x: u32, y: u32) -> u32 {
    return (seed + index * 1664525u * x + 1013904223u * y) & 0xFFFFFFFFu;
}

// Function to calculate normalized texture coordinates for each block
fn get_texture_coords(
    width: u32,
    height: u32,
    pos: vec2<u32>,
    output_size: PackedVec2,
    vertex_index: u32 // 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
) -> PackedVec2 {
    // Base normalized coordinates based on the rectangle's position
    let base_tex_x = f32(pos.x) / output_size.x;
    let base_tex_y = f32(pos.y) / output_size.y;

    // Add width and height to calculate the coordinates based on the vertex
    var tex_x: f32 = base_tex_x;
    var tex_y: f32 = base_tex_y;

    // Adjust tex_x and tex_y based on which corner (vertex) we are calculating for
    if (vertex_index == 1u || vertex_index == 2u) {
        // Top-right or bottom-right corner
        tex_x = base_tex_x + f32(width) / output_size.x;
    }
    if (vertex_index == 2u || vertex_index == 3u) {
        // Bottom-right or bottom-left corner
        tex_y = base_tex_y + f32(height) / output_size.y;
    }

    return PackedVec2(tex_x, tex_y);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rect_index = global_id.x;

    // Get rect position in the output texture (top-left corner)
    let rect_pos = rect_positions[rect_index];

    // Define array starts and ends for the current rect being rendered
    let vertices_start = rects_vertices_sizes[rect_index];
    let indices_start = rects_indices_sizes[rect_index];
    let blocks_start = rects_blocks_sizes[rect_index];
    let tints_start = rects_tints_sizes[rect_index];
    let vertices_end = rects_vertices_sizes[rect_index + 1];
    let indices_end = rects_indices_sizes[rect_index + 1];
    let blocks_end = rects_blocks_sizes[rect_index + 1];
    let tints_end = rects_tints_sizes[rect_index + 1];

    let rect_width = rect_widths[rect_index];
    let rect_height = (blocks_end - blocks_start) / rect_width;

    let block_size = vec2<f32>(1.0 / f32(rect_width), 1.0 / f32(rect_height));  // Normalized size of each block

    // Process vertices and write to the output buffer as ChunkMeshVertex
    for (var i = vertices_start; i < vertices_end; i = i + 1u) {
        let vertex_position = vertices_flat[i];

        // Compute block position within the rect (row and column)
        let block_index = (i - vertices_start);  // Cycle through blocks
        let block_x = block_index % rect_width;
        let block_y = block_index / rect_width;

        // Get the block type from blocks_flat
        let block_type = blocks_flat[blocks_start + block_index];

        // Get the number of cumulative variants up to that block
        let num_variants_cumulative = variant_lengths[block_type];
        let num_variants_previous_cumulative = select(0u, variant_lengths[block_type - 1u], block_type > 0u);
        let num_variants = num_variants_cumulative - num_variants_previous_cumulative;

        // If num_variants > 0, select a variant, otherwise fallback to block_type directly
        var selected_variant_index: u32 = 0u;
        if num_variants > 0u {
            let random_value = random(seed, block_index, block_x, block_y);
            selected_variant_index = random_value % num_variants;  // Randomly select a variant
        }

        // Compute the base index in the texture map using the previous size plus the selected variant index
        let variant_base_index = num_variants_previous_cumulative + selected_variant_index;

        // Get the texture map entry for the selected variant
        let texture_entry = texture_map[variant_base_index];

        // Compute the normalized texture coordinates for this vertex
        let vertex_index = (i - vertices_start);  // Determines which corner we are dealing with
        //todo make output texture size automatically calculated
        let tex_coords = get_texture_coords(rect_width, rect_height, rect_pos, PackedVec2(32.0, 32.0), vertex_index);

        // Write vertex data to output
        vertex_output[i] = ChunkMeshVertex(
            vertex_position,
            tex_coords
        );
    }

    // Write processed index data to the index output buffer
    for (var i = indices_start; i < indices_end; i = i + 1u) {
        index_output[i] = indices_flat[i] + rect_index * 4;
    }

    // Write the blocks to the output texture using the correct normalized texture coordinates
    for (var block_x = 0u; block_x < rect_height; block_x = block_x + 1u) {
        for (var block_y = 0u; block_y < rect_width; block_y = block_y + 1u) {
            let block_index = block_y * rect_height + block_x;

            // Get the block type from blocks_flat
            let block_type = blocks_flat[blocks_start + block_index];
            // Get the tint from the tints_flat
            let tint = tints_flat[tints_start + block_index];

            // Get the number of cumulative variants up to that block
            let num_variants_cumulative = variant_lengths[block_type];
            let num_variants_previous_cumulative = select(0u, variant_lengths[block_type - 1u], block_type > 0u);
            let num_variants = num_variants_cumulative - num_variants_previous_cumulative;

            // If num_variants > 0, select a variant, otherwise fallback to block_type directly
            var selected_variant_index: u32 = 0u;
            if num_variants > 0u {
                let random_value = random(seed, block_index, block_x, block_y);
                selected_variant_index = random_value % num_variants;  // Randomly select a variant
            }

            // Compute the base index in the texture map using the previous size plus the selected variant index
            let variant_base_index = num_variants_previous_cumulative + selected_variant_index;

            // Get the texture map entry for the selected variant
            let texture_entry = texture_map[variant_base_index];

            // Sample the atlas texture for this block and apply the tint
            for (var pixel_x = 0u; pixel_x < 4u; pixel_x = pixel_x + 1u) {
                for (var pixel_y = 0u; pixel_y < 4u; pixel_y = pixel_y + 1u) {
                    let tex_coords_int = vec2<i32>(
                        (i32(f32(texture_entry.position.x) / 0.03125f) * 4) + i32(pixel_x),
                        (i32(f32(texture_entry.position.y) / 0.03125f) * 4) + i32(pixel_y)
                    );

                    let tex_color = textureLoad(atlas_texture, tex_coords_int, 0) + vec4<f32>(tint.x, tint.y, tint.z, 0);
                    // Write the sampled texture color to the correct location on the output texture
                    let output_coords = vec2<i32>(
                        i32(rect_pos.x) * 4 + i32(block_x) * 4 + i32(pixel_x),
                        i32(rect_pos.y) * 4 + i32(block_y) * 4 + i32(pixel_y)
                    );

                    textureStore(output_texture, output_coords, tex_color);
                }
            }
        }
    }
}