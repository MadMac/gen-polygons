// Compute shader for fitness calculation
// This shader calculates the difference between goal image and rendered image
// and computes a fitness score based on the differences.

struct FitnessParams {
    image_width: u32;
    image_height: u32;
    sample_step: u32;
    padding: u32;
    color_weight: u32;
    structure_weight: u32;
};

struct FitnessResult {
    value: u32;
};

// Helper function for RGB to grayscale conversion
fn rgb_to_grayscale(color: vec3<f32>) -> f32 {
    return color.r * 0.299 + color.g * 0.587 + color.b * 0.114;
}

// Multi-scale fitness evaluation
// Returns fitness value for a given scale (0.0-1.0)
fn evaluate_at_scale(texture: texture_2d<f32>, goal_texture: texture_2d<f32>, 
                    x: u32, y: u32, scale: f32) -> f32 {
    // For now, simple implementation
    // In future: actual multi-scale rendering and comparison
    return 1.0 - perceptual_color_diff(
        textureLoad(texture, vec2<i32>(i32(x), i32(y)), 0).rgb,
        textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0).rgb
    );
}

// Helper function for perceptual color difference
fn perceptual_color_diff(a: vec3<f32>, b: vec3<f32>) -> f32 {
    // Use weighted RGB difference based on human vision sensitivity
    let r_diff = abs(a.r - b.r);
    let g_diff = abs(a.g - b.g);
    let b_diff = abs(a.b - b.b);
    
    // Human vision is most sensitive to green, then red, then blue
    return r_diff * 0.3 + g_diff * 0.59 + b_diff * 0.11;
}



[[group(0), binding(0)]]
var<uniform> params: FitnessParams;

[[group(0), binding(1)]]
var goal_texture: texture_2d<f32>;

[[group(0), binding(2)]]
var rendered_texture: texture_2d<f32>;

[[group(0), binding(3)]]
var<storage, read_write> fitness_result: FitnessResult;

[[stage(compute), workgroup_size(8, 8, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    // Calculate pixel coordinates
    let x = global_id.x;
    let y = global_id.y;
    
    // Check bounds
    if (x >= params.image_width || y >= params.image_height) {
        return;
    }
    
    // Apply sampling step for performance
    if (x % params.sample_step != 0u || y % params.sample_step != 0u) {
        return;
    }
    
    // Sample textures
    let goal_pixel = textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0);
    let rendered_pixel = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0);
    
    // Calculate perceptual color difference
    let color_diff = perceptual_color_diff(goal_pixel.rgb, rendered_pixel.rgb);
    
    // Simplified edge difference (temporarily disabled for compatibility)
    let edge_diff = 0.0;
    
    // Combined fitness metric with adaptive weights from params
    let color_weight = f32(params.color_weight) / 1000.0;
    let structure_weight = f32(params.structure_weight) / 1000.0;
    let combined_diff = color_diff * color_weight + edge_diff * structure_weight;
    
    // Multi-scale fitness evaluation
    let scales = array<f32, 3>(0.5, 0.75, 1.0);
    let scale_weights = array<f32, 3>(0.3, 0.3, 0.4);
    
    var multi_scale_fitness: f32 = 0.0;
    for (var i: u32 = 0; i < 3; i++) {
        let scale_fitness = evaluate_at_scale(
            rendered_texture, goal_texture, x, y, scales[i]
        );
        multi_scale_fitness += scale_fitness * scale_weights[i];
    }
    
    // Combine single-scale and multi-scale fitness
    let combined_diff = combined_diff * 0.7 + (1.0 - multi_scale_fitness) * 0.3;
    
    // Convert to fitness value (lower difference = higher fitness)
    // Use sigmoid-like curve for better fitness distribution
    let max_possible_diff = 1.0; // Normalized maximum difference
    let normalized_diff = combined_diff / max_possible_diff;
    
    // Apply sigmoid function for better fitness scaling
    let fitness_value = u32((1.0 - normalized_diff) * 1000000.0);
    
    // Edge bonus temporarily disabled for compatibility
    // let edge_bonus = u32((1.0 - min(edge_diff / 0.5, 1.0)) * 200000.0);
    // fitness_value += edge_bonus;
    
    // For this demo, we'll just store the fitness for a representative sample
    // In a real implementation, you'd want to accumulate properly
    if (x == params.image_width / 2u && y == params.image_height / 2u) {
        fitness_result.value = fitness_value;
    }
}