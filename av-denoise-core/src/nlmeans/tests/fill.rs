/// The host version of [halo_slot_index](crate::nlmeans::kernels::fill::halo_slot_index).
fn halo_slot_index_host(halo: u32, block_x: u32, block_y: u32, patch_radius: u32) -> u32 {
    let tile_width = block_x + 2 * patch_radius;
    let top_count = patch_radius * tile_width;
    let side_width = 2 * patch_radius;
    let side_count = block_y * side_width;

    if halo < top_count {
        return halo;
    }

    let side = halo - top_count;
    if side < side_count {
        let side_row = side / side_width + patch_radius;
        let side_column = side % side_width;
        let column = if side_column < patch_radius {
            side_column
        } else {
            side_column + block_x
        };
        return side_row * tile_width + column;
    }

    let bottom = side - side_count;
    (block_y + patch_radius) * tile_width + bottom
}

#[test]
fn own_pixels_and_halo_slots_cover_the_tile_once() {
    let shapes = [
        (32u32, 8u32, 2u32),
        (32, 16, 4),
        (16, 16, 3),
        (64, 32, 4),
        (8, 8, 1),
    ];
    for (block_x, block_y, patch_radius) in shapes {
        let tile_width = block_x + 2 * patch_radius;
        let tile_height = block_y + 2 * patch_radius;
        let tile_elems = tile_width * tile_height;
        let halo_elems = tile_elems - block_x * block_y;
        let mut seen = vec![false; tile_elems as usize];

        for own_y in patch_radius..patch_radius + block_y {
            for own_x in patch_radius..patch_radius + block_x {
                let own_index = own_y * tile_width + own_x;
                seen[own_index as usize] = true;
            }
        }

        for halo in 0..halo_elems {
            let index = halo_slot_index_host(halo, block_x, block_y, patch_radius);
            assert!(index < tile_elems, "halo slot {halo} maps past the tile");
            assert!(!seen[index as usize], "halo slot {halo} repeats element {index}");
            seen[index as usize] = true;
        }
    }
}
