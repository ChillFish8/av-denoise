use crate::nlmeans::kernels::fill::halo_slot_index_host;

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
