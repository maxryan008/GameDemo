use std::collections::HashMap;
use cgmath::Vector2;
use image::{GenericImage, RgbaImage};

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Rect {
    fn fits(&self, other: &Rect) -> bool {
        self.width >= other.width && self.height >= other.height
    }
}

fn calculate_total_area(images: &Vec<RgbaImage>) -> u32 {
    images.iter()
        .map(|img| {
            let (w, h) = img.dimensions();
            w * h
        })
        .sum()
}

fn maxrects_pack(
    images: Vec<(u32, RgbaImage)>,
    atlas_width: u32,
    atlas_height: u32
) -> Option<(RgbaImage, HashMap<u32, (Vec<Vector2<u32>>, u32, u32)>)> {
    let mut atlas = RgbaImage::new(atlas_width, atlas_height);
    let mut texture_map: HashMap<u32, (Vec<Vector2<u32>>, u32, u32)> = HashMap::new();

    let mut free_rectangles = vec![Rect { x: 0, y: 0, width: atlas_width, height: atlas_height }];

    for (id, img) in images.iter() {
        let (img_width, img_height) = img.dimensions();
        let mut placed = false;

        let img_rect = Rect { x: 0, y: 0, width: img_width, height: img_height };

        for i in 0..free_rectangles.len() {
            let free_rect = free_rectangles[i];

            if free_rect.fits(&img_rect) {
                atlas.copy_from(img, free_rect.x, free_rect.y).unwrap();

                if texture_map.contains_key(&id)
                {
                    texture_map.get_mut(&id).unwrap().0.push(Vector2::new(free_rect.x, free_rect.y));
                } else
                {
                    texture_map.insert(
                        id.clone(),
                        (Vec::from([Vector2::new(free_rect.x, free_rect.y)]), img_width, img_height)
                    );
                }



                let remaining_free_rectangles = split_free_rectangles(free_rect, img_rect);

                free_rectangles.remove(i);
                free_rectangles.extend(remaining_free_rectangles);

                placed = true;
                break;
            }
        }

        if !placed {
            return None;
        }
    }

    Some((atlas, texture_map))
}

fn split_free_rectangles(free_rect: Rect, placed_rect: Rect) -> Vec<Rect> {
    let mut result = vec![];

    if free_rect.width > placed_rect.width {
        result.push(Rect {
            x: free_rect.x + placed_rect.width,
            y: free_rect.y,
            width: free_rect.width - placed_rect.width,
            height: placed_rect.height,
        });
    }

    if free_rect.height > placed_rect.height {
        result.push(Rect {
            x: free_rect.x,
            y: free_rect.y + placed_rect.height,
            width: free_rect.width,
            height: free_rect.height - placed_rect.height,
        });
    }

    result
}

pub fn find_optimal_atlas_size(
    images: Vec<(u32, RgbaImage)>
) -> (RgbaImage, HashMap<u32, (Vec<Vector2<u32>>, u32, u32)>) {
    let total_area = calculate_total_area(&images.iter().map(|(_, img)| img.clone()).collect());

    let mut atlas_size = (f32::sqrt(total_area as f32).ceil() as u32).max(128);

    loop {
        if let Some((atlas, texture_map)) = maxrects_pack(images.clone(), atlas_size, atlas_size) {
            return (atlas, texture_map);
        }

        atlas_size *= 2;
    }
}