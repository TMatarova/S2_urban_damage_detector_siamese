import os
import rasterio
import numpy as np

def create_siamese_patches(before_path, after_path, mask_path, out_path,
                           patch_size=64, samples_per_class=1000):
    """Create before/after patch pairs with masks for Siamese CNN training. 
    Set at 64 x 64 image size and 1000 sample per class to ensure collection of all data. """

    # Load input raster files
    with rasterio.open(before_path) as src_b, rasterio.open(after_path) as src_a, rasterio.open(mask_path) as src_m:
        before_img, after_img, mask = src_b.read(), src_a.read(), src_m.read(1)

    H, W = mask.shape
    damaged, undamaged= [], []

    # Random sampling loop made until both sets are filled 
    # This is set to loop four times and stops for each sample which hits 1000, if it does not then the loop will collect the max amount
    # The loop is looped four times to ensure that there are 4000 attempts of finding damaged patches, to reduce missing any damaged patches.
    for _ in range(samples_per_class * 4):
        x, y = np.random.randint(0, W - patch_size), np.random.randint(0, H - patch_size)
        before_patch = before_img[:, y:y+patch_size, x:x+patch_size]
        after_patch = after_img[:, y:y+patch_size, x:x+patch_size]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]

        if before_patch.shape[1:] != (patch_size, patch_size):
            continue  # skip incomplete edge patches

        target = damaged if mask_patch.sum() > 0 else undamaged
        if len(target) < samples_per_class:
            target.append((before_patch, after_patch, mask_patch))

        if len(damaged) >= samples_per_class and len(undamaged) >= samples_per_class:
            break

    # Stack results into arrays
    before_array = np.array([p[0] for p in damaged + undamaged])
    after_array = np.array([p[1] for p in damaged + undamaged])
    masks_array = np.array([p[2] for p in damaged + undamaged])

    # Save dataset
    np.savez_compressed(out_path, before=before_array, after=after_array, masks=masks_array)
    print(f"saved {masks_array.shape[0]} pairs to {out_path}")


base_dir = r"C:\Users\Taula\Downloads\Ukraine cities"
out_dir = os.path.join(base_dir, "patches_siamese")
os.makedirs(out_dir, exist_ok=True)  # make sure folder exists

# File paths
# Each entry specifies the before image, after image, and mask file for a city
cities = [
    {
        "name": "Antonivka",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Antonivka\28072021\Antonivka_202107_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Antonivka\16102022\Antonivka_202210_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Antonivka\16102022\Antonivka_202210_CDA_mask.tif"
    },
    {
        "name": "Avdiivka",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Avdiivka\20211002\Avdiivka_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Avdiivka\20220920\Avdiivka_202209_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Avdiivka\20220920\Avdiivka_202209_CDA_mask.tif"
    },
    {
        "name": "Azovstal industrial",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Azovstal industrial\20210910\Azovstal_202109_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Azovstal industrial\20220528\Azovstal_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Azovstal industrial\20220528\Azovstal_202205_CDA_mask.tif"
    },
    {
        "name": "Bucha",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Bucha\20211022\Bucha_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Bucha\20220505\Bucha_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Bucha\20220505\Bucha_202205_CDA_mask.tif"
    },
    {
        "name": "Chernihiv",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Chernihiv\20211002\Chernihiv_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Chernihiv\20220505\Chernihiv_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Chernihiv\20220505\Chernihiv_202205_CDA_mask.tif"
    },
    {
        "name": "Hostomel",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Hostomel\20211022\Hostomel_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Hostomel\20220505\Hostomel_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Hostomel\20220505\Hostomel_202205_CDA_mask.tif"
    },
    {
        "name": "Irpin",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Irpin\20211022\Irpin_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Irpin\20220505\Irpin_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Irpin\20220505\Irpin_202205_CDA_mask.tif"
    },
    {
        "name": "Kharkiv",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kharkiv\202111\Kharkiv_202111_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kharkiv\202206\Kharkiv_202206_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kharkiv\202206\Kharkiv_202206_CDA_mask.tif"
    },
    {
        "name": "Kherson",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kherson\20210802\Kherson_202108_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kherson\20220807\Kherson_202208_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kherson\20220807\Kherson_202208_CDA_mask.tif"
    },
    {
        "name": "Kramatorsk",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kramatorsk\20210910\Kramatorsk_202109_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kramatorsk\20220809\Kramatorsk_202208_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kramatorsk\20220809\Kramatorsk_202208_CDA_mask.tif"
    },
    {
        "name": "Kremenchuk",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kremenchuk\202110\Kremenchuk_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kremenchuk\202206\Kremenchuk_202206_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Kremenchuk\202206\Kremenchuk_202206_CDA_mask.tif"
    },
    {
        "name": "Lysychansk",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Lysychansk\2772021\Lysychansk_202107_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Lysychansk\15102022\Lysychansk_202210_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Lysychansk\15102022\Lysychansk_202210_CDA_mask.tif"
    },
    {
        "name": "Makariv",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Makariv\20211002\Makariv_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Makariv\20220321\Makariv_202203_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Makariv\20220321\Makariv_202203_CDA_mask.tif"
    },
    {
        "name": "Melitopol",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Melitopol\20210625\Melitopol_202106_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Melitopol\20220918\Melitopol_202209_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Melitopol\20220918\Melitopol_202209_CDA_mask.tif"
    },
    {
        "name": "Mykolaiv",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Mykolaiv\20211026\Mykolaiv_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Mykolaiv\20220807\Mykolaiv_202208_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Mykolaiv\20220807\Mykolaiv_202208_CDA_mask.tif"
    },
    {
        "name": "Okhtyrka",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Okhtyrka\20210708\Okhtyrka_202107_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Okhtyrka\20220325\Okhtyrka_202203_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Okhtyrka\20220325\Okhtyrka_202203_CDA_mask.tif"
    },
    {
        "name": "Rubizhne",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Rubizhne\20210930\Rubizhne_202109_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Rubizhne\01082022\Rubizhne_20220801_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Rubizhne\01082022\Rubizhne_20220801_CDA_mask.tif"
    },
    {
        "name": "Shchastia",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Shchastia\20210917\Shchastia_202109_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Shchastia\20220707\Shchastia_202207_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Shchastia\20220707\Shchastia_202207_CDA_mask.tif"
    },
    {
        "name": "Sumy",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Sumy\20211001\Sumy_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Sumy\20220807\Sumy_202208_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Sumy\20220807\Sumy_202208_CDA_mask.tif"
    },
    {
        "name": "Trostianets",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Trostianets\20211001\Trostianets_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Trostianets\20220325\Trostianets_202203_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Trostianets\20220325\Trostianets_202203_CDA_mask.tif"
    },
    {
        "name": "Volnovakha",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Volnovakha\20210910\Volnovakha_202109_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Volnovakha\20220508\Volnovakha_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Volnovakha\20220508\Volnovakha_202205_CDA_mask.tif"
    },
    {
        "name": "Vorzel",
        "before": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Vorzel\20211022\Vorzel_202110_RGBNIR_clip.tif",
        "after": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Vorzel\20220505\Vorzel_202205_RGBNIR_clip.tif",
        "mask": r"C:\Users\Taula\Downloads\Ukraine cities\processed tifs\Vorzel\20220505\Vorzel_202205_CDA_mask.tif"
    }
]

# Run patches for the cities
for city in cities:
    out_path = os.path.join(out_dir, f"{city['name']}_pairs.npz")
    create_siamese_patches(city["before"], city["after"], city["mask"], out_path)
