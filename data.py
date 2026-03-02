import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, Resized, EnsureTyped,
    RandRotated, RandFlipd, CropForegroundd, ToTensord
)
from monai.data import Dataset, DataLoader

def get_loaders(tsv_path, batch_size=2, use_mask=True, perform_crop=True):
    df = pd.read_csv(tsv_path, sep="\t")

    # --- column checks ---
    required = ["image_path", "status", "figo_stage"]
    if use_mask:
        required.append("mask_path")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in TSV: {missing}")

    # labels
    status_map = {"malignant": 0, "benign": 1, "normal": 2}
    df["status_label"] = df["status"].map(status_map)

    figo_categories = sorted(df["figo_stage"].unique())
    figo_map = {stage: i for i, stage in enumerate(figo_categories)}
    df["figo_label"] = df["figo_stage"].map(figo_map)

    # patient id from parent folder of image_path
    if "patient_id" not in df.columns:
        df["patient_id"] = df["image_path"].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])

    # split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["patient_id"]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    keys = ["image", "mask"] if use_mask else ["image"]

    def get_transforms(is_train=True):
        nodes = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
        ]

        modes = ("bilinear", "nearest") if use_mask else "bilinear"
        nodes.append(Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=modes))

        if use_mask and perform_crop:
            nodes.append(CropForegroundd(keys=keys, source_key="mask", allow_smaller=True))

        nodes.append(
            ScaleIntensityRanged(
                keys=["image"], a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True
            )
        )

        res_modes = ("trilinear", "nearest") if use_mask else "trilinear"
        nodes.append(Resized(keys=keys, spatial_size=(128, 128, 64), mode=res_modes))

        if is_train:
            nodes.append(RandRotated(keys=keys, range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3))
            nodes.append(RandFlipd(keys=keys, spatial_axis=[0, 1], prob=0.2))

        # type for image/mask only
        nodes.append(EnsureTyped(keys=keys))
        # convert labels to tensors (optional but clean)
        nodes.append(ToTensord(keys=["status", "figo"]))
        return Compose(nodes)

    def to_dict(df_split):
        data_list = []
        for _, r in df_split.iterrows():
            item = {
                "image": r["image_path"],
                "status": r["status_label"],
                "figo": r["figo_label"],
            }
            if use_mask:
                item["mask"] = r["mask_path"]
            data_list.append(item)
        return data_list

    train_ds = Dataset(data=to_dict(train_df), transform=get_transforms(True))
    val_ds   = Dataset(data=to_dict(val_df),   transform=get_transforms(False))
    test_ds  = Dataset(data=to_dict(test_df),  transform=get_transforms(False))

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_ds, batch_size=1, shuffle=False),
        "test":  DataLoader(test_ds, batch_size=1, shuffle=False),
        "maps":  (status_map, figo_map),
    }