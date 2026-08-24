import collections
import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

DEBUG = False  # debug mode flag


def get_all_action_annotations_entries(
    root_dir, map_label, delete_base_dirs=None, drop_unknown_label=True
):
    full_dataset_entries = list()

    annotation_file = os.path.join(root_dir, "master.csv")

    with open(annotation_file, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            fields = line.strip().split(",")
            if len(fields) != len(header):
                continue
            info = dict(zip(header, fields))

            label = map_label.get(info["Label"], -1)
            if drop_unknown_label and label == -1:
                continue
            info["label"] = label

            for delete_base_dir in delete_base_dirs:
                info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                info["pose_path"] = info["pose_path"].replace(delete_base_dir, "")
            info["image_path"] = os.path.join(root_dir, info["image_path"])
            info["pose_path"] = os.path.join(root_dir, info["pose_path"])

            full_dataset_entries.append(info)

    annotation_file = os.path.join(root_dir, "additive_lying_dataset.csv")
    use_additive_lying_dataset = os.path.exists(annotation_file)
    if use_additive_lying_dataset:
        with open(annotation_file, "r") as f:
            header = f.readline().strip().split(",")
            for line in f:
                fields = line.strip().split(",")
                if len(fields) != len(header):
                    continue
                info = dict(zip(header, fields))

                label = map_label.get(info["Label"], -1)
                if drop_unknown_label and label == -1:
                    continue
                info["label"] = label

                for delete_base_dir in delete_base_dirs:
                    info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                    info["pose_path"] = info["pose_path"].replace(delete_base_dir, "")
                info["image_path"] = os.path.join(root_dir, info["image_path"])
                info["pose_path"] = os.path.join(root_dir, info["pose_path"])

                full_dataset_entries.append(info)

    annotation_file = os.path.join(root_dir, "additive_riding_dataset.csv")
    use_additive_riding_dataset = os.path.exists(annotation_file)
    if use_additive_riding_dataset:
        with open(annotation_file, "r") as f:
            header = f.readline().strip().split(",")
            for line in f:
                fields = line.strip().split(",")
                if len(fields) != len(header):
                    continue
                info = dict(zip(header, fields))

                label = map_label.get(info["Label"], -1)
                if drop_unknown_label and label == -1:
                    continue
                info["label"] = label

                for delete_base_dir in delete_base_dirs:
                    info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                    info["pose_path"] = info["pose_path"].replace(delete_base_dir, "")
                info["image_path"] = os.path.join(root_dir, info["image_path"])
                info["pose_path"] = os.path.join(root_dir, info["pose_path"])

                full_dataset_entries.append(info)

    return full_dataset_entries


def get_all_interaction_annotations_entries(
    root_dir,
    map_label,
    delete_base_dirs=None,
    use_more_than_three_cattles=False,
    supplemental_map_label=None,
):
    full_dataset_entries = list()
    supplemental_map_label = {
        "no_interaction": 0,
        "interest": 1,
        "conflict": 2,
        "mount": 3,
    }
    annotation_file = os.path.join(root_dir, "master_v3.csv")

    with open(annotation_file, "r") as f:
        header = f.readline().strip().split(",")
        for i, line in enumerate(f):
            fields = line.strip().split(",")
            if len(fields) != len(header):
                continue
            info = dict(zip(header, fields))

            if (
                not use_more_than_three_cattles
                and "more_than_three" in info["label_v1"]
            ):
                continue

            info["label"] = map_label.get(info["label_v1"], -1)
            if info["label"] == -1 and info["label_v1"] == "interaction":
                info["label"] = map_label.get(info["label_v2"], -1)
            elif len(map_label) == 2 and info["label"] == 1:
                info["supplemental_label"] = supplemental_map_label.get(
                    info["label_v2"], -1
                )

            if info["label"] == -1:
                continue

            # add bbox info
            info["bbox1_xyxy"] = info.get("bbox1_xyxy", "[0 0 0 0]")
            info["bbox2_xyxy"] = info.get("bbox2_xyxy", "[0 0 0 0]")
            info["merged_bbox_xyxy"] = info.get("merged_bbox_xyxy", "[0 0 0 0]")

            for delete_base_dir in delete_base_dirs:
                info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                info["pose_path_1"] = info["pose_path_1"].replace(delete_base_dir, "")
                info["pose_path_2"] = info["pose_path_2"].replace(delete_base_dir, "")

            info["image_path"] = os.path.join(root_dir, info["image_path"])
            info["pose_path_1"] = os.path.join(root_dir, info["pose_path_1"])
            info["pose_path_2"] = os.path.join(root_dir, info["pose_path_2"])

            if not os.path.exists(info["image_path"]):
                print(f"[WARN] Image path does not exist: {info['image_path']}")
                continue

            full_dataset_entries.append(info)

    return full_dataset_entries


def split_action_dataset_entries(full_dataset_entries, split_type="filename"):
    train_entries, val_entries, test_entries = [], [], []
    if split_type == "date":
        for entry in full_dataset_entries:
            if "train" in entry["image_path"]:
                train_entries.append(entry)
            elif "val" in entry["image_path"]:
                val_entries.append(entry)
            elif "test" in entry["image_path"]:
                test_entries.append(entry)
    elif split_type == "stratified":
        train_val_entries = []
        for entry in full_dataset_entries:
            if "test" in entry["image_path"]:
                test_entries.append(entry)
            else:
                train_val_entries.append(entry)
        # split per label
        train_entries, val_entries = train_test_split(
            train_val_entries,
            test_size=0.2,
            stratify=[e["label"] for e in train_val_entries],
            random_state=42,
        )
    else:
        raise NotImplementedError(f"Unknown split_type: {split_type}")

    return train_entries, val_entries, test_entries


def split_interaction_dataset_entries(full_dataset_entries, split_type="handle"):
    if split_type == "video_ids":
        for entry in full_dataset_entries:
            # 2. group entries by video clip ID
            entries_by_clip = collections.defaultdict(list)
            for entry in full_dataset_entries:
                filename = os.path.basename(entry["image_path"])
                clip_id = "_".join(filename.split("_")[:3])
                entries_by_clip[clip_id].append(entry)

            # 3. split the clip-ID list chronologically into train/val/test
            clip_ids = sorted(list(entries_by_clip.keys()))  # sort to keep chronological order
            train_val_clip_ids, test_clip_ids = train_test_split(
                clip_ids, test_size=0.2, shuffle=False
            )
            train_clip_ids, val_clip_ids = train_test_split(
                train_val_clip_ids, test_size=(1 / 8), shuffle=False
            )

            # 4. rebuild entry lists from the split IDs
            train_entries = [
                entry
                for clip_id in train_clip_ids
                for entry in entries_by_clip[clip_id]
            ]
            val_entries = [
                entry for clip_id in val_clip_ids for entry in entries_by_clip[clip_id]
            ]
            test_entries = [
                entry for clip_id in test_clip_ids for entry in entries_by_clip[clip_id]
            ]
    elif split_type == "stratified":
        full_dataset_df = pd.DataFrame(full_dataset_entries)
        full_dataset_df["entries"] = full_dataset_df["source_video"].str.replace(
            ".avi", ""
        )

        test_1_entries = full_dataset_df.sort_values(
            by=["entries", "frame_number"], ascending=False
        ).iloc[:149]["entries"]
        conflict_df = full_dataset_df.loc[full_dataset_df["label"] == 2]
        test_conflict_entries = sorted(conflict_df["entries"].unique())[:2]
        test_entries = sorted(set(test_1_entries) | set(test_conflict_entries))
        test_df = full_dataset_df[full_dataset_df["entries"].isin(test_entries)]
        train_val_df = full_dataset_df.drop(test_df.index)

        X = train_val_df.index  # use the index as the feature
        y = train_val_df["label"]
        groups = train_val_df["entries"]

        # --- Stage 1: split into Train and Temp (Val+Test) (80% / 20%) ---
        # n_splits=5 splits 1/5 (20%) off for the test set
        n_splits_train_test = 5
        sgkf_train_test = StratifiedGroupKFold(
            n_splits=n_splits_train_test, shuffle=True, random_state=42
        )

        # take the first split from the iterator
        train_idx, val_idx = next(sgkf_train_test.split(X, y, groups))

        # split the DataFrame
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_entries = train_df.to_dict("records")
        val_entries = val_df.to_dict("records")
        test_entries = test_df.to_dict("records")
    elif split_type == "handle":
        test_videos = [
            # version 1
            "2025-03-05 17-10-00~17-20-00.avi",
            "2019-03-23 16-50-00~17-00-00.avi",
            # version 2
            '2025-03-05 14-10-00~14-19-59.avi',
            '2025-03-05 16-40-00~16-50-00.avi',
        ]

        val_videos = [
            # version 1
            "2025-03-05 17-30-00~17-39-59.avi",
            "2021-09-24 08-10-00~08-20-00.avi",
            # version 2
            '2025-03-05 17-00-00~17-10-00.avi',
        ]

        full_df = pd.DataFrame(full_dataset_entries)

        # split the DataFrame.
        test_df = full_df[full_df["source_video"].isin(test_videos)].copy()
        val_df = full_df[full_df["source_video"].isin(val_videos)].copy()
        train_df = full_df[~full_df["source_video"].isin(test_videos + val_videos)].copy()

        train_entries = train_df.to_dict("records")
        val_entries = val_df.to_dict("records")
        test_entries = test_df.to_dict("records")
    else:
        raise NotImplementedError(f"Unknown split_type: {split_type}")

    return train_entries, val_entries, test_entries


class CattleInteractionDataset(Dataset):
    """
    A dataset class for recognizing interactions between pairs of cattle.
    It takes a list of data entries, adjusts skeleton coordinates to match cropped images,
    and applies custom and standard image transforms.
    """

    def __init__(self, entries, transform=None, custom_transform=None, use_pose=True):
        self.entries = entries
        self.transform = transform
        self.custom_transform = custom_transform
        self.use_pose = use_pose
        self.num_instance = 2

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # if the image is corrupted, treat this sample as unusable
        try:
            img = Image.open(entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as err:
            if DEBUG:
                print(
                    f"[WARN] Skipping corrupted image at index {idx} ({entry.get('image_path', 'N/A')}). Error: {err}"
                )
            # for a corrupted image, return dummy data or None to be handled in collate_fn
            # here we recursively try the next index (a simple fallback)
            return self.__getitem__((idx + 1) % len(self.entries))

        skeleton1_orig, skeleton2_orig = None, None
        if self.use_pose:
            try:
                skeleton1_orig = np.load(entry["pose_path_1"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_1: {entry['pose_path_1']}")
                skeleton1_orig = np.array([])  # on failure, assign an empty array

            try:
                skeleton2_orig = np.load(entry["pose_path_2"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_2: {entry['pose_path_2']}")
                skeleton2_orig = np.array([])  # on failure, assign an empty array

        # decide whether both poses are valid
        are_poses_valid = (
            skeleton1_orig is not None
            and skeleton1_orig.size > 0
            and skeleton2_orig is not None
            and skeleton2_orig.size > 0
        )

        supplemental_info = dict()
        supplemental_info["image_path"] = entry["image_path"]
        supplemental_info["pose_path_1"] = entry["pose_path_1"]
        supplemental_info["pose_path_2"] = entry["pose_path_2"]
        supplemental_info["merged_bbox_xyxy"] = entry["merged_bbox_xyxy"]
        supplemental_info["bbox1_xyxy"] = entry["bbox1_xyxy"]
        supplemental_info["bbox2_xyxy"] = entry["bbox2_xyxy"]
        supplemental_info["is_pose_valid"] = are_poses_valid
        supplemental_info["supplemental_label"] = entry.get("supplemental_label", -1)

        label = torch.tensor(entry["label"], dtype=torch.long)

        if are_poses_valid:
            try:
                bbox_str = entry["merged_bbox_xyxy"]
                merged_bbox = [int(n) for n in bbox_str.strip("[]").split()]
                crop_x_min, crop_y_min = merged_bbox[0], merged_bbox[1]
            except (ValueError, KeyError, TypeError, AttributeError):
                crop_x_min, crop_y_min = 0, 0

            skeleton1_adj = skeleton1_orig.copy()
            if skeleton1_adj.size > 0:
                skeleton1_adj[:, 0] -= crop_x_min
                skeleton1_adj[:, 1] -= crop_y_min

            skeleton2_adj = skeleton2_orig.copy()
            if skeleton2_adj.size > 0:
                skeleton2_adj[:, 0] -= crop_x_min
                skeleton2_adj[:, 1] -= crop_y_min

            if self.custom_transform:
                img = self.custom_transform(
                    image=img,
                    skeleton1=skeleton1_adj,
                    skeleton2=skeleton2_adj,
                    label=label,
                )

        if self.transform:
            img = self.transform(img)

        if are_poses_valid:
            pose1 = (
                torch.tensor(skeleton1_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose2 = (
                torch.tensor(skeleton2_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose = torch.cat((pose1, pose2), dim=-1)
        else:
            # if pose is invalid, create a dummy tensor so downstream code doesn't error
            num_coords = 3  # x, y, conf
            num_joints = 17
            pose = torch.zeros(
                (num_coords, 1, num_joints, self.num_instance), dtype=torch.float32
            )

        return img, pose, label, supplemental_info


class CattleCroppedInteractionDataset(Dataset):
    """
    Dataset class for recognizing interactions between pairs of cattle.
    __getitem__ crops the two cattle separately and returns two image tensors after resize and transform.
    """

    # --- change 1: add skeleton_aware_transform to __init__ ---
    def __init__(self, entries, transform=None, use_pose=True, skeleton_aware_transform=None, is_aware_skeleton=True):
        self.entries = entries
        self.transform = transform
        self.use_pose = use_pose
        self.skeleton_aware_transform = skeleton_aware_transform
        self.is_aware_skeleton = is_aware_skeleton
        self.num_instance = 2

    def __len__(self):
        return len(self.entries)

    def _parse_bbox(self, bbox_str: str) -> list:
        """
        Robustly parse bbox strings of various formats and return a list of ints.
        """
        cleaned_str = (
            bbox_str.replace("[", " ")
            .replace("]", " ")
            .replace(",", " ")
            .replace('"', "")
            .replace("'", "")
        )
        return [int(num) for num in cleaned_str.split() if num]

    def __getitem__(self, idx):
        entry = self.entries[idx]

        try:
            # load the source image
            cropped_img = Image.open(entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as err:
            if DEBUG:
                print(
                    f"[WARN] Skipping corrupted image: {entry['image_path']}. Error: {err}"
                )
            return self.__getitem__((idx + 1) % len(self.entries))

        skeleton1_orig, skeleton2_orig = None, None
        if self.use_pose:
            try:
                skeleton1_orig = np.load(entry["pose_path_1"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_1: {entry['pose_path_1']}")
                skeleton1_orig = np.array([])  # on failure, assign an empty array

            try:
                skeleton2_orig = np.load(entry["pose_path_2"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_2: {entry['pose_path_2']}")
                skeleton2_orig = np.array([])  # on failure, assign an empty array

        # decide whether both poses are valid
        are_poses_valid = (
            skeleton1_orig is not None
            and skeleton1_orig.size > 0
            and skeleton2_orig is not None
            and skeleton2_orig.size > 0
        )

        # --- change 1: moved skeleton-coordinate adjustment here ---
        skeleton1_adj, skeleton2_adj = None, None
        if are_poses_valid:
            try:
                bbox_str = entry["merged_bbox_xyxy"]
                merged_bbox = [int(n) for n in bbox_str.strip("[]").split()]
                crop_x_min, crop_y_min = merged_bbox[0], merged_bbox[1]
            except (ValueError, KeyError, TypeError, AttributeError):
                crop_x_min, crop_y_min = 0, 0

            # adjust skeleton coordinates to the bounding box
            skeleton1_adj = skeleton1_orig.copy()
            skeleton1_adj[:, 0] -= crop_x_min
            skeleton1_adj[:, 1] -= crop_y_min

            skeleton2_adj = skeleton2_orig.copy()
            skeleton2_adj[:, 0] -= crop_x_min
            skeleton2_adj[:, 1] -= crop_y_min

        # --- change 2: apply skeleton-based masking right after loading the image ---
        if self.skeleton_aware_transform is not None:
            if self.is_aware_skeleton:
                # run the transform only when valid skeleton info exists
                if are_poses_valid:
                    label = entry.get('label')
                    if label is not None:
                        # pass the adjusted coordinates to the transform
                        output_img = self.skeleton_aware_transform(
                            cropped_img, skeleton1_adj, skeleton2_adj, label
                        )
                        cropped_img = output_img
            else:
                # apply a transform that ignores skeleton info
                cropped_img = self.skeleton_aware_transform(cropped_img, None, None)

        # --- below: existing bbox parsing and cropping ---
        try:
            merged_bbox = self._parse_bbox(entry["merged_bbox_xyxy"])
            bbox1 = self._parse_bbox(entry["bbox1_xyxy"])
            bbox2 = self._parse_bbox(entry["bbox2_xyxy"])
        except (ValueError, AttributeError) as e:
            if DEBUG:
                print(
                    f"[WARN] Failed to parse bbox for {entry['image_path']}. Error: {e}"
                )
            return self.__getitem__((idx + 1) % len(self.entries))

        if len(merged_bbox) != 4 or len(bbox1) != 4 or len(bbox2) != 4:
            if DEBUG:
                print(
                    f"[WARN] Invalid bbox format for {entry['image_path']}. Skipping."
                )
            return self.__getitem__((idx + 1) % len(self.entries))

        offset_x, offset_y = merged_bbox[0], merged_bbox[1]
        rel_bbox1 = (
            bbox1[0] - offset_x,
            bbox1[1] - offset_y,
            bbox1[2] - offset_x,
            bbox1[3] - offset_y,
        )
        rel_bbox2 = (
            bbox2[0] - offset_x,
            bbox2[1] - offset_y,
            bbox2[2] - offset_x,
            bbox2[3] - offset_y,
        )

        img1 = cropped_img.crop(rel_bbox1)
        img2 = cropped_img.crop(rel_bbox2)

        if self.transform:
            cropped_img = self.transform(cropped_img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(entry["label"], dtype=torch.long)
        supplemental_info = {k: v for k, v in entry.items()}

        if are_poses_valid:
            # in this block skeleton1_adj and skeleton2_adj are always defined
            pose1 = (
                torch.tensor(skeleton1_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose2 = (
                torch.tensor(skeleton2_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose = torch.cat((pose1, pose2), dim=-1)
        else:
            num_coords = 3
            num_joints = 23  # Animal Kingdom keypoint count
            pose = torch.zeros(
                (num_coords, 1, num_joints, self.num_instance), dtype=torch.float32
            )
        supplemental_info['pose'] = pose

        return img1, img2, cropped_img, label, supplemental_info


class CattleActionDataset(Dataset):
    """Dataset class for cattle action recognition."""

    def __init__(
        self,
        entries,
        label_map,
        image_transform=None,
        custom_image_transform=None,
        pose_transform=None,
    ):
        """
        Args:
            entries (list): list of data entries.
            label_map (dict): mapping from label name to index.
            image_transform (callable, optional): standard image transform.
            custom_image_transform (callable, optional): custom transform that also uses skeleton info.
        """
        self.entries = entries
        self.label_map = label_map
        self.image_transform = image_transform
        self.custom_image_transform = custom_image_transform
        self.pose_transform = pose_transform
        self.edge_index = torch.tensor(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (5, 6),
                (6, 7),
                (3, 8),
                (8, 9),
                (9, 10),
                (4, 11),
                (11, 12),
                (12, 13),
                (4, 14),
                (14, 15),
                (15, 16),
            ]
        )
        self.num_instance = 1

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        for _ in range(len(self.entries)):
            entry = self.entries[idx]
            img_path = entry["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except (UnidentifiedImageError, OSError) as err:
                print(f"[WARN] Unrecognized image, skipping: {img_path}, error: {err}")
                idx = (idx + 1) % len(self.entries)
        else:
            raise RuntimeError("Failed to load any valid image in the dataset.")

        supplemental_info = dict()
        supplemental_info["image_path"] = img_path
        supplemental_info["pose_path"] = entry["pose_path"]

        # load skeleton info; use an empty array on failure
        try:
            pose_array = np.load(entry["pose_path"], allow_pickle=True)
        except (
            UnidentifiedImageError,
            OSError,
            FileNotFoundError,
            pickle.UnpicklingError,
            ValueError,
            EOFError,
        ) as err:
            pose_array = np.array([])

        is_pose_valid = (
            pose_array.size > 0
            and not np.isnan(pose_array).any()
            and not np.isinf(pose_array).any()
        )
        supplemental_info["is_pose_valid"] = is_pose_valid

        if not is_pose_valid:
            pose_array = np.array([])

        label = torch.tensor(entry["label"], dtype=torch.long)

        # apply custom augmentation only when skeleton info is valid
        if self.custom_image_transform and is_pose_valid:
            # temporarily convert to a tensor for this transform
            skeleton_for_aug = torch.tensor(pose_array, dtype=torch.float32)
            img = self.custom_image_transform(
                image=img, skeleton=skeleton_for_aug, label=label
            )

        # apply the skeleton transform
        if self.pose_transform and is_pose_valid:
            pose_array = self.pose_transform(pose_array)

        if is_pose_valid:
            # calc image size from image_path
            img_size = Image.open(entry["image_path"]).size
            # normalize pose coodinate with image size 0-1-2 is x-y-conf
            pose_array[:, 0] = pose_array[:, 0] / img_size[0]  # x
            pose_array[:, 1] = pose_array[:, 1] / img_size[1]  # y

            supplemental_info["image_size"] = img_size
        else:
            supplemental_info["image_size"] = (0, 0)

        # apply the standard image transform
        if self.image_transform:
            img = self.image_transform(img)

        # build the pose tensor
        if is_pose_valid:
            skeleton = torch.tensor(pose_array, dtype=torch.float32)
            # based on the original code, build a (C, 1, V, M) tensor
            # skeleton: (V, C) -> permute -> (C, V) -> unsqueeze -> (C, 1, V, 1) -> repeat -> (C, 1, V, M)
            pose = skeleton.permute(1, 0).unsqueeze(1).unsqueeze(-1)
            pose = pose.repeat(1, 1, 1, self.num_instance)
        else:
            # if pose is invalid, create a dummy tensor so downstream code doesn't error
            num_coords = 3  # x, y, confidence
            num_joints = 17  # number of joints in the dataset
            # zero tensor matching the (C, 1, V, M) shape
            pose = torch.zeros(
                (num_coords, 1, num_joints, self.num_instance), dtype=torch.float32
            )

        return img, pose, label, supplemental_info


import os
import re
import pandas as pd

# Do not swallow errors; prioritise specific patterns to keep quality and correctness.

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Generate Table 1: Categories and sample counts for behaviors (Fixed)
    # -------------------------------------------------------------------------

    action_root_dir = "/mnt/nfs/processed/action_data"
    interaction_root_dir = "/mnt/nfs/processed/interaction"
    
    delete_base_dirs = []

    print(f"Loading Action data from: {action_root_dir}")
    print(f"Loading Interaction data from: {interaction_root_dir}")

    # --- shared aggregation logic (enhanced pattern matching) ---
    def calculate_counts_with_grouping(entries):
        if not entries:
            return pd.Series(dtype=int), pd.Series(dtype=int)

        df = pd.DataFrame(entries)

        # 1. derive 'source_video' and 'frame_number' from the image path
        if 'source_video' not in df.columns or 'frame_number' not in df.columns:
            def extract_info(path):
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                
                # --- Strategy A: Explicit "frame" keyword (preferred) ---
                # e.g. "2019..._frame_00000090_pair_01" -> Video="2019...", Frame=90
                # capture the digits after "frame_"; everything before is the video name
                match_frame = re.search(r'(.*)_frame_(\d+)', name)
                if match_frame:
                    video_name = match_frame.group(1)
                    frame_num = int(match_frame.group(2))
                    return pd.Series([video_name, frame_num, False])

                # --- Strategy B: Generic suffix number (fallback) ---
                # e.g. "CowVideo_123" -> Video="CowVideo", Frame=123
                # legacy logic: the trailing number is the frame number
                match_suffix = re.search(r'^(.*)[_-](\d+)$', name)
                if match_suffix:
                    video_name = match_suffix.group(1)
                    frame_num = int(match_suffix.group(2))
                    return pd.Series([video_name, frame_num, False])

                # --- Strategy C: Parse Failed ---
                return pd.Series([name, 0, True]) 

            # extract info from the path
            extracted = df['image_path'].apply(extract_info)
            extracted.columns = ['source_video', 'frame_number', 'parse_error']
            
            # error check
            error_count = extracted['parse_error'].sum()
            if error_count > 0:
                print(f"  [Warning] Failed to parse frame number for {error_count} images. Treated as single-frame actions.")

            # merge, preferring existing columns
            if 'source_video' not in df.columns:
                df['source_video'] = extracted['source_video']
            if 'frame_number' not in df.columns:
                df['frame_number'] = extracted['frame_number']

        # 2. type conversion
        df['label'] = df['label'].astype(int)
        df['frame_number'] = df['frame_number'].astype(int)

        # 3. sort (strictly)
        df_sorted = df.sort_values(['source_video', 'label', 'frame_number']).reset_index(drop=True)

        # 4. group-boundary conditions
        #    - the video changed
        #    - the label changed
        #    - the frame gap exceeded 100 (~3.3 s at 30 fps)
        is_new_group_start = (df_sorted['source_video'] != df_sorted['source_video'].shift(1)) | \
                             (df_sorted['label'] != df_sorted['label'].shift(1)) | \
                             (df_sorted['frame_number'].diff() > 100)

        # 5. assign group IDs
        df_sorted['group'] = is_new_group_start.cumsum()

        # 6. aggregate
        action_counts_series = df_sorted.groupby('label')['group'].nunique()
        image_counts_series = df_sorted['label'].value_counts()

        return image_counts_series, action_counts_series


    # --- 1. Individual Behaviors Counting ---
    action_map = {
        "grazing": 0,
        "standing": 1,
        "lying": 2,
        "riding": 3
    }
    
    # NOTE: the function is assumed to be defined elsewhere
    action_entries = get_all_action_annotations_entries(
        action_root_dir, action_map, delete_base_dirs, drop_unknown_label=True
    )

    print("Processing Individual Behaviors...")
    indiv_img_counts, indiv_act_counts = calculate_counts_with_grouping(action_entries)


    # --- 2. Interactions Counting ---
    interaction_map = {
        "no_interaction": 0,
        "interest": 1,
        "conflict": 2,
        "mount": 3,
    }
    
    # NOTE: the function is assumed to be defined elsewhere
    interaction_entries = get_all_interaction_annotations_entries(
        interaction_root_dir, 
        interaction_map, 
        delete_base_dirs, 
        use_more_than_three_cattles=False
    )

    print("Processing Interactions...")
    inter_img_counts, inter_act_counts = calculate_counts_with_grouping(interaction_entries)


    # --- 3. Print Formatting (Table 1) ---
    print("\n")
    print(f"Table 1. Categories and sample counts for behaviors.")
    print("=" * 72)
    # Header
    print(f"{'Category':<12} {'Images':>8} {'Actions':>8} | {'Category':<15} {'Images':>8} {'Actions':>8}")
    print("-" * 72)
    print(f"{'Individual Behaviors':<30} | {'Interactions':<33}")
    
    left_order = ["grazing", "standing", "lying", "riding"]
    right_order = ["no_interaction", "interest", "conflict", "mount"]
    
    subtotal_left_img = 0
    subtotal_left_act = 0
    subtotal_right_img = 0
    subtotal_right_act = 0

    for i in range(4):
        # Left Side (Individual)
        l_name = left_order[i]
        l_idx = action_map[l_name]
        
        l_img_count = indiv_img_counts.get(l_idx, 0)
        l_act_count = indiv_act_counts.get(l_idx, 0)
        
        subtotal_left_img += l_img_count
        subtotal_left_act += l_act_count

        # Right Side (Interaction)
        r_name = right_order[i]
        r_idx = interaction_map[r_name]
        
        r_img_count = inter_img_counts.get(r_idx, 0)
        r_act_count = inter_act_counts.get(r_idx, 0)

        subtotal_right_img += r_img_count
        subtotal_right_act += r_act_count

        print(
            f"  {l_name:<10} {l_img_count:>8} {l_act_count:>8} | "
            f"  {r_name:<13} {r_img_count:>8} {r_act_count:>8}"
        )

    print("-" * 72)
    print(
        f"  {'Subtotal':<10} {subtotal_left_img:>8} {subtotal_left_act:>8} | "
        f"  {'Subtotal':<13} {subtotal_right_img:>8} {subtotal_right_act:>8}"
    )
    print("=" * 72)