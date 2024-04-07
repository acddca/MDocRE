# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2023/1/17 11:58
# @File: preprocess_offline.py
# @Email: wangjl.nju.2020@gmail.com.
import os
import json
import ntpath
import traceback

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms

def main():
    dataset_dir = "/home/nlper_data/wangjl/dataset/MDocRED/"
    visual_data_path = "/home/nlper_data/wangjl/dataset/MDocRED/frame"

    # with open(os.path.join(dataset_dir, "train_lens.json")) as fp:
    #     data_lens = json.load(fp)

    datasets = [
        "train_annotated0.json",
        "train_annotated1.json",
        "train_annotated2.json",
        "train_annotated3.json",
        "train_annotated4.json",
        "train_annotated5.json",
        "train_annotated6.json",
        "train_annotated7.json",
        "train_annotated8.json",
        "train_annotated9.json",
        "dev.json"
    ]

    datasets = [os.path.join(dataset_dir, dataset) for dataset in datasets]
    image_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for dataset in datasets:
        with open(dataset) as fp:
            samples = json.load(fp)

        image2tensor = {}
        for sample in tqdm(samples, desc=f"### Processing dataset {os.path.basename(dataset)}"):
            video_label_dir = sample["visual_label_path"]
            if not ntpath.isabs(video_label_dir):
                video_label_file = os.path.basename(video_label_dir)
            else:
                video_label_file = ntpath.basename(video_label_dir)

            video_image_dir = os.path.join(visual_data_path, video_label_file)
            images = os.listdir(video_image_dir)
            images = [
                os.path.join(video_image_dir, image_file)
                for image_file in images if image_file.endswith(".jpg")
            ]

            for image_file in images:
                try:
                    image = Image.open(image_file)
                    v_feat = image_transforms(image)
                    image2tensor[image_file] = v_feat.tolist()
                except:
                    print(f"### Warnings: {image_file}")
                    traceback.print_exc()

        save_path = dataset + ".img.data"
        with open(save_path, "w") as fp:
            print(f"### Write data to {save_path}")
            json.dump(image2tensor, fp)


if __name__ == "__main__":
    main()