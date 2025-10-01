import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from src.detection.custom_utils import collate_fn, get_train_transform, get_valid_transform

# The dataset class.
class CustomDataset(Dataset):
    SUPPORTED_ANNOTATION_FORMATS = {
        'pascal_voc': 'pascal_voc',
        'voc': 'pascal_voc',
        'xml': 'pascal_voc',
        'yolo': 'yolo_txt',
        'yolo_txt': 'yolo_txt',
        'txt': 'yolo_txt',
    }

    def __init__(
            self,
            img_path,
            annot_path,
            width,
            height,
            classes,
            transforms=None,
            annotation_format='pascal_voc'
    ):
        self.transforms = transforms
        self.img_path = img_path
        self.annot_path = annot_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        annotation_format = annotation_format.lower()
        if annotation_format not in self.SUPPORTED_ANNOTATION_FORMATS:
            raise ValueError(
                f"Unsupported annotation format: {annotation_format}. "
                f"Supported formats are: {list(self.SUPPORTED_ANNOTATION_FORMATS.keys())}"
            )
        self.annotation_format = self.SUPPORTED_ANNOTATION_FORMATS[annotation_format]
        self.background_offset = 1 if len(self.classes) > 0 and self.classes[0] == '__background__' else 0
        
        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.img_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # Capture the corresponding XML file for getting the annotations.
        if self.annotation_format == 'pascal_voc':
            annot_filename = os.path.splitext(image_name)[0] + '.xml'
        else:
            annot_filename = os.path.splitext(image_name)[0] + '.txt'
        annot_file_path = os.path.join(self.annot_path, annot_filename)

        boxes = []
        labels = []
        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]

        if os.path.exists(annot_file_path):
            if self.annotation_format == 'pascal_voc':
                boxes, labels = self._parse_voc_annotation(
                    annot_file_path,
                    image_width,
                    image_height
                )
            else:
                boxes, labels = self._parse_yolo_annotation(
                    annot_file_path,
                    image_width,
                    image_height
                )

        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
            if target['boxes'].shape[0] > 0:
                target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (
                    target['boxes'][:, 2] - target['boxes'][:, 0]
                )
            else:
                target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((target['boxes'].shape[0],), dtype=torch.int64)
        else:
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def _rescale_and_clip_box(self, xmin, ymin, xmax, ymax, image_width, image_height):
        xmin_final = (xmin / image_width) * self.width
        xmax_final = (xmax / image_width) * self.width
        ymin_final = (ymin / image_height) * self.height
        ymax_final = (ymax / image_height) * self.height

        xmin_final = max(min(xmin_final, self.width), 0)
        xmax_final = max(min(xmax_final, self.width), 0)
        ymin_final = max(min(ymin_final, self.height), 0)
        ymax_final = max(min(ymax_final, self.height), 0)

        if xmax_final <= xmin_final:
            xmax_final = min(xmin_final + 1, self.width)
        if ymax_final <= ymin_final:
            ymax_final = min(ymin_final + 1, self.height)

        return [xmin_final, ymin_final, xmax_final, ymax_final]

    def _parse_voc_annotation(self, annot_file_path, image_width, image_height):
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        for member in root.findall('object'):
            label_name = member.find('name').text
            if label_name not in self.classes:
                continue
            labels.append(self.classes.index(label_name))

            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            boxes.append(
                self._rescale_and_clip_box(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    image_width,
                    image_height
                )
            )
        return boxes, labels

    def _parse_yolo_annotation(self, annot_file_path, image_width, image_height):
        boxes = []
        labels = []
        with open(annot_file_path, 'r') as annot_file:
            for line in annot_file.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0])) + self.background_offset
                if class_id < 0 or class_id >= len(self.classes):
                    continue

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                if max(abs(x_center), abs(y_center), abs(width), abs(height)) <= 1.0:
                    x_center *= image_width
                    y_center *= image_height
                    width *= image_width
                    height *= image_height

                xmin = x_center - (width / 2)
                ymin = y_center - (height / 2)
                xmax = x_center + (width / 2)
                ymax = y_center + (height / 2)

                boxes.append(
                    self._rescale_and_clip_box(
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        image_width,
                        image_height
                    )
                )
                labels.append(class_id)

        return boxes, labels

    def __len__(self):
        return len(self.all_images)

# Prepare the final datasets and data loaders.
def create_train_dataset(img_dir, annot_dir, classes, resize=(640, 640), annotation_format='pascal_voc'):
    train_dataset = CustomDataset(
        img_dir,
        annot_dir,
        resize[0],
        resize[1],
        classes,
        get_train_transform(),
        annotation_format=annotation_format
    )
    return train_dataset
def create_valid_dataset(img_dir, annot_dir, classes, resize=(640, 640), annotation_format='pascal_voc'):
    valid_dataset = CustomDataset(
        img_dir,
        annot_dir,
        resize[0],
        resize[1],
        classes,
        get_valid_transform(),
        annotation_format=annotation_format
    )
    return valid_dataset
def create_train_loader(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True
    )
    return train_loader
def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True
    )
    return valid_loader


# execute `datasets.py` using Python command from 
# Terminal to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    import yaml
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='path to the configuration yaml file in detection_configs folder',
        default='detection_configs/voc.yaml'
    )
    args = parser.parse_args()
    print(args)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    TRAIN_IMG =  config['TRAIN_IMG']
    TRAIN_ANNOT = config['TRAIN_ANNOT']
    VALID_IMG = config['VALID_IMG']
    VALID_ANNOT = config['VALID_ANNOT']
    CLASSES = config['CLASSES']
    ANNOT_FORMAT = config.get('ANNOT_FORMAT', 'pascal_voc')

    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_IMG,
        TRAIN_ANNOT,
        640,
        640,
        CLASSES,
        get_train_transform(),
        annotation_format=ANNOT_FORMAT
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        image = np.array(image).transpose((1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 0, 255), 
                2
            )
            cv2.putText(
                image, 
                label, 
                (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 50
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)