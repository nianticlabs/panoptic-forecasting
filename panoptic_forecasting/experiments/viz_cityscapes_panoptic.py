import argparse
import cityscapesscripts.helpers.labels as cslabels
import json
import numpy as np
import os
import cv2
from tqdm import tqdm


def get_image_id(annotation):
    assert isinstance(annotation, dict)
    image_id = annotation.get('image_id')
    assert isinstance(image_id, str)
    return image_id


def read_label(annotation, label_dir):
    # Get the file name.
    assert isinstance(annotation, dict)
    file_name = annotation.get('file_name')
    assert isinstance(file_name, str)

    # Read png as numpy.
    file_path = os.path.join(label_dir, file_name)
    if not os.path.isfile(file_path):
        print("Missing file: {}".format(file_path))
        return None
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3

    # Decode the pixel into id.
    img = img.astype(dtype=np.int32)
    return img[:, :, 2] + img[:, :, 1] * 256 + img[:, :, 0] * 256 * 256

def read_gt(img_id, gt_dir):
    path = os.path.join(gt_dir, '{}_gtFine_panoptic.png'.format(img_id))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(dtype=np.int32)
    return img[:, :, 2] + img[:, :, 1] * 256 + img[:, :, 0] * 256 * 256

def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = (img == 1) | (img == 0) | (img == 2)
    return mask

def get_mask_from_dir(mask_dir, image_id):
    info = image_id.split('_')
    city, seq = info[0], info[1]
    frame = int(info[2])
    frame_options = range(frame-19, frame+11)
    mask_path_options = [os.path.join(mask_dir, city,
                                      '%s_%s_%06d_gtFine_labelIds.png'%(city, seq, fr))
                         for fr in frame_options]
    for option in mask_path_options:
        if os.path.exists(option):
            return read_mask(option)
    raise ValueError('Could not find gt mask for id: ',image_id)

def read_rgb(annotation, rgb_dir):
    # The image_id can be decoded as <city>_<seq>_<frame>.
    image_id = get_image_id(annotation)
    elements = image_id.split('_')
    assert len(elements) == 3, "Cannot decode image id: {}".format(image_id)
    city, seq, frame = elements

    # The rgb-dir is assumed to be structured as <city>/city_seq_frame_leftImg8bit.png
    img_path = os.path.join(rgb_dir, city, "{}_{}_{}_leftImg8bit.png".format(city, seq, frame))
    if not os.path.isfile(img_path):
        print("Missing file: {}".format(img_path))
        return None
    return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def get_category_color_bgr(category_id):
    # NOTE: Cityscapes's color code is (R, G, B), but our image is BGR.
    return list(reversed(cslabels.id2label[category_id].color))


def get_seg_id_to_category_id(annotation):
    assert isinstance(annotation, dict)
    segments_info = annotation.get('segments_info')
    assert isinstance(segments_info, list)
    return {seg_info['id'] : seg_info['category_id']  for seg_info in segments_info }


def overlay_color_on_image(img, color):
    assert isinstance(img, np.ndarray) and 2 <= len(img.shape) <= 3
    assert isinstance(color, np.ndarray) and len(color.shape) == 3 and color.shape[2] == 3
    assert img.shape[0] == color.shape[0] and img.shape[1] == color.shape[1]

    # Get a grayscale image as the background.
    if len(img.shape) == 3 and img.shape[2] == 3:
        background = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert len(background.shape) == 2 or background.shape[2] == 1
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    assert len(background.shape) == 3 and background.shape[2] == 3

    # Blend the color image with the background image.
    return cv2.addWeighted(background, 0.5, color, 0.5, 0.0, dtype=cv2.CV_8UC3)


def get_color_label_image(label_img, annotation, gt_img, gt_mask, vary_by_instance=False):
    assert isinstance(label_img, np.ndarray) and len(label_img.shape) == 2
    seg_id_to_category_id = get_seg_id_to_category_id(annotation)

    # Start with a black image.
    img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)

    # Color each segment id.
    category_id_segment_counts = dict()
    for seg_id, category_id in seg_id_to_category_id.items():
        # NOTE: Cityscapes's color code is (R, G, B), but our image is BGR.
        color = get_category_color_bgr(category_id)

        # Add instance variation.
        category_id_segment_counts.setdefault(category_id, 0)
        if vary_by_instance and cslabels.id2label[category_id].hasInstances:
            '''
            # Set a step size for the instance color to vary.
            COLOR_STEP = 5
            # Vary the channel dominating the color.
            color_change = COLOR_STEP * category_id_segment_counts[category_id]
            ch = color.index(max(color))
            color[ch] = min(color[ch] + color_change, 255)
            '''
            color_offset = np.random.randint(-35, 36, size=(3,))
            color = (color + color_offset).clip(0, 255)
        category_id_segment_counts[category_id] += 1

        mask = label_img == seg_id
        img[mask] = color
    if gt_mask is not None:
        img[gt_mask] = [0,0,0]
    elif gt_img is not None:
        mask = gt_img[800:] == 0
        img[800:][mask] = [0,0,0]
    return img


def get_instance_label_contours_mask(label_img, annotation):
    assert isinstance(label_img, np.ndarray) and len(label_img.shape) == 2
    seg_id_to_category_id = get_seg_id_to_category_id(annotation)

    # Start with a black image.
    contours_mask = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)

    for seg_id, category_id in seg_id_to_category_id.items():
        if not cslabels.id2label[category_id].hasInstances:
            continue
        contours, _ = \
            cv2.findContours((label_img == seg_id).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(contours_mask, contours, -1, (255), 3)
    return contours_mask


def visualize_one_frame(annotation, label_img, rgb_img, gt_img, mask):
    color = get_color_label_image(label_img, annotation, gt_img, mask, vary_by_instance=False)
    img_overlay =  overlay_color_on_image(rgb_img, color)

    # Get the contours, and draw with corresponding pixel colors.
    contours_mask = get_instance_label_contours_mask(label_img, annotation)
    img_overlay[contours_mask != 0] = 255 - color[contours_mask != 0]
    if gt_img is not None:
        mask = gt_img[800:] == 0
        img_overlay[800:][mask] = [0,0,0]
    elif mask is not None:
        img_overlay[mask] = [0,0,0]

    return img_overlay


def visualize_frames(annotations, label_dir, rgb_dir, output_dir, gt_dir, mask_path, mask_dir):
    assert isinstance(annotations, list)
    mask = None
    if mask_path is not None:
        mask = read_mask(mask_path)
    for annotation in tqdm(annotations):
        image_id = get_image_id(annotation)
        if mask_dir is not None:
            mask = get_mask_from_dir(mask_dir, image_id)
        #print("Processing {} ...".format(image_id))

        # Read data.
        rgb_img = read_rgb(annotation, rgb_dir)
        label_img = read_label(annotation, label_dir)
        if rgb_img is None and label_img is None:
            continue

        if gt_dir is not None:
            gt_img = read_gt(image_id, gt_dir)
        else:
            gt_img = None

        # Visualize and output.
        viz_img = visualize_one_frame(annotation, label_img, rgb_img, gt_img, mask)
        output_path = os.path.join(output_dir, "{}_seg.png".format(image_id))
        cv2.imwrite(output_path, viz_img)


def read_annotation_json(annotation_json):
    with open(annotation_json) as infile:
        json_dict = json.load(infile)
    annotations = json_dict.get("annotations")
    assert isinstance(annotations, list), "Failed to parse {}.".format(annotation_json)
    print("Read annotations for {} images.".format(len(annotations)))
    return annotations


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_json', help="The JSON file containing annotations for the frames.")
    parser.add_argument('label_dir', help="The folder containing the label (.png) files for annotated frames.")
    parser.add_argument('rgb_dir', help="The folder containing rgb images (folder/city/city_seq_frame_leftImg8bit.png)")
    parser.add_argument('output_dir', help="Output directory for storing visualizations.")
    parser.add_argument('--gt_dir')
    parser.add_argument('--mask_path')
    parser.add_argument('--mask_dir')
    args = parser.parse_args()
    print("MASK DIR: ",args.mask_dir)

    # Check inputs files and directories.
    assert os.path.isfile(args.annotation_json), "File {} is missing.".format(args.annotation_json)
    assert os.path.isdir(args.label_dir), "Not a valid directory: {}".format(args.label_dir)
    assert os.path.isdir(args.rgb_dir), "Not a valid directory: {}".format(args.rgb_dir)

    # Check output directories.
    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isdir(args.output_dir), "Can't create output directory: {}".format(args.output_dir)

    return args


def main():
    args = parse_inputs()
    annotations = read_annotation_json(args.annotation_json)
    visualize_frames(annotations, args.label_dir, args.rgb_dir, args.output_dir, args.gt_dir, args.mask_path, args.mask_dir)


if __name__ == "__main__":
    main()

