import argparse
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools.coco import COCO
import os
import os.path as osp
import numpy as np
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def show_coco_json(args):
    if args.data_root is not None:
        coco = COCO(osp.join(args.data_root, args.ann_file))
    else:
        coco = COCO(args.ann_file)
    print(f'Total number of images: {len(coco.getImgIds())}')
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print(f'Total number of Categories: {len(category_names)}')
    print('Categories: \n{}\n'.format(' '.join(category_names)))

    if args.category_names is None:
        category_ids = []
    else:
        assert set(category_names) > set(args.category_names)
        category_ids = coco.getCatIds(args.category_names)

    image_ids = coco.getImgIds(catIds=category_ids)

    if args.shuffle:
        np.random.shuffle(image_ids)

    for i in range(len(image_ids)):
        image_data = coco.loadImgs(image_ids[i])[0]
        if args.data_root is not None:
            image_path = osp.join(args.data_root, args.img_dir, image_data['file_name'])
        else:
            image_path = osp.join(args.img_dir, image_data['file_name'])
        save_path = osp.join(args.out_dir, image_data['file_name'])

        # Create directory if it doesn't exist
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=0)
        annotations = coco.loadAnns(annotation_ids)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.disp_all:
            coco.showAnns(annotations)
        else:
            show_bbox_only(coco, annotations, image, save_path=save_path)

def show_bbox_only(coco, anns, image, show_label_bbox=True, is_filling=False, save_path=None):
    """Show bounding box of annotations Only."""
    if len(anns) == 0:
        return

    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (0, 0, 255) if cat == 0 else (255, 0, 0)

    for ann in anns:
        color = image2color[ann['category_id']]
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        pt1 = (int(bbox_x), int(bbox_y))
        pt2 = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))
        cv2.rectangle(image, pt1, pt2, color, 2)

        # if show_label_bbox:
        #     category_name = coco.loadCats(ann['category_id'])[0]['name']
        #     cv2.putText(image, category_name, (int(bbox_x), int(bbox_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if save_path:
        # Convert the image back to BGR before saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
#
#
# def show_coco_json(args):
#     if args.data_root is not None:
#         coco = COCO(osp.join(args.data_root, args.ann_file))
#     else:
#         coco = COCO(args.ann_file)
#     print(f'Total number of images：{len(coco.getImgIds())}')
#     categories = coco.loadCats(coco.getCatIds())
#     category_names = [category['name'] for category in categories]
#     print(f'Total number of Categories : {len(category_names)}')
#     print('Categories: \n{}\n'.format(' '.join(category_names)))
#
#     if args.category_names is None:
#         category_ids = []
#     else:
#         assert set(category_names) > set(args.category_names)
#         category_ids = coco.getCatIds(args.category_names)
#
#     image_ids = coco.getImgIds(catIds=category_ids)
#
#     if args.shuffle:
#         np.random.shuffle(image_ids)
#
#     for i in range(len(image_ids)):
#         image_data = coco.loadImgs(image_ids[i])[0]
#         if args.data_root is not None:
#             image_path = osp.join(args.data_root, args.img_dir,
#                                   image_data['file_name'])
#         else:
#             image_path = osp.join(args.img_dir, image_data['file_name'])
#         save_path = osp.join(args.out_dir, image_data['file_name'])
#         import os
#         os.makedirs(osp.dirname(save_path), exist_ok=True)
#         annotation_ids = coco.getAnnIds(
#             imgIds=image_data['id'], catIds=category_ids, iscrowd=0)
#         annotations = coco.loadAnns(annotation_ids)
#
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         plt.figure()
#         plt.imshow(image)
#
#         if args.disp_all:
#             coco.showAnns(annotations)
#         else:
#             show_bbox_only(coco, annotations,save_path=save_path)
#
#         if args.wait_time == 0:
#             plt.show()
#         else:
#             plt.show(block=False)
#             plt.pause(args.wait_time)
#
#         plt.close()
#
#
# def show_bbox_only(coco, anns, show_label_bbox=True, is_filling=False,save_path=None):
#     """Show bounding box of annotations Only."""
#     if len(anns) == 0:
#         return
#
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#
#     image2color = dict()
#     for cat in coco.getCatIds():
#         image2color[cat] = [0,0,1] if cat == 0 else [1 ,0 ,0]
#
#     polygons = []
#     colors = []
#     # print(image2color)
#     for ann in anns:
#         color = image2color[ann['category_id']]
#         bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
#         poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
#                 [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
#         polygons.append(Polygon(np.array(poly).reshape((4, 2))))
#         colors.append(color)
#
#         if show_label_bbox:
#             label_bbox = dict(facecolor=color)
#         else:
#             label_bbox = None
#
#         # ax.text(
#         #     bbox_x,
#         #     bbox_y,
#         #     # '%s' % (coco.loadCats(ann['category_id'])[0]['name']),
#         #     color='white',
#         #     bbox=label_bbox)
#     # print(colors)
#     if is_filling:
#         p = PatchCollection(
#             polygons, facecolor=colors, linewidths=0, alpha=0.4)
#         ax.add_collection(p)
#     p = PatchCollection(
#         polygons, facecolor='none', edgecolors=colors, linewidths=1)
#     ax.add_collection(p)
#     if save_path:
#         plt.axis('off')
#         plt.savefig(save_path, bbox_inches='tight')

def parse_args():
    parser = argparse.ArgumentParser(description='Show coco json file')
    parser.add_argument('--data-root', default='./newcoco15coco/', help='dataset root')
    parser.add_argument(
        '--img-dir', default='images/val', help='image folder path')
    parser.add_argument(
        '--out-dir', default='output/VAL2', help='image folder path')
    parser.add_argument(
        '--ann-file',
        default='annotations/instances_val.json',
        help='ann file path')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--disp-all',
        action='store_true',
        help='Whether to display all types of data, '
        'such as bbox and mask.'
        ' Default is to display only bbox')
    parser.add_argument(
        '--category-names',
        type=str,
        default=None,
        nargs='+',
        help='Display category-specific data, e.g., "bicycle", "person"')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Whether to display in disorder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    show_coco_json(args)


if __name__ == '__main__':
    main()
