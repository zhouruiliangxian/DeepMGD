"""
python demo\image_demo.py demo\1.jpg "PTH 2\new15_mobilenet_RepVGGBlock_small_1gpu_b4_repvit_2\REPVGG-TEN_Mobilenet_300e_coco_SAMLL.py" "PTH 2\new15_mobilenet_Rep
VGGBlock_small_1gpu_b4_repvit_2\best_coco_bbox_mAP_50_epoch_293.pth"

model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

<DetDataSample(

    META INFORMATION
    img_shape: (960, 960, 3)
    img_path: 'demo/1.jpg'
    batch_input_shape: (960, 960)
    img_id: 0
    pad_param: array([0., 0., 1., 2.], dtype=float32)
    ori_shape: (1032, 1029)
    pad_shape: (960, 960)
    scale_factor: (0.9300291545189504, 0.9302325581395349)

    DATA FIELDS
    ignored_instances: <InstanceData(

            META INFORMATION

            DATA FIELDS
            labels: tensor([], device='cuda:0', dtype=torch.int64)
            bboxes: tensor([], device='cuda:0', size=(0, 4))
        ) at 0x13972e49190>
    gt_instances: <InstanceData(

            META INFORMATION

            DATA FIELDS
            labels: tensor([], device='cuda:0', dtype=torch.int64)
            bboxes: tensor([], device='cuda:0', size=(0, 4))
        ) at 0x13972e49160>
    pred_instances: <InstanceData(

            META INFORMATION

            DATA FIELDS
            labels: tensor([1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
            scores: tensor([0.9687, 0.0036, 0.0030, 0.0027, 0.0023, 0.0022, 0.0020, 0.0020, 0.0020,
                        0.0018, 0.0018, 0.0017, 0.0017, 0.0017, 0.0014, 0.0013, 0.0013, 0.0010],
                       device='cuda:0')
            bboxes: tensor([[ 298.8452,  525.0196,  580.6752,  998.5887],
                        [ 296.7545,  556.0909,  575.4324, 1000.8933],
                        [ 895.9421,    6.2192,  920.1857,   29.4943],
                        [  22.8016,  265.4389,   40.6891,  280.8874],
                        [ 883.1254,    4.9271,  921.4995,   31.1227],
                        [ 143.3548,  514.8217,  602.9014, 1001.1914],
                        [ 663.5807,  917.3110, 1006.8932, 1032.0000],
                        [ 219.2155,    0.0000,  823.0557,  500.9189],
                        [ 892.9113,    3.5078,  922.8265,   27.7250],
                        [ 291.1401,  464.3803,  645.9191, 1003.5341],
                        [ 266.3294,  543.6967,  829.8965, 1032.0000],
                        [ 493.5009,    0.0000, 1029.0000,  532.0739],
                        [   0.0000,    0.0000,   52.8916,   68.2023],
                        [ 385.1058,    0.0000,  995.4342,  501.4936],
                        [ 715.1480,  905.1956, 1029.0000, 1032.0000],
                        [ 597.6408,  917.0856,  948.8046, 1032.0000],
                        [   5.7793,  416.3841,  587.4075,  954.3188],
                        [ 322.5927,    0.0000,  909.4130,  559.4173]], device='cuda:0')
        ) at 0x13972e42f10>
) at 0x13972e49250>
"""

from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)
import numpy as np
import os
from werkzeug.utils import secure_filename

# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes

from mmdet.apis import init_detector, inference_detector

config_file = r"PTH 2\new15_mobilenet_RepVGGBlock_small_1gpu_b4_repvit_2\REPVGG-TEN_Mobilenet_300e_coco_SAMLL.py"
checkpoint_file = r"PTH 2\new15_mobilenet_RepVGGBlock_small_1gpu_b4_repvit_2\best_coco_bbox_mAP_50_epoch_293.pth"
model = init_detector(config_file, checkpoint_file, device="cuda")  # or device='cuda:0'


@app.route("/process_image", methods=["POST"])
def process_image():
    # 检查是否有文件上传
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    out_file = request.form.get("out_file")
    if not out_file:
        return jsonify({"error": "没有上传输出文件的字符串"}), 400
    print(out_file)
    file = request.files["image"]
    try:
        demo_folder = "demo"
        if not os.path.exists(demo_folder):
            os.makedirs(demo_folder)

        # 保存上传的图片到 demo 文件夹
        filename = secure_filename(file.filename)  # 安全地获取文件名
        file_path = os.path.join(demo_folder, filename)
        file.save(file_path)

        result = inference_detector(model, file_path)
        print(result)

        img = mmcv.imread(file_path)
        img = mmcv.imconvert(img, "bgr", "rgb")

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[result.pred_instances.scores > 0.2]
        labels = pred_instances.labels.cpu().numpy()  # 转为 NumPy 数组
        bboxes = pred_instances.bboxes.cpu().numpy()  # 转为 NumPy 数组
        bbox_dict = {}

        # 遍历 labels 和 bboxes
        for label, bbox in zip(labels, bboxes):
            # 根据 label 的值修改标签
            if label == 1:
                label_name = "void"
            elif label == 0:
                label_name = "impurity"

            # 如果标签已经存在，追加到列表中；如果不存在，创建新的列表
            if label_name in bbox_dict:
                bbox_dict[label_name].append(bbox.tolist())
            else:
                bbox_dict[label_name] = [bbox.tolist()]

        print(labels, bboxes, bbox_dict)

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = {
            "classes": ("impurity", "void"),
            "palette": [
                (0, 0, 255),
                (255, 0, 0),
            ],
        }

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=0.2,
        )

        # 模拟处理逻辑，这里返回图片的尺寸
        processed_result = {
            "message": "Image processed successfully",
            #     "size": image.size,  # 返回图片尺寸
            "defects": bbox_dict,
        }
        return jsonify(processed_result)  # 返回JSON响应
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
