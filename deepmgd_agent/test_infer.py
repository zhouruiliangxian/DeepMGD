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

config_file = r"D:\deepLearning\YOLOv10-main\repvgg\REPVGG.py"
checkpoint_file = (
    r"D:\deepLearning\YOLOv10-main\repvgg\best_coco_bbox_mAP_50_epoch_280.pth"
)
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
        scores = pred_instances.scores.cpu().numpy()  # 转为 NumPy 数组
        bbox_dict = {}

        # 遍历 labels 和 bboxes
        for label, bbox, score in zip(labels, bboxes, scores):
            # 根据 label 的值修改标签
            if label == 1:
                label_name = "void"
            elif label == 0:
                label_name = "impurity"

            score = f"confidence score: {score}"
            # 如果标签已经存在，追加到列表中；如果不存在，创建新的列表
            if label_name in bbox_dict:
                bbox_dict[label_name].append([bbox.tolist(), score])
            else:
                bbox_dict[label_name] = [[bbox.tolist(), score]]

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
