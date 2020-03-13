修改了dataset/underwater的源码
只能用cascade_rcnn_r50_fpn_1x
可能还有bug

配置项目
* git clone https://github.com/ddghost/underwater.git
* cd underwater
* mkdir data
* mkdir results
* mkdir submit
* cd data
* mkdir pretrained

* 把test-A-image.zip  train.zip解压到data成为 data/test-A-image data/train
* mkdir data/train/annotations

配置conda环境和依赖:
*	conda create -n underwater python=3.7 -y 
*	conda activate underwater
*	conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
*	pip install -r requirements/build.txt

注意pillow版本可能会过高
*	conda install pillow=6.1

配置mmd	
* pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
* pip install -v -e .  # or "python setup.py develop"

生成数据集
* python tools/data_process/xml2coco.py
* python tools/data_process/generate_test_json.py

预训练模型下载
- 下载mmdetection官方开源的casacde-rcnn-r50-fpn-2x的COCO预训练模型[cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth](https://open-mmlab.oss-cn-beijing.aliyuncs.com/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth)并放置于 data/pretrained 目录下

训练
* chmod +x tools/dist_train.sh 
* ./tools/dist_train.sh configs/cascade_rcnn_r50_fpn_1x.py 4

输出测试结果
* chmod +x tools/dist_test.sh
* ./tools/dist_test.sh configs/cascade_rcnn_r50_fpn_1x.py ./work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth 4  --format_only --options "jsonfile_prefix=./cas_r50"
* mv cas_r50.bbox.json results
* python tools/post_process/json2submit.py --test_json cas_r50.bbox.json --submit_file cas_r50.csv


* ./tools/dist_train.sh configs/cascade_rcnn_r34_fpn_1x_dp.py 4