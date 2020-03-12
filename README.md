配置项目

* git clone https://github.com/zhengye1995/underwater-objection-detection.git
* cd underwater-objection-detection
* mkdir data
* cd data
* mkdir pretrained
* mkdir results
* mkdir submit
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

训练
* chmod +x tools/dist_train.sh 
* ./tools/dist_train.sh configs/cascade_rcnn_r50_fpn_1x.py 4

输出测试结果(还没试过，暂时没训练好)
* chmod +x tools/dist_test.sh
* ./tools/dist_test.sh configs/cascade_rcnn_r50_fpn_1x.py ./work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth 4  --format_only --options "outfile_prefix=./cascade_rcnn_r50_fpn_1x_results"
