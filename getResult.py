import os 
for i in range(1):
	exePath = './tools/dist_test.sh'
	trainConfigFile = './configs/cascade_htc_x101_reTrain_multiTest_0404_testTrain.py'
	outputTrainFile = 'train_{}'.format(i+1)

	modelFile = './work_dirs/cas_x101_64x4d_fpn_htc_reTrain_checkCurve_trainAndVal/epoch_{}.pth'.format(i+1)
	os.system('{} {} {} 4  --format_only --options "jsonfile_prefix={}"'.format(exePath, trainConfigFile, modelFile, outputTrainFile) )
	outputValFile = 'val_{}'.format(i+1)
	valConfigFile = './configs/cascade_htc_x101_reTrain_multiTest_0404_testVal.py'
	os.system('{} {} {} 4  --format_only --options "jsonfile_prefix={}"'.format(exePath, valConfigFile, modelFile, outputValFile) )
	