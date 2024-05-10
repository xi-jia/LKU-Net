# 预处理

## 图像预处理

我们只使用min max normalization;
假设min = -1000, max = 500, 则
precessed_image = image.clip(min, max)
precessed_image= (precessed_image - min)/(max - min)

在Functions.py里面,我们使用了六种不同的min max 组合来**增强**原始数据。

frame = np.random.choice(['imagesTr_400.0_-1000.0', 'imagesTr_450.0_-1000.0', 'imagesTr_500.0_-1000.0','imagesTr_400.0_-900.0', 'imagesTr_450.0_-900.0', 'imagesTr_500.0_-900.0'])

## Key points 预处理

为了利用提供的landmark / keypoints， 我们将这些keypoints 转换成了图像，一个keypoints round 之后 对应 3x3x3的patch。
那么有多少个关键点，预处理之后在HxWxD的灰度图像（初始化为0，黑色）上就有多少个3x3x3的patch为白色（intensity ==1）。

## Mask 预处理

将提供的Mask和上一步key points组成的mask合并，那么就有两种标签。一种是肺部区域 ==1 ， 一种是keypoints patch == 2.

# 训练

训练非常简单，参数也不需要怎么调节，start channel == 16/32, using_l2 == 2, data:mask:smth== 1.0:1.0:4.0， iteration 100001就能得到很好的结果。

# 测试

将Validation Dice最高的模型进行提交即可。

