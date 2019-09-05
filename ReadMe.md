## 钓鱼网站识别

> 识别钓鱼网站及网站的钓鱼对象

#### **Process**

1. 获取钓鱼网站页面截图
2. 通过 YOLO 模型获取网站 Logo 及 Input 输入框的有无及位置信息
3. 针对网站 logo, 通过图像相似度比较获取其钓鱼对象

#### Catalog

- yoloTrainsetGet : 构建用于YOLO模型的数据集
- phishTestsetGet : 构建端到端模型整体的测试集
- ImgSimilarity : 图像相似度比较
- PhishingGUI : 用于端到端模型的GUI