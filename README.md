无人机航拍火灾烟雾检测数据集-高清无人机航拍森林起火烟雾检测数据集，1467张，yolo标注方式
图像尺寸:2048*1080![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/333b1ef7e2694edda2a4bc566c2f124a.png)

类别数量:1 类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1328857f30884788b5c60f2bca4618a2.png)

训练集图像数量:1179; 验证集图像数量:248； 测试集图像数量:40
类别名称: smoke![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/35ffd2ffe12b4943bbaea9c63cde00f5.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2016bc295b74abf9e6fd1ea4da95ee6.png)



2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6231e03a949c45fd9b1aa07efc5420d6.png)



3.qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8dcb71f7c8e04f938112dd2de7c0304d.png)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4364bf61a5249899e3ec36f7089cd17.png)
# 基于YOLOv11森林火灾烟雾检测系统完整工程代码
## 1. 数据集yaml配置文件 `smoke_data.yaml`
```yaml
path: ./forest_smoke
train: images/train
val: images/val
test: images/test

names:
  0: smoke
```

## 2. 模型训练代码 `train.py`
```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")
    model.train(
        data="smoke_data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        patience=15,
        cos_lr=True,
        mosaic=0.7,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.6,
        hsv_v=0.4,
        project="runs/train",
        name="forest_smoke_detect"
    )
```

## 3. 验证评估代码 `val.py`
```python
from ultralytics import YOLO

model = YOLO(r"./runs/train/forest_smoke_detect/weights/best.pt")
metrics = model.val(
    data="smoke_data.yaml",
    imgsz=640,
    conf=0.25,
    iou=0.45
)
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"Precision:{metrics.box.mp:.3f}, Recall:{metrics.box.mr:.3f}")
```

## 4. PyQt5可视化界面完整代码 `main_gui.py`
```python
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTableWidgetItem)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
from ui_smoke import Ui_MainWindow

class DetThread(QThread):
    signal_img = pyqtSignal(np.ndarray)
    signal_table = pyqtSignal(list)
    def __init__(self,model,source):
        super().__init__()
        self.model = model
        self.source = source
    def run(self):
        results = self.model.predict(source=self.source,conf=0.25,iou=0.45)
        res = results[0]
        img = res.plot()
        table_data = []
        for box in res.boxes:
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            table_data.append([cls_name,conf,xyxy])
        self.signal_img.emit(img)
        self.signal_table.emit(table_data)

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = YOLO(r"./runs/train/forest_smoke_detect/weights/best.pt")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video.clicked.connect(self.load_video)

    def load_img(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择图片","","Image(*.jpg *.png *.jpeg)")
        if not path:
            return
        self.th = DetThread(self.model,path)
        self.th.signal_img.connect(self.show_img)
        self.th.signal_table.connect(self.fill_table)
        self.th.start()

    def show_img(self,img):
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = rgb.shape
        qimg = QImage(rgb.data,w,h,c*w,QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qimg).scaled(self.label_img.size(),Qt.KeepAspectRatio))

    def fill_table(self,data_list):
        self.tableWidget.setRowCount(len(data_list))
        for i,row in enumerate(data_list):
            name,conf,pos = row
            self.tableWidget.setItem(i,0,QTableWidgetItem(name))
            self.tableWidget.setItem(i,1,QTableWidgetItem(f"{conf:.2f}"))
            self.tableWidget.setItem(i,2,QTableWidgetItem(str(pos)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
```

## 5. 依赖安装命令
```bash
pip install ultralytics opencv-python pyqt5 numpy
```

## 6. UI文件`ui_smoke.ui`转为py命令
```bash
pyuic5 ui_smoke.ui -o ui_smoke.py
```

功能：图片检测、视频检测，展示烟雾检测框、置信度、坐标信息，检测结果表格输出。以上文字及代码仅供参考学习。


