# Team23 cv final
## 環境架設
* python 3.7
* ```pip install -r requirement.txt```

## 取出影片中的frames、建Hierarchical-B reference table
* 下載影片: http://140.112.48.121:5000/sharing/ahoZKsMDw
* ```python3 yuv2png.py --yuv_file 影片路徑 --output_dir 輸出路徑```
* ```python3 csv_table_generator.py```

## main.py

* 下載yolov3.weights: https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights