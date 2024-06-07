# Team23 cv final
## 環境架設
* python 3.7
* ```pip install -r requirement.txt```

## 取出影片中的frames、建Hierarchical-B reference table
* 下載影片: http://140.112.48.121:5000/sharing/ahoZKsMDw
* ```python3 yuv2png.py --yuv_file 影片路徑 --output_dir 輸出路徑```
* ```python3 csv_table_generator.py``` 生成 processing_order.csv

## 主程式執行
* ```python3 main.py --input_path 輸入的資料夾 --output_path 輸出的資料夾 --csv_file 順序檔的位置(e.g.processing_order.csv)```
* 輸出的png與selection map結果就會在--output_path所輸入的資料夾
* 輸出的model map結果就會在./model_map資料夾 (若一開始沒有，程式會自己幫你建一個model_map資料夾)

## 生成yuv影片
* ```python3 png2yuv.py --so_path 合成影像的資料夾 --gt_path GT的資料夾 --png_dir 暫時的資料夾 --output_file 影片名稱.yuv -n 129``` \
e.g. ```python3 png2yuv.py --so_path ./output/ --gt_path ./frames/ --png_dir ./temp/ --output_file output.yuv -n 129```