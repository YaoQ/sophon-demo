# YOLOv5-face模型导出
## 1. 准备工作
YOLOv5模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv5-face官方开源仓库](https://github.com/deepcam-cn/yolov5-face)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。
> **注意**：建议使用`1.8.0+cpu`的torch版本，避免因pytorch版本导致模型编译失败。

## 2. 主要步骤

### 2.1 获取torch模型
从[yolov5-face model](https://github.com/deepcam-cn/yolov5-face#pretrained-models)下载预训练模型.

### 2.2 修改models/yolo.py

YOLOv5-face不同版本的代码导出的YOLOv5模型的输出会有所不同，根据不同的组合可能会有1、2、3、4个输出的情况，主要取决于model/yolo.py文件中的class Detect的forward函数。建议修改Detect类的forward函数的最后return语句，实现1个输出或3个输出。若模型为3输出，需在后处理进行sigmoid和预测坐标的转换。

```python
    ....
    
    class Detect(nn.Module):
    stride = None  # strides computed during build
    export_cat = True  # onnx export cat output

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        #self.no = nc + 5  # number of outputs per anchor
        self.no = nc + 5 + 10  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        if self.export_cat:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.grid[i], self.anchor_grid[i] = self._make_grid_new(nx, ny,i)
                
                ...

        return  x                # 3个输出
        # return torch.cat(z, 1) # 1输出
        ....
```

### 2.3 导出torchscript模型
修改yolov5-face `export.py 用于支持torchscript的导出

```python
parser.add_argument('--torchscript', action='store_true', default=True, help='torchscript model')

...
    y = model(img)  # dry run

    # torch script export
    if opt.torchscript_infer:
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename

        ts = torch.jit.trace(model, img, strict=False)
        d = {'shape': img.shape, 'stride': int(max(model.stride)), 'names': model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        ts.save(str(f), _extra_files=extra_files)
```
生成torchscript模型
```bash
# 下述脚本可能会根据不用版本的YOLOv5有所调整，请以官方仓库说明为准
python3 export.py --weights ${PATH_TO_YOLOV5S_FACE_MODEL}/yolov5s_face.pt --img_size 640 640 --torchscript
```

上述脚本会在原始pt模型所在目录下生成导出的torchscript.pt 模型文件: `yolov5s-face.torchscript.pt`
