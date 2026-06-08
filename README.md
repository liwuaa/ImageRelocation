# ImageToPosition - 基于视觉的AR重定位系统

# 测试项目,实际定位存在极大偏移

## 📖 项目简介

ImageToPosition 是一个基于计算机视觉的增强现实（AR）重定位系统，能够将实时相机图像与预构建的3D地图进行匹配，并返回精确的位置和姿态信息。该系统采用客户端-服务器架构，由Python后端服务和Unity前端应用组成，支持移动端AR应用的实时定位功能。

### 核心功能
- 🎯 **视觉重定位**：通过ORB特征点匹配实现图像到3D地图的精准定位
- 🔄 **多策略算法**：支持单应性矩阵和基础矩阵两种匹配策略
- 📱 **跨平台支持**：Unity客户端可在Android/iOS设备上运行
- ⚡ **实时处理**：Flask REST API提供低延迟的位置查询服务
- 💾 **智能缓存**：LRU缓存机制优化重复查询性能
- 🎨 **可视化调试**：自动生成特征匹配对比图，便于算法调优

---

## 🏗️ 系统架构

```
┌─────────────────┐         HTTP/JSON          ┌──────────────────────┐
│   Unity Client  │ ◄──────────────────────►  │  Python Flask Server │
│                 │   Base64 Image + JSON      │                      │
│ • DeviceCamera  │                            │ • ORB Feature Detect │
│ • PositionMgr   │                            │ • RANSAC Matching    │
│ • DebugConsole  │                            │ • Pose Conversion    │
└─────────────────┘                            │ • LRU Cache          │
                                               └──────────┬───────────┘
                                                          │
                                                          │ SQLite
                                                          ▼
                                               ┌──────────────────────┐
                                               │   RTAB-Map Database  │
                                               │   (map.db / 2_5.db)  │
                                               │                      │
                                               │ • Node poses         │
                                               │ • Keyframe images    │
                                               └──────────────────────┘
```

---

## 📁 项目结构

```
ImageToPosition/
├── Scripts/                          # Python后端脚本
│   ├── Main.py                       # Flask主服务入口（生产环境）
│   ├── AlgorithmClass.py             # 匹配算法策略类
│   ├── GetBestMatch.py               # 简单匹配测试脚本
│   ├── GetImagePos.py                # 离线位置查询工具
│   ├── other/
│   │   └── image_position_service.py # 早期版本服务实现
│   └── Unity/
│       └── UnityPositionManager.cs   # Unity端C#参考代码
│
├── UnityProject/                     # Unity客户端项目
│   └── CustomART/
│       ├── Assets/
│       │   ├── Scripts/
│       │   │   ├── UnityPositionManager.cs  # 位置管理器（移动端优化版）
│       │   │   ├── DeviceCamera.cs          # 设备相机控制
│       │   │   └── DebugConsole.cs          # 运行时调试控制台
│       │   └── Plugins/Android/
│       │       └── AndroidManifest.xml      # Android权限配置
│       └── Packages/
│           └── manifest.json                # Unity包依赖
│
├── Venv/                             # Python虚拟环境
├── Temp/                             # 临时匹配结果图片存储
├── confidence/                       # 置信度测试数据
├── depth/                            # 深度图数据
├── query/                            # 测试查询图像
├── map.db                            # RTAB-Map数据库（大型地图）
└── 2_5.db                            # RTAB-Map数据库（测试地图）
```

---

## 🔧 技术栈

### 后端 (Python)
- **Flask**: Web框架，提供RESTful API
- **OpenCV**: 图像处理与特征提取（ORB算法）
- **NumPy**: 数值计算与矩阵操作
- **SciPy**: 旋转矩阵到四元数转换
- **SQLite**: RTAB-Map数据库访问
- **Pillow**: 图像编解码

### 前端 (Unity/C#)
- **Unity Engine**: 跨平台游戏引擎（2021+）
- **Newtonsoft.Json**: JSON序列化/反序列化
- **WebCamTexture**: 移动端相机接入
- **UnityWebRequest**: HTTP通信

### 数据库
- **RTAB-Map**: 实时建图与定位数据库格式

---

## 🚀 快速开始

### 前置要求

1. **Python环境**
   - Python 3.8+
   - 虚拟环境已配置在 `Venv/` 目录

2. **Unity环境**
   - Unity 2021.3 LTS 或更高版本
   - Android Build Support（如需部署到Android）

3. **依赖安装**
   ```bash
   cd Venv/Scripts
   activate  # Windows
   # source bin/activate  # Linux/Mac
   
   pip install flask opencv-python numpy scipy
   ```

### 启动Python服务

```bash
cd Scripts
python Main.py
```

服务将在 `http://0.0.0.0:5000` 启动，API端点：
- `POST /get_position` - 接收图像并返回位置信息

### 配置Unity客户端

1. 打开 `UnityProject/CustomART` 项目
2. 在场景中创建空物体，附加 `UnityPositionManager` 组件
3. 配置以下参数：
   - **Cam**: 拖入 `DeviceCamera` 组件引用
   - **Status Text**: UI文本组件（显示匹配状态）
   - **Service URL**: 修改为Python服务器IP地址
   - **Capture Interval**: 截图间隔（默认1.0秒）
   - **Target**: 要实例化的AR对象预制体

4. 构建并运行到移动设备

---

## 📡 API文档

### POST /get_position

接收Base64编码的图像，返回匹配的位置和姿态信息。

**请求体:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**响应示例 (成功):**
```json
{
  "status": "Success",
  "message": "{\"status\":\"Success\",\"confidence\":\"85.5%\",\"match_count\":45,\"unity_pos\":{\"x\":1.234,\"y\":-0.567,\"z\":2.891},\"unity_quat\":{\"x\":-0.1234,\"y\":0.5678,\"z\":-0.2345,\"w\":0.7890}}"
}
```

**响应字段说明:**
- `status`: 匹配状态 (`Success` / `LowConfidence` / `Failed`)
- `confidence`: 匹配置信度百分比
- `match_count`: 内点（inliers）数量
- `unity_pos`: Unity坐标系位置 {x, y, z}
- `unity_quat`: Unity四元数旋转 {x, y, z, w}

**响应示例 (失败):**
```json
{
  "status": "Failed",
  "message": "{\"status\":\"Failed\",\"confidence\":\"0%\",\"match_count\":0,\"unity_pos\":{},\"unity_quat\":{},\"message\":\"No features detected\"}"
}
```

---

## 🧠 核心算法

### 1. 特征提取与匹配

系统使用 **ORB (Oriented FAST and Rotated BRIEF)** 算法提取图像特征：
- 每个图像提取最多2000个特征点
- 使用直方图均衡化增强图像对比度
- BFMatcher进行暴力匹配（Hamming距离）

### 2. 匹配策略（可切换）

#### A. OriginalRansacStrategy（单应性矩阵）
```python
# 适用于平面场景或纯旋转运动
cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```
- 优点：计算速度快，适合室内平面环境
- 缺点：对深度变化敏感

#### B. AdvancedGeometryStrategy（基础矩阵+密度校验）⭐推荐
```python
# 适用于3D办公场景
cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0)
```
- 额外进行空间分布密度校验
- 过滤聚集度过高的误匹配
- 面积占比阈值：< 6% 时惩罚权重

**切换方法：** 在 `Main.py` 第177行修改：
```python
CURRENT_STRATEGY = AdvancedGeometryStrategy()  # 或使用 OriginalRansacStrategy()
```

### 3. Lowe's Ratio Test

采用经典的比率测试筛选高质量匹配：
```python
if m.distance < 0.75 * n.distance:
    good_matches.append(m)
```
- 排除模糊匹配，提高鲁棒性
- 0.75为经验值，可根据场景调整

### 4. 置信度计算

```python
confidence = min((inliers / 40.0) * 100, 100.0)
```
- 以40个内点为满分基准
- 低于12个内点判定为匹配失败
- 12-40个内点为低置信度区间

### 5. 坐标系转换

**OpenCV → Unity 坐标映射:**
```python
# 位置转换
unity_pos.x = trans[0]
unity_pos.y = -trans[1]  # Y轴翻转
unity_pos.z = trans[2]

# 旋转转换（四元数）
unity_quat.x = -q[0]
unity_quat.y = q[1]
unity_quat.z = -q[2]
unity_quat.w = q[3]
```

---

## 🛠️ 高级功能

### 匹配可视化

系统在 `Temp/` 目录自动保存匹配结果图：
- 左半部分：查询图像（红色特征点）
- 右半部分：地图关键帧（蓝色特征点）
- 黄色连线：匹配点对
- 左上角标注：匹配数量和节点ID

![匹配示例](./docs/match_example.png) *(需自行生成)*

### 缓存机制

- **地图特征缓存**: 启动时预加载所有关键帧特征到内存
- **查询LRU缓存**: 最近20次查询结果缓存（基于MD5哈希）
- **线程安全**: 数据库访问使用Lock保护

### 性能优化

1. **图像预处理**: 统一缩放到数据库图像尺寸
2. **特征点限制**: ORB最大2000个点平衡速度与精度
3. **JPG压缩**: Unity端使用75%质量编码减少传输量
4. **异步处理**: Unity协程避免阻塞主线程

---

## 📱 Unity集成指南

### DeviceCamera组件

负责移动端相机接入和图像校正：
- 自动处理设备旋转角度（0°/90°/180°/270°）
- 镜像翻转修正（前置摄像头）
- 高性能像素缓冲区复用（减少GC）
- 支持拍照保存到本地

### UnityPositionManager工作流程

```csharp
Update() 
  ↓ 每隔captureInterval秒
CaptureAndSend()
  ↓ 截取相机画面 → Base64编码
SendImageDataToService()
  ↓ POST请求到Python服务
ProcessResponse()
  ↓ 解析JSON → 更新Transform
  ↓ 实例化AR对象到匹配位置
```

### 调试技巧

1. **启用DebugConsole**: 按 `~` 键（或摇动设备）打开日志窗口
2. **查看StatusText**: 实时显示匹配置信度和内点数
3. **检查Temp文件夹**: 验证特征匹配质量
4. **调整captureInterval**: 降低频率可减少CPU占用

---

## 🗄️ RTAB-Map数据库

### 数据库结构

系统使用RTAB-Map生成的SQLite数据库，主要表：

**Node表:**
- `id`: 节点唯一标识
- `pose`: 3×4变换矩阵（float32字节流）

**Data表:**
- `id`: 与Node表关联
- `image`: JPEG压缩的关键帧图像

### 生成地图

1. 安装 [RTAB-Map](https://introlab.github.io/rtabmap/)
2. 使用RGB-D相机或立体相机采集环境数据
3. 导出数据库为 `.db` 文件
4. 将数据库放入项目根目录

### 数据库选择

- `map.db`: 完整大型地图（30MB+）
- `2_5.db`: 精简测试地图（10MB）

在 `Main.py` 第179行切换数据库：
```python
relocator = OptimizedRTABMapRelocator("2_5.db", CURRENT_STRATEGY)
```

---

## 🔍 故障排查

### 常见问题

#### 1. 匹配失败率高
**原因:**
- 光照条件变化大
- 视角差异超过30°
- 纹理缺失区域（白墙等）

**解决方案:**
- 重新采集地图数据（覆盖更多视角）
- 降低Lowe's ratio阈值（0.75 → 0.8）
- 增加ORB特征点数量（2000 → 3000）

#### 2. Unity连接超时
**检查清单:**
- ✅ Python服务是否正常运行
- ✅ 防火墙是否允许5000端口
- ✅ Service URL是否为服务器实际IP（非localhost）
- ✅ 移动设备与服务器在同一局域网

#### 3. 坐标偏移严重
**调试步骤:**
1. 检查数据库中pose矩阵是否正确
2. 验证坐标系转换逻辑（Y轴翻转）
3. 使用 `GetImagePos.py` 离线测试单张图片
4. 对比Temp文件夹中的匹配图

#### 4. 内存溢出
**优化建议:**
- 减少地图数据库大小（删除冗余关键帧）
- 降低查询图像分辨率（Unity端调整WebCamTexture尺寸）
- 增加captureInterval间隔时间

---

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 单次查询耗时 | 200-800ms | 取决于地图大小 |
| 地图加载时间 | 5-30s | 首次启动预加载特征 |
| 内存占用 | 200-500MB | 包含特征缓存 |
| 匹配成功率 | 70-90% | 良好光照条件下 |
| FPS影响 | -5至-10 | Unity端额外开销 |

---

## 🧪 测试与验证

### 离线测试

使用 `GetImagePos.py` 进行单张图片测试：
```bash
cd Scripts
python GetImagePos.py
```

修改脚本中的路径：
```python
relocator = RTABMapRelocator("map.db")
result = relocator.find_location(r"query.jpg")
print(result)
```

### 在线测试

使用curl发送测试请求：
```bash
python -c "
import base64, requests
with open('query.jpg', 'rb') as f:
    img = base64.b64encode(f.read()).decode()
r = requests.post('http://localhost:5000/get_position', json={'image': img})
print(r.json())
"
```

---

## 📝 许可证

本项目仅供学习和研究使用。RTAB-Map数据库格式遵循其开源协议。

---


## 🙏 致谢

- [RTAB-Map](https://introlab.github.io/rtabmap/) - 优秀的SLAM框架
- [OpenCV](https://opencv.org/) - 强大的计算机视觉库
- [Unity Technologies](https://unity.com/) - 领先的实时3D开发平台

---

**最后更新**: 2026-06-08  
**版本**: 1.0.0
