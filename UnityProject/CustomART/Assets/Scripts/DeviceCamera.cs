using System.Collections;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class DeviceCamera : MonoBehaviour
{
    [SerializeField] private RawImage m_rawImage;
    [SerializeField] private Button m_btnDown;
    [SerializeField] private TMP_Text m_text;

    private WebCamTexture m_webCamTexture;
    private Color32[] m_pixelBuffer; // 预分配缓冲区，减少 GC

    private void Start()
    {
        if (m_btnDown != null)
            m_btnDown.onClick.AddListener(TakePic);
        
        StartCoroutine(OpenCamera());
    }

    IEnumerator OpenCamera()
    {
        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);

        if (Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            WebCamDevice[] devices = WebCamTexture.devices;
            if (devices.Length > 0)
            {
                // 请求主流分辨率
                m_webCamTexture = new WebCamTexture(devices[0].name, 1280, 720, 30);
                m_rawImage.texture = m_webCamTexture;
                m_webCamTexture.Play();
            }
        }
    }

    private void Update()
    {
        // 实时修正预览：不仅修正旋转，还需处理镜像问题
        if (m_webCamTexture != null && m_webCamTexture.didUpdateThisFrame)
        {
            float angle = m_webCamTexture.videoRotationAngle;
            m_rawImage.rectTransform.localEulerAngles = new Vector3(0, 0, -angle);

            // 修正缩放比例，防止竖屏时拉伸
            float aspectRatio = (float)m_webCamTexture.width / m_webCamTexture.height;
            if (angle % 180 != 0) aspectRatio = 1f / aspectRatio;
            
            // 镜像处理：某些前置摄像头需要 scaleY = -1
            float scaleY = m_webCamTexture.videoVerticallyMirrored ? -1f : 1f;
            m_rawImage.rectTransform.localScale = new Vector3(1, scaleY, 1);
        }
    }

    /// <summary>
    /// 提取出来的公用方法：高性能获取当前画面纹理
    /// </summary>
    public Texture2D GetCameraTexture()
    {
        if (m_webCamTexture == null || !m_webCamTexture.isPlaying) return null;

        int width = m_webCamTexture.width;
        int height = m_webCamTexture.height;
    
        // 关键点：获取旋转角度。如果结果转了180度，我们在这里手动对齐角度。
        int angle = m_webCamTexture.videoRotationAngle;

        // 1. 获取原始像素数据
        if (m_pixelBuffer == null || m_pixelBuffer.Length != width * height)
            m_pixelBuffer = new Color32[width * height];
    
        m_webCamTexture.GetPixels32(m_pixelBuffer);

        // 2. 确定目标画布尺寸
        bool isRotated = (angle == 90 || angle == 270);
        int targetW = isRotated ? height : width;
        int targetH = isRotated ? width : height;

        Texture2D result = new Texture2D(targetW, targetH, TextureFormat.RGBA32, false);
        Color32[] rotatedPixels = new Color32[m_pixelBuffer.Length];

        // 预计算偏移量
        int wMinus1 = width - 1;
        int hMinus1 = height - 1;

        // 3. 像素重排逻辑 (修复后的坐标映射)
        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++)
            {
                int newIndex;
                switch (angle)
                {
                    case 90:
                        // 顺时针旋转 90 度：(x, y) -> (y, wMinus1 - x)
                        newIndex = (wMinus1 - x) * targetW + y;
                        break;
                    case 180:
                        // 旋转 180 度：(x, y) -> (wMinus1 - x, hMinus1 - y)
                        newIndex = (hMinus1 - y) * targetW + (wMinus1 - x);
                        break;
                    case 270:
                        // 顺时针旋转 270 度 (或逆时针 90)：(x, y) -> (hMinus1 - y, x)
                        newIndex = x * targetW + (hMinus1 - y);
                        break;
                    default:
                        // 0 度：保持原样
                        newIndex = rowOffset + x;
                        break;
                }
                rotatedPixels[newIndex] = m_pixelBuffer[rowOffset + x];
            }
        }

        result.SetPixels32(rotatedPixels);
        result.Apply();
        return result;
    }

    public void TakePic()
    {
        Texture2D finalPic = GetCameraTexture();
        if (finalPic == null) return;

        // 性能优化：将编码和写入移出主线程（Texture2D数据必须在主线程提取）
        byte[] imgBytes = finalPic.EncodeToPNG();
        Destroy(finalPic); // 提取完字节后立即销毁纹理，释放内存

        string savePath = Path.Combine(Application.persistentDataPath, $"Snap_{System.DateTime.Now:yyyyMMdd_HHmmss}.png");
        
        // 使用多线程保存文件，完全不卡主线程
        System.Threading.Tasks.Task.Run(() => {
            File.WriteAllBytes(savePath, imgBytes);
        });

        if (m_text != null) m_text.text = "Saved to: " + savePath;
    }
}
