using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using UnityEngine.Networking;
using Newtonsoft.Json;
using System;

public class UnityPositionManager : MonoBehaviour
{
    public Camera cam;                           // 需要截图的相机
    public float captureInterval = 1.0f;         // 截图间隔（秒）
    public Text statusText;                      // 显示状态的UI文本组件
    public float maxRequestTimeout = 5.0f;       // 请求最大超时时间
    
    private float lastCaptureTime = 0f;
    private bool isProcessing = false;           // 是否正在处理
    private string serviceUrl = "http://localhost:5000/get_position";  // 服务地址
    
    // 存储位置和旋转信息
    [System.Serializable]
    public class UnityPositionData
    {
        public string status;
        public string confidence;
        public int match_count;
        public UnityVector3 unity_pos;
        public UnityQuaternion unity_quat;
    }
    
    [System.Serializable]
    public class UnityVector3
    {
        public float x, y, z;
    }
    
    [System.Serializable]
    public class UnityQuaternion
    {
        public float x, y, z, w;
    }

    void Start()
    {
        if (cam == null)
            cam = Camera.main;  // 如果未设置，则使用主相机
    }

    void Update()
    {
        // 检查是否达到截图时间间隔
        if (!isProcessing && Time.time - lastCaptureTime >= captureInterval)
        {
            StartCoroutine(CaptureAndSend());
            lastCaptureTime = Time.time;
        }
    }

    IEnumerator CaptureAndSend()
    {
        yield return new WaitForEndOfFrame(); // 等待帧结束以确保截图正确
        
        isProcessing = true;
        
        if (statusText != null)
            statusText.text = "Capturing...";
        
        // 截取当前相机画面
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;
        
        Texture2D imageTex = new Texture2D(cam.pixelWidth, cam.pixelHeight, TextureFormat.RGB24, false);
        imageTex.ReadPixels(new Rect(0, 0, cam.pixelWidth, cam.pixelHeight), 0, 0);
        imageTex.Apply();
        
        RenderTexture.active = currentRT; // 恢复渲染纹理
        
        // 将图像编码为Base64
        byte[] imageBytes = imageTex.EncodeToJPG(75); // 使用JPG压缩，质量75%
        string base64Image = System.Convert.ToBase64String(imageBytes);
        
        // 准备JSON数据
        var requestData = new Dictionary<string, object>();
        requestData["image"] = base64Image;
        
        string jsonData = JsonConvert.SerializeObject(requestData);
        
        Destroy(imageTex); // 清理临时纹理
        
        // 发送数据到Python服务
        StartCoroutine(SendImageDataToService(jsonData));
    }
    
    IEnumerator SendImageDataToService(string jsonData)
    {
        using (UnityWebRequest www = new UnityWebRequest(serviceUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            // 设置超时
            www.timeout = (int)maxRequestTimeout;

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Error sending image to service: {www.error}");
                
                if (statusText != null)
                    statusText.text = $"Network Error: {www.error}";
            }
            else
            {
                ProcessResponse(www.downloadHandler.text);
            }
        }
        
        isProcessing = false;
    }
    
    void ProcessResponse(string jsonResponse)
    {
        try
        {
            // 解析响应数据
            UnityPositionData positionData = JsonConvert.DeserializeObject<UnityPositionData>(jsonResponse);
            
            if (positionData.status == "Success")
            {
                // 更新物体的位置和旋转
                Vector3 newPosition = new Vector3(positionData.unity_pos.x, positionData.unity_pos.y, positionData.unity_pos.z);
                Quaternion newRotation = new Quaternion(positionData.unity_quat.x, positionData.unity_quat.y, positionData.unity_quat.z, positionData.unity_quat.w);
                
                transform.position = newPosition;
                transform.rotation = newRotation;
                
                if (statusText != null)
                    statusText.text = $"Conf: {positionData.confidence}, Matches: {positionData.match_count}";
                
                Debug.Log($"Position updated: {newPosition}, Rotation: {newRotation}, Confidence: {positionData.confidence}");
            }
            else
            {
                if (statusText != null)
                    statusText.text = $"Failed - {positionData.message}, Conf: {positionData.confidence}";
                
                Debug.LogWarning($"Position lookup failed: {positionData.message}, Confidence: {positionData.confidence}");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Error parsing response: {ex.Message}");
            
            if (statusText != null)
                statusText.text = $"Parse Error: {ex.Message}";
        }
    }
}