using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class UnityPositionManager : MonoBehaviour
{
    public DeviceCamera cam; // 需要截图的相机
    public float captureInterval = 1.0f; // 截图间隔（秒）
    public TMP_Text statusText; // 显示状态的UI文本组件
    public float maxRequestTimeout = 5.0f; // 请求最大超时时间
    
    [SerializeField] private string serviceUrl = "http://172.30.248.163:5000/get_position"; // 服务地址
    [SerializeField] private Toggle m_Toggle;
    [SerializeField] private GameObject target;
    
    
    private float lastCaptureTime = 0f;
    private bool isProcessing = false; // 是否正在处理

    void Update()
    {
        // 检查是否达到截图时间间隔
        if (m_Toggle.isOn && !isProcessing && Time.time - lastCaptureTime >= captureInterval)
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


        Texture2D imageTex = cam.GetCameraTexture();


        // 将图像编码为Base64
        byte[] imageBytes = imageTex.EncodeToJPG(75); // 使用JPG压缩，质量75%
        string base64Image = Convert.ToBase64String(imageBytes);

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

                var a = Instantiate(target);
                a.SetActive(true);
                
                a.transform.position = newPosition;
                a.transform.rotation = newRotation;

                if (statusText != null)
                    statusText.text = $"Conf: {positionData.confidence}, Matches: {positionData.match_count}";

                Debug.Log($"Position updated: {newPosition}, Rotation: {newRotation}, Confidence: {positionData.confidence}");
            }
            else
            {
                if (statusText != null)
                    statusText.text = $"Failed - {positionData.status}, Conf: {positionData.confidence}";

                Debug.LogWarning($"Position lookup failed: {positionData.status}, Confidence: {positionData.confidence}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error parsing response: {ex.Message}");

            if (statusText != null)
                statusText.text = $"Parse Error: {ex.Message}";
        }
    }
}

// 存储位置和旋转信息
[Serializable]
public class UnityPositionData
{
    public string status;
    public string confidence;
    public int match_count;
    public UnityVector3 unity_pos;
    public UnityQuaternion unity_quat;
}

[Serializable]
public class UnityVector3
{
    public float x, y, z;
}

[Serializable]
public class UnityQuaternion
{
    public float x, y, z, w;
}