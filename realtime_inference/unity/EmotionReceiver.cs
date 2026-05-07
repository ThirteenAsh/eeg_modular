using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using WebSocketSharp;

public class EmotionReceiver : MonoBehaviour
{
    [Header("WebSocket Settings")]
    public string serverUrl = "ws://localhost:8765";
    public float reconnectDelay = 3f;
    
    [Header("Emotion Settings")]
    public float transitionSmoothTime = 1f;
    
    [Header("Skybox Settings")]
    public Material happySkybox;
    public Material sadSkybox;
    public Material normalSkybox;
    
    [Header("Music Settings")]
    public AudioClip happyMusic;
    public AudioClip sadMusic;
    public AudioClip normalMusic;
    public float musicTransitionTime = 2f;
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    public string currentEmotion = "normal";
    public float currentConfidence = 0f;
    public float transitionProgress = 0f;
    
    private WebSocket ws;
    private Coroutine reconnectCoroutine;
    private string targetEmotion = "normal";
    private float currentMusicVolume = 1f;
    private AudioSource audioSource;
    private AudioSource crossfadeSource;
    private Material currentSkybox;
    private Material targetSkybox;
    private float skyboxBlend = 0f;
    private Dictionary<string, Material> emotionSkyboxes;
    private Dictionary<string, AudioClip> emotionMusic;

    private void Start()
    {
        InitializeEmotionMaps();
        InitializeAudio();
        Connect();
    }

    private void InitializeEmotionMaps()
    {
        emotionSkyboxes = new Dictionary<string, Material>
        {
            {"happy", happySkybox},
            {"sad", sadSkybox},
            {"normal", normalSkybox}
        };
        
        emotionMusic = new Dictionary<string, AudioClip>
        {
            {"happy", happyMusic},
            {"sad", sadMusic},
            {"normal", normalMusic}
        };
        
        currentSkybox = normalSkybox;
        targetSkybox = normalSkybox;
        RenderSettings.skybox = currentSkybox;
    }

    private void InitializeAudio()
    {
        audioSource = gameObject.AddComponent<AudioSource>();
        audioSource.loop = true;
        audioSource.volume = 1f;
        audioSource.playOnAwake = false;
        
        crossfadeSource = gameObject.AddComponent<AudioSource>();
        crossfadeSource.loop = true;
        crossfadeSource.volume = 0f;
        crossfadeSource.playOnAwake = false;
        
        if (normalMusic != null)
        {
            audioSource.clip = normalMusic;
            audioSource.Play();
        }
    }

    private void Connect()
    {
        if (ws != null && ws.IsAlive)
        {
            return;
        }

        Debug.Log($"Connecting to {serverUrl}...");
        ws = new WebSocket(serverUrl);
        
        ws.OnOpen += OnOpen;
        ws.OnMessage += OnMessage;
        ws.OnError += OnError;
        ws.OnClose += OnClose;
        
        ws.ConnectAsync();
    }

    private void OnOpen(object sender, System.EventArgs e)
    {
        Debug.Log("Connected to emotion server!");
        if (reconnectCoroutine != null)
        {
            StopCoroutine(reconnectCoroutine);
            reconnectCoroutine = null;
        }
    }

    private void OnMessage(object sender, MessageEventArgs e)
    {
        try
        {
            var emotionData = JsonUtility.FromJson<EmotionData>(e.Data);
            ProcessEmotionData(emotionData);
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to parse emotion data: {ex.Message}");
        }
    }

    private void ProcessEmotionData(EmotionData data)
    {
        currentConfidence = data.confidence;
        transitionProgress = data.transition_progress;
        
        if (data.emotion != targetEmotion)
        {
            targetEmotion = data.emotion;
            StartEmotionTransition(data.emotion);
        }
    }

    private void StartEmotionTransition(string newEmotion)
    {
        Debug.Log($"Starting emotion transition to: {newEmotion}");
        
        if (emotionSkyboxes.TryGetValue(newEmotion, out Material newSkybox))
        {
            targetSkybox = newSkybox;
            skyboxBlend = 0f;
        }
        
        if (emotionMusic.TryGetValue(newEmotion, out AudioClip newMusic))
        {
            StartCoroutine(CrossfadeMusic(newMusic));
        }
    }

    private IEnumerator CrossfadeMusic(AudioClip newClip)
    {
        if (newClip == null) yield break;
        
        crossfadeSource.clip = newClip;
        crossfadeSource.time = audioSource.time;
        crossfadeSource.volume = 0f;
        crossfadeSource.Play();
        
        float elapsed = 0f;
        while (elapsed < musicTransitionTime)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / musicTransitionTime;
            
            audioSource.volume = 1f - t;
            crossfadeSource.volume = t;
            
            yield return null;
        }
        
        var temp = audioSource;
        audioSource = crossfadeSource;
        crossfadeSource = temp;
        
        crossfadeSource.Stop();
        crossfadeSource.volume = 0f;
        audioSource.volume = 1f;
    }

    private void Update()
    {
        UpdateSkyboxTransition();
    }

    private void UpdateSkyboxTransition()
    {
        if (currentSkybox != targetSkybox)
        {
            skyboxBlend += Time.deltaTime / transitionSmoothTime;
            skyboxBlend = Mathf.Clamp01(skyboxBlend);
            
            if (currentSkybox != null && targetSkybox != null)
            {
                Material blended = new Material(currentSkybox);
                blended.Lerp(currentSkybox, targetSkybox, skyboxBlend);
                RenderSettings.skybox = blended;
            }
            
            if (skyboxBlend >= 1f)
            {
                currentSkybox = targetSkybox;
                currentEmotion = targetEmotion;
            }
        }
    }

    private void OnError(object sender, ErrorEventArgs e)
    {
        Debug.LogError($"WebSocket error: {e.Message}");
    }

    private void OnClose(object sender, CloseEventArgs e)
    {
        Debug.Log($"Disconnected from server. Code: {e.Code}, Reason: {e.Reason}");
        
        if (reconnectCoroutine == null)
        {
            reconnectCoroutine = StartCoroutine(Reconnect());
        }
    }

    private IEnumerator Reconnect()
    {
        while (ws == null || !ws.IsAlive)
        {
            yield return new WaitForSeconds(reconnectDelay);
            Connect();
        }
    }

    private void OnDestroy()
    {
        if (ws != null)
        {
            ws.Close();
            ws = null;
        }
    }

    private void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label("EEG Emotion Receiver", GUI.skin.box);
        GUILayout.Space(10);
        
        GUILayout.Label($"Connection: {(ws != null && ws.IsAlive ? "Connected" : "Disconnected")}");
        GUILayout.Label($"Current Emotion: {currentEmotion}");
        GUILayout.Label($"Confidence: {currentConfidence:F2}");
        GUILayout.Label($"Transition: {transitionProgress:F2}");
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }

    [System.Serializable]
    private class EmotionData
    {
        public string emotion;
        public float confidence;
        public float transition_progress;
        public Probabilities probabilities;
        public float timestamp;
    }

    [System.Serializable]
    private class Probabilities
    {
        public float happy;
        public float sad;
        public float normal;
    }
}
