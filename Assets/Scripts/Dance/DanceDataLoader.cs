using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace DanceDemo
{
    public static class DanceDataLoader
    {
        public static IEnumerator LoadJson<T>(string relativePath, Action<T> onSuccess, Action<string> onError) where T : class
        {
            var absolutePath = Path.Combine(Application.streamingAssetsPath, relativePath);
            if (!absolutePath.Contains("://") && File.Exists(absolutePath))
            {
                T parsed = null;
                try
                {
                    var json = File.ReadAllText(absolutePath);
                    parsed = JsonUtility.FromJson<T>(json);
                }
                catch (Exception ex)
                {
                    onError?.Invoke(string.Format("Failed to read {0}: {1}", relativePath, ex.Message));
                    yield break;
                }

                onSuccess?.Invoke(parsed);
                yield break;
            }

            var uri = absolutePath;
            if (!uri.Contains("://"))
            {
                uri = new Uri(absolutePath).AbsoluteUri;
            }

            using (var request = UnityWebRequest.Get(uri))
            {
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(string.Format("Failed to load {0}: {1}", relativePath, request.error));
                    yield break;
                }

                T parsed = null;
                try
                {
                    parsed = JsonUtility.FromJson<T>(request.downloadHandler.text);
                }
                catch (Exception ex)
                {
                    onError?.Invoke(string.Format("Failed to parse {0}: {1}", relativePath, ex.Message));
                    yield break;
                }

                onSuccess?.Invoke(parsed);
            }
        }
    }
}
