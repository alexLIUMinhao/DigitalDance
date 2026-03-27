using System;
using System.Collections.Generic;
using UnityEngine;

namespace DanceDemo
{
    public static class DanceAssetResolver
    {
        public static AudioClip ResolveAudioClip(SongAnalysisData songData, string fallbackResourcePath, Action<string> onWarning = null)
        {
            if (songData != null && !string.IsNullOrEmpty(songData.audioResourcePath))
            {
                var clip = Resources.Load<AudioClip>(songData.audioResourcePath);
                if (clip != null)
                {
                    return clip;
                }

                onWarning?.Invoke("Configured audio resource was not found: " + songData.audioResourcePath);
            }

            if (songData == null || string.IsNullOrEmpty(songData.audioResourcePath))
            {
                var discoveredClip = ResolveFirstNonDebugAudioClip();
                if (discoveredClip != null)
                {
                    onWarning?.Invoke("Using discovered audio clip from Resources: " + discoveredClip.name);
                    return discoveredClip;
                }
            }

            if (!string.IsNullOrEmpty(fallbackResourcePath))
            {
                var fallbackClip = Resources.Load<AudioClip>(fallbackResourcePath);
                if (fallbackClip != null)
                {
                    onWarning?.Invoke("Falling back to debug audio clip: " + fallbackClip.name);
                }

                return fallbackClip;
            }

            return null;
        }

        public static Dictionary<string, AnimationClip> ResolveClips(MotionManifestData manifest, Action<string> onWarning)
        {
            var lookup = new Dictionary<string, AnimationClip>();
            if (manifest == null || manifest.clips == null)
            {
                return lookup;
            }

            foreach (var entry in manifest.clips)
            {
                if (entry == null || string.IsNullOrEmpty(entry.clipId) || string.IsNullOrEmpty(entry.resourcePath))
                {
                    continue;
                }

                var resolved = ResolveClip(entry);
                if (resolved == null)
                {
                    onWarning?.Invoke("Could not resolve AnimationClip for " + entry.clipId + " at " + entry.resourcePath);
                    continue;
                }

                if (!resolved.humanMotion)
                {
                    onWarning?.Invoke("Skipping non-humanoid AnimationClip for " + entry.clipId + " at " + entry.resourcePath);
                    continue;
                }

                lookup[entry.clipId] = resolved;
            }

            return lookup;
        }

        public static AnimationClip ResolveClip(MotionClipEntry entry)
        {
            var clips = Resources.LoadAll<AnimationClip>(entry.resourcePath);
            if (clips == null || clips.Length == 0)
            {
                return null;
            }

            if (!string.IsNullOrEmpty(entry.clipName))
            {
                foreach (var clip in clips)
                {
                    if (string.Equals(clip.name, entry.clipName, StringComparison.OrdinalIgnoreCase))
                    {
                        return clip;
                    }
                }
            }

            foreach (var clip in clips)
            {
                if (!clip.name.StartsWith("__preview__", StringComparison.OrdinalIgnoreCase) && clip.humanMotion)
                {
                    return clip;
                }
            }

            foreach (var clip in clips)
            {
                if (!clip.name.StartsWith("__preview__", StringComparison.OrdinalIgnoreCase))
                {
                    return clip;
                }
            }

            return clips[0];
        }

        private static AudioClip ResolveFirstNonDebugAudioClip()
        {
            var allClips = Resources.LoadAll<AudioClip>(string.Empty);
            foreach (var clip in allClips)
            {
                if (clip == null)
                {
                    continue;
                }

                if (!string.Equals(clip.name, "debug_click_track", StringComparison.OrdinalIgnoreCase))
                {
                    return clip;
                }
            }

            return null;
        }
    }
}
