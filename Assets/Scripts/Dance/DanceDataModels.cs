using System;
using System.Collections.Generic;
using UnityEngine;

namespace DanceDemo
{
    [Serializable]
    public class SongAnalysisData
    {
        public string songId;
        public string displayName;
        public string audioResourcePath;
        public float durationSec;
        public float bpm;
        public int beatsPerBar = 4;
        public float[] beats;
        public BeatWindowData[] beatWindows;
        public float[] downbeats;
        public float[] energyEnvelope;
        public GrooveSampleData[] grooveEnvelope;
        public TempoMapEntryData[] tempoMap;
        public SongSegmentData[] segments;
        public string generatedAtUtc;

        public bool HasBeatGrid => beats != null && beats.Length > 0;

        public SongSegmentData GetSegmentForBeat(int beatIndex)
        {
            if (segments == null || segments.Length == 0)
            {
                return null;
            }

            for (var i = 0; i < segments.Length; i++)
            {
                var segment = segments[i];
                if (beatIndex >= segment.startBeat && beatIndex < segment.endBeatExclusive)
                {
                    return segment;
                }
            }

            return segments[segments.Length - 1];
        }

        public BeatWindowData GetBeatWindow(int beatIndex)
        {
            if (beatWindows != null && beatIndex >= 0 && beatIndex < beatWindows.Length)
            {
                return beatWindows[beatIndex];
            }

            var beatTime = GetBeatTime(beatIndex);
            var nextBeatTime = GetBeatTime(beatIndex + 1);
            var duration = Mathf.Max(0.2f, nextBeatTime - beatTime);
            return new BeatWindowData
            {
                index = beatIndex,
                startSec = beatTime,
                endSec = nextBeatTime,
                durationSec = duration,
                localBpm = duration > 0.001f ? 60f / duration : bpm,
                isDownbeat = beatIndex % Mathf.Max(1, beatsPerBar) == 0,
            };
        }

        public float GetBeatTime(int beatIndex)
        {
            if (beatIndex < 0)
            {
                return 0f;
            }

            if (beats != null && beatIndex < beats.Length)
            {
                return beats[beatIndex];
            }

            if (beatWindows != null && beatIndex < beatWindows.Length)
            {
                return beatWindows[beatIndex].startSec;
            }

            var spacing = ResolveFallbackBeatSpacing();
            if (beats != null && beats.Length > 0)
            {
                return beats[beats.Length - 1] + ((beatIndex - (beats.Length - 1)) * spacing);
            }

            if (beatWindows != null && beatWindows.Length > 0)
            {
                return beatWindows[beatWindows.Length - 1].startSec + ((beatIndex - (beatWindows.Length - 1)) * spacing);
            }

            return beatIndex * spacing;
        }

        public float ResolveGrooveStrength(float songTimeSec)
        {
            if (grooveEnvelope == null || grooveEnvelope.Length == 0)
            {
                return 0.5f;
            }

            if (grooveEnvelope.Length == 1 || songTimeSec <= grooveEnvelope[0].timeSec)
            {
                return grooveEnvelope[0].value;
            }

            for (var i = 0; i < grooveEnvelope.Length - 1; i++)
            {
                var a = grooveEnvelope[i];
                var b = grooveEnvelope[i + 1];
                if (songTimeSec <= b.timeSec)
                {
                    var duration = Mathf.Max(0.0001f, b.timeSec - a.timeSec);
                    var t = Mathf.Clamp01((songTimeSec - a.timeSec) / duration);
                    return Mathf.Lerp(a.value, b.value, t);
                }
            }

            return grooveEnvelope[grooveEnvelope.Length - 1].value;
        }

        public float ResolveFallbackBeatSpacing()
        {
            if (bpm > 0.01f)
            {
                return 60f / bpm;
            }

            if (tempoMap != null && tempoMap.Length > 0 && tempoMap[0].bpm > 0.01f)
            {
                return 60f / tempoMap[0].bpm;
            }

            if (beats != null && beats.Length >= 2)
            {
                var start = beats[0];
                var end = beats[beats.Length - 1];
                var intervals = Mathf.Max(1, beats.Length - 1);
                return Mathf.Max(0.2f, (end - start) / intervals);
            }

            return 0.6f;
        }
    }

    [Serializable]
    public class BeatWindowData
    {
        public int index;
        public float startSec;
        public float endSec;
        public float durationSec;
        public float localBpm;
        public bool isDownbeat;
    }

    [Serializable]
    public class GrooveSampleData
    {
        public float timeSec;
        public float value;
    }

    [Serializable]
    public class TempoMapEntryData
    {
        public float startSec;
        public float endSec;
        public float bpm;
    }

    [Serializable]
    public class SongSegmentData
    {
        public string segmentId;
        public string label;
        public int startBeat;
        public int endBeatExclusive;
        public float energy;

        public override string ToString()
        {
            return string.Format("{0} [{1}, {2})", string.IsNullOrEmpty(label) ? "segment" : label, startBeat, endBeatExclusive);
        }
    }

    [Serializable]
    public class MotionManifestData
    {
        public string libraryId;
        public int defaultPhraseBeats = 8;
        public MotionClipEntry[] clips;

        public bool HasClips => clips != null && clips.Length > 0;
    }

    [Serializable]
    public class SongCatalogData
    {
        public string defaultSongId;
        public SongCatalogEntry[] songs;

        public bool HasSongs => songs != null && songs.Length > 0;
    }

    [Serializable]
    public class SongCatalogEntry
    {
        public string songId;
        public string displayName;
        public string analysisPath;
        public string audioResourcePath;
        public SongPlaybackProfile playbackProfile;

        public string EffectiveDisplayName
        {
            get
            {
                if (!string.IsNullOrEmpty(displayName))
                {
                    return displayName;
                }

                if (!string.IsNullOrEmpty(songId))
                {
                    return songId;
                }

                return "Song";
            }
        }

        public SongPlaybackProfile EffectivePlaybackProfile
        {
            get
            {
                return playbackProfile != null ? playbackProfile.ResolveDefaults() : SongPlaybackProfile.CreateDefault();
            }
        }
    }

    [Serializable]
    public class SongPlaybackProfile
    {
        public float tempoScale = 1f;
        public int beatGrouping = 1;
        public string energyMode = "balanced";

        public float EffectiveTempoScale => tempoScale > 0.01f ? tempoScale : 1f;
        public int EffectiveBeatGrouping => Mathf.Max(1, beatGrouping);

        public string EffectiveEnergyMode
        {
            get
            {
                if (string.IsNullOrEmpty(energyMode))
                {
                    return "balanced";
                }

                var normalized = energyMode.ToLowerInvariant();
                switch (normalized)
                {
                    case "gentle":
                    case "balanced":
                    case "driving":
                        return normalized;
                    default:
                        return "balanced";
                }
            }
        }

        public SongPlaybackProfile ResolveDefaults()
        {
            return new SongPlaybackProfile
            {
                tempoScale = EffectiveTempoScale,
                beatGrouping = EffectiveBeatGrouping,
                energyMode = EffectiveEnergyMode,
            };
        }

        public static SongPlaybackProfile CreateDefault()
        {
            return new SongPlaybackProfile();
        }
    }

    [Serializable]
    public class MotionClipEntry
    {
        public string clipId;
        public string displayName;
        public string resourcePath;
        public string clipName;
        public string[] styleTags;
        public string energyBand;
        public string[] preferredSegments;
        public int phraseBeats;
        public bool loopable = true;
        public int cooldownBeats = 8;
        public float nativeBpm = 120f;
        public int nativePhraseBeats = 8;
        public float speedMin = 0.85f;
        public float speedMax = 1.15f;
        public string retimeProfile = "smooth";
        public string varietyGroup;
        public string role = "loop";
        public int[] entryOffsetsBeats;
        public int[] sliceBeatsOptions;
        public int maxConsecutiveSelections = 2;
        public int groupCooldownBeats = 8;
        public float accentBias = 0f;

        public int EffectivePhraseBeats(int defaultPhraseBeats)
        {
            return phraseBeats > 0 ? phraseBeats : Mathf.Max(1, defaultPhraseBeats);
        }

        public int EffectiveNativePhraseBeats(int defaultPhraseBeats)
        {
            return nativePhraseBeats > 0 ? nativePhraseBeats : EffectivePhraseBeats(defaultPhraseBeats);
        }

        public float EffectiveNativeBpm => nativeBpm > 0.01f ? nativeBpm : 120f;
        public float EffectiveSpeedMin => Mathf.Clamp(speedMin > 0.01f ? speedMin : 0.85f, 0.5f, 2f);
        public float EffectiveSpeedMax => Mathf.Max(EffectiveSpeedMin + 0.01f, Mathf.Clamp(speedMax > 0.01f ? speedMax : 1.15f, 0.55f, 2f));
        public string EffectiveVarietyGroup => string.IsNullOrEmpty(varietyGroup) ? clipId : varietyGroup;
        public string EffectiveRole => string.IsNullOrEmpty(role) ? "loop" : role.ToLowerInvariant();
        public int EffectiveMaxConsecutiveSelections => Mathf.Max(1, maxConsecutiveSelections);
        public int EffectiveGroupCooldownBeats => Mathf.Max(0, groupCooldownBeats);

        public int[] EffectiveEntryOffsetsBeats
        {
            get
            {
                if (entryOffsetsBeats == null || entryOffsetsBeats.Length == 0)
                {
                    return new[] { 0 };
                }

                return entryOffsetsBeats;
            }
        }

        public int[] EffectiveSliceBeatsOptions
        {
            get
            {
                if (sliceBeatsOptions == null || sliceBeatsOptions.Length == 0)
                {
                    return new[] { EffectiveNativePhraseBeats(8) };
                }

                return sliceBeatsOptions;
            }
        }
    }

    public class DanceVariantCandidate
    {
        public string variantId;
        public MotionClipEntry sourceClip;
        public string varietyGroup;
        public string role;
        public int startOffsetBeats;
        public int sliceBeats;

        public string SourceClipId => sourceClip != null ? sourceClip.clipId : null;
        public float EffectiveNativeBpm => sourceClip != null ? sourceClip.EffectiveNativeBpm : 120f;
        public int NativePhraseBeats => sourceClip != null ? sourceClip.EffectiveNativePhraseBeats(Mathf.Max(1, sliceBeats)) : Mathf.Max(1, sliceBeats);
        public string EffectiveVarietyGroup => !string.IsNullOrEmpty(varietyGroup) ? varietyGroup : (sourceClip != null ? sourceClip.EffectiveVarietyGroup : "default");
        public string EffectiveRole => string.IsNullOrEmpty(role) ? "loop" : role.ToLowerInvariant();
        public bool IsAccent => string.Equals(EffectiveRole, "accent", StringComparison.OrdinalIgnoreCase);
        public bool IsLoop => !IsAccent;

        public float ResolveStartOffsetTimeSec()
        {
            if (sourceClip == null || sourceClip.EffectiveNativeBpm <= 0.01f || startOffsetBeats <= 0)
            {
                return 0f;
            }

            return startOffsetBeats * (60f / sourceClip.EffectiveNativeBpm);
        }

        public float ResolveSliceDurationSec()
        {
            if (sourceClip == null || sourceClip.EffectiveNativeBpm <= 0.01f || sliceBeats <= 0)
            {
                return 0f;
            }

            return sliceBeats * (60f / sourceClip.EffectiveNativeBpm);
        }
    }

    public struct BeatEventInfo
    {
        public BeatEventInfo(int beatIndex, int barIndex, float songTimeSec, float beatTimeSec, double beatDspTime, SongSegmentData activeSegment, BeatWindowData beatWindow)
        {
            BeatIndex = beatIndex;
            BarIndex = barIndex;
            SongTimeSec = songTimeSec;
            BeatTimeSec = beatTimeSec;
            BeatDspTime = beatDspTime;
            ActiveSegment = activeSegment;
            BeatWindow = beatWindow;
        }

        public int BeatIndex { get; }
        public int BarIndex { get; }
        public float SongTimeSec { get; }
        public float BeatTimeSec { get; }
        public double BeatDspTime { get; }
        public SongSegmentData ActiveSegment { get; }
        public BeatWindowData BeatWindow { get; }
    }

    public struct SegmentEventInfo
    {
        public SegmentEventInfo(int segmentIndex, SongSegmentData segment, float songTimeSec)
        {
            SegmentIndex = segmentIndex;
            Segment = segment;
            SongTimeSec = songTimeSec;
        }

        public int SegmentIndex { get; }
        public SongSegmentData Segment { get; }
        public float SongTimeSec { get; }
    }

    public struct MusicRhythmSnapshot
    {
        public float SongTimeSec;
        public int BeatIndex;
        public int BarIndex;
        public int SegmentIndex;
        public float BeatPhase;
        public float LocalBpm;
        public float DanceLocalBpm;
        public float GrooveStrength;
        public bool IsDownbeat;
        public bool IsPerceivedDownbeat;
        public bool IsStrongBeatWindow;
        public string StrongBeatReason;
        public int PerceivedBeatIndex;
        public int BeatGrouping;
        public string EnergyMode;
        public BeatWindowData BeatWindow;
        public SongSegmentData Segment;
    }

    public class DanceSelectionContext
    {
        public string SongId;
        public int CurrentBeatIndex;
        public int CurrentBarIndex;
        public SongSegmentData CurrentSegment;
        public string CurrentClipId;
        public string CurrentVariantId;
        public string CurrentVarietyGroup;
        public string CurrentRole;
        public bool SegmentChanged;
        public bool Forced;
        public int LastActivatedBeatIndex;
        public int PhraseIndex;
        public float LocalBpm;
        public float DanceLocalBpm;
        public float CurrentSpeed;
        public float TargetSpeed;
        public float RetimeStress;
        public int ClampHitCount;
        public float GrooveStrength;
        public bool IsDownbeat;
        public bool IsPerceivedDownbeat;
        public bool AllowAccent;
        public bool IsStrongBeatWindow;
        public string StrongBeatReason;
        public int BeatsSinceLastSwitch;
        public int PerceivedBeatIndex;
        public int BeatGrouping;
        public string EnergyMode;
    }

    public class DanceSelectionResult
    {
        public DanceVariantCandidate SelectedVariant;
        public int CandidateCount;
        public string Reason;
        public string DiversityReason;
        public float Score;

        public MotionClipEntry SelectedClip => SelectedVariant != null ? SelectedVariant.sourceClip : null;
    }

    public class DanceDebugState
    {
        public string SongName;
        public string RuntimeMessage;
        public string PlaybackStateLabel;
        public string SelectedSongName;
        public string PlaybackHint;
        public bool IsBusy;
        public string ActiveCameraName;
        public float SongTimeSec;
        public float DurationSec;
        public float Bpm;
        public float LocalBpm;
        public float DanceBpm;
        public float CurrentSpeed;
        public float TargetSpeed;
        public float GrooveStrength;
        public float AvatarViewportCoverage;
        public float AvatarDistance;
        public int BeatIndex = -1;
        public int BarIndex = -1;
        public string SegmentLabel;
        public string CurrentClipName;
        public string PendingClipName;
        public string NextReason;
        public int CandidateCount;
        public bool IsPlaying;
        public bool RetimeClampHit;
        public bool AvatarInView;
        public bool AvatarCenterVisible;
        public float NativeBpm;
        public int NativePhraseBeats;
        public string CurrentVariantId;
        public string PendingVariantId;
        public string CurrentVariantRole;
        public string CurrentVarietyGroup;
        public int VariantStartOffsetBeats;
        public int VariantSliceBeats;
        public string DiversityReason;
        public bool StrongBeatWindow;
        public string StrongBeatReason;
        public int BeatsSinceLastSwitch;
        public float SwitchPressure;
        public Vector3 CameraPosition;
        public Vector3 CameraRotationEuler;
        public float CameraFieldOfView;
        public Vector3 AvatarRootPosition;
        public Vector3 AvatarBoundsCenter;
        public Vector3 AvatarBoundsSize;
        public float CurrentClipTimeSec;
        public bool CurrentClipIsHumanMotion;
        public int AvailableSongCount;
        public int BeatGrouping;
        public string EnergyMode;
        public string CompactCurrentClip;
        public string CompactNextClip;
        public string CompactReason;
        public string CompactStrongBeat;
        public string CompactSegment;

        public IEnumerable<string> ToLines()
        {
            yield return "Song: " + (string.IsNullOrEmpty(SongName) ? "n/a" : SongName);
            yield return string.Format(
                "Selected Song: {0} | Library: {1}",
                string.IsNullOrEmpty(SelectedSongName) ? "n/a" : SelectedSongName,
                AvailableSongCount);
            yield return string.Format("Time: {0:0.00}s / {1:0.00}s", SongTimeSec, DurationSec);
            yield return string.Format("Tempo: Raw {0:0.0} | Local {1:0.0} | Dance {2:0.0}", Bpm, LocalBpm, DanceBpm);
            yield return string.Format(
                "Rhythm: Group x{0} | Mode: {1}",
                Mathf.Max(1, BeatGrouping),
                string.IsNullOrEmpty(EnergyMode) ? "balanced" : EnergyMode);
            yield return string.Format("Beat: {0} | Bar: {1}", BeatIndex, BarIndex);
            yield return "Segment: " + (string.IsNullOrEmpty(SegmentLabel) ? "n/a" : SegmentLabel);
            yield return string.Format("Speed: {0:0.000} -> {1:0.000}", CurrentSpeed, TargetSpeed);
            yield return string.Format("Groove: {0:0.000} | Clamp: {1}", GrooveStrength, RetimeClampHit ? "hit" : "clear");
            yield return string.Format(
                "Strong Beat: {0} | Reason: {1}",
                StrongBeatWindow ? "yes" : "no",
                string.IsNullOrEmpty(StrongBeatReason) ? "none" : StrongBeatReason);
            yield return string.Format(
                "Switch: {0} beats since | Pressure: {1:0.00}",
                BeatsSinceLastSwitch,
                SwitchPressure);
            yield return string.Format("Clip Native: {0:0.0} BPM | Phrase: {1}", NativeBpm, NativePhraseBeats);
            yield return string.Format(
                "Variant: {0} | Role: {1} | Group: {2}",
                string.IsNullOrEmpty(CurrentVariantId) ? "n/a" : CurrentVariantId,
                string.IsNullOrEmpty(CurrentVariantRole) ? "n/a" : CurrentVariantRole,
                string.IsNullOrEmpty(CurrentVarietyGroup) ? "n/a" : CurrentVarietyGroup);
            yield return string.Format(
                "Variant Offset: {0} beats | Slice: {1} beats",
                VariantStartOffsetBeats,
                VariantSliceBeats);
            yield return string.Format(
                "Camera: {0} | Avatar: {1}/{2} | Cover: {3:0.0}% | Dist: {4:0.00}",
                string.IsNullOrEmpty(ActiveCameraName) ? "n/a" : ActiveCameraName,
                AvatarInView ? "frustum" : "out",
                AvatarCenterVisible ? "center" : "offcenter",
                AvatarViewportCoverage * 100f,
                AvatarDistance);
            yield return string.Format(
                "Cam Pos: {0:0.00}, {1:0.00}, {2:0.00} | Rot: {3:0.0}, {4:0.0}, {5:0.0} | FOV: {6:0.0}",
                CameraPosition.x,
                CameraPosition.y,
                CameraPosition.z,
                CameraRotationEuler.x,
                CameraRotationEuler.y,
                CameraRotationEuler.z,
                CameraFieldOfView);
            yield return string.Format(
                "Avatar Root: {0:0.00}, {1:0.00}, {2:0.00}",
                AvatarRootPosition.x,
                AvatarRootPosition.y,
                AvatarRootPosition.z);
            yield return string.Format(
                "Avatar Bounds: center {0:0.00}, {1:0.00}, {2:0.00} | size {3:0.00}, {4:0.00}, {5:0.00}",
                AvatarBoundsCenter.x,
                AvatarBoundsCenter.y,
                AvatarBoundsCenter.z,
                AvatarBoundsSize.x,
                AvatarBoundsSize.y,
                AvatarBoundsSize.z);
            yield return string.Format(
                "Clip Time: {0:0.00}s | HumanMotion: {1}",
                CurrentClipTimeSec,
                CurrentClipIsHumanMotion ? "yes" : "no");
            yield return "Current Clip: " + (string.IsNullOrEmpty(CurrentClipName) ? "n/a" : CurrentClipName);
            yield return "Pending Clip: " + (string.IsNullOrEmpty(PendingClipName) ? "n/a" : PendingClipName);
            yield return "Pending Variant: " + (string.IsNullOrEmpty(PendingVariantId) ? "n/a" : PendingVariantId);
            yield return "Selection: " + (string.IsNullOrEmpty(NextReason) ? "n/a" : NextReason);
            yield return "Diversity: " + (string.IsNullOrEmpty(DiversityReason) ? "n/a" : DiversityReason);
            yield return "Candidates: " + CandidateCount;
            yield return "Playback: " + (string.IsNullOrEmpty(PlaybackStateLabel) ? (IsPlaying ? "playing" : "paused") : PlaybackStateLabel);
            if (!string.IsNullOrEmpty(PlaybackHint))
            {
                yield return "Hint: " + PlaybackHint;
            }
            if (!string.IsNullOrEmpty(RuntimeMessage))
            {
                yield return "Status: " + RuntimeMessage;
            }
        }

        public IEnumerable<string> ToCompactLines()
        {
            yield return "Song: " + (string.IsNullOrEmpty(SelectedSongName) ? "n/a" : SelectedSongName);
            yield return "Now: " + (string.IsNullOrEmpty(CompactCurrentClip) ? "n/a" : CompactCurrentClip);
            yield return "Next: " + (string.IsNullOrEmpty(CompactNextClip) ? "Hold" : CompactNextClip);
            yield return "Why: " + (string.IsNullOrEmpty(CompactReason) ? "n/a" : CompactReason);
            yield return "Strong Beat: " + (string.IsNullOrEmpty(CompactStrongBeat) ? "no" : CompactStrongBeat);
            yield return "Segment: " + (string.IsNullOrEmpty(CompactSegment) ? "n/a" : CompactSegment);
        }
    }

    public static class DanceDataValidator
    {
        public static List<string> ValidateSong(SongAnalysisData data)
        {
            var errors = new List<string>();
            if (data == null)
            {
                errors.Add("Song analysis JSON is missing or could not be parsed.");
                return errors;
            }

            if (string.IsNullOrEmpty(data.songId))
            {
                errors.Add("Song analysis is missing songId.");
            }

            if (data.beats == null || data.beats.Length == 0)
            {
                errors.Add("Song analysis has no beats.");
            }

            if (data.segments == null || data.segments.Length == 0)
            {
                errors.Add("Song analysis has no segments.");
            }

            if (data.durationSec <= 0f)
            {
                errors.Add("Song analysis has an invalid durationSec.");
            }

            if (data.beatsPerBar <= 0)
            {
                errors.Add("Song analysis has an invalid beatsPerBar.");
            }

            if (data.beatWindows == null || data.beatWindows.Length == 0)
            {
                errors.Add("Song analysis has no beatWindows.");
            }

            if (data.grooveEnvelope == null || data.grooveEnvelope.Length == 0)
            {
                errors.Add("Song analysis has no grooveEnvelope.");
            }

            return errors;
        }

        public static List<string> ValidateSongCatalog(SongCatalogData data)
        {
            var errors = new List<string>();
            if (data == null)
            {
                errors.Add("Song catalog JSON is missing or could not be parsed.");
                return errors;
            }

            if (!data.HasSongs)
            {
                errors.Add("Song catalog contains no songs.");
                return errors;
            }

            var ids = new HashSet<string>();
            foreach (var song in data.songs)
            {
                if (song == null)
                {
                    errors.Add("Song catalog contains a null song entry.");
                    continue;
                }

                if (string.IsNullOrEmpty(song.songId))
                {
                    errors.Add("Song catalog entry is missing songId.");
                }
                else if (!ids.Add(song.songId))
                {
                    errors.Add("Duplicate songId found in song catalog: " + song.songId);
                }

                if (string.IsNullOrEmpty(song.analysisPath))
                {
                    errors.Add("Song " + song.songId + " is missing analysisPath.");
                }

                if (string.IsNullOrEmpty(song.audioResourcePath))
                {
                    errors.Add("Song " + song.songId + " is missing audioResourcePath.");
                }
            }

            return errors;
        }

        public static List<string> ValidateManifest(MotionManifestData data)
        {
            var errors = new List<string>();
            if (data == null)
            {
                errors.Add("Motion manifest JSON is missing or could not be parsed.");
                return errors;
            }

            if (!data.HasClips)
            {
                errors.Add("Motion manifest contains no clips.");
                return errors;
            }

            var ids = new HashSet<string>();
            foreach (var clip in data.clips)
            {
                if (clip == null)
                {
                    errors.Add("Motion manifest contains a null clip entry.");
                    continue;
                }

                if (string.IsNullOrEmpty(clip.clipId))
                {
                    errors.Add("Motion clip entry is missing clipId.");
                }
                else if (!ids.Add(clip.clipId))
                {
                    errors.Add("Duplicate clipId found: " + clip.clipId);
                }

                if (string.IsNullOrEmpty(clip.resourcePath))
                {
                    errors.Add("Clip " + clip.clipId + " is missing resourcePath.");
                }

                if (clip.EffectiveSpeedMax <= clip.EffectiveSpeedMin)
                {
                    errors.Add("Clip " + clip.clipId + " has an invalid speed range.");
                }
            }

            return errors;
        }
    }
}
