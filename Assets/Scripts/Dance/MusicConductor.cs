using System;
using System.Collections.Generic;
using UnityEngine;

namespace DanceDemo
{
    public class MusicConductor : MonoBehaviour
    {
        private const double StartLeadTimeSec = 0.15d;
        private const int GrooveSpikeWindowSize = 8;
        private const float StrongDownbeatGrooveThreshold = 0.55f;
        private const float GentleStrongDownbeatGrooveThreshold = 0.34f;
        private const float GrooveSpikeThreshold = 0.08f;

        private SongAnalysisData songData;
        private SongPlaybackProfile playbackProfile;
        private AudioSource audioSource;
        private bool initialized;
        private bool isPlaying;
        private double songDspStartTime;
        private float pausedSongTimeSec;
        private int currentBeatIndex = -1;
        private int currentSegmentIndex = -1;
        private MusicRhythmSnapshot currentRhythm;
        private readonly Queue<float> recentBeatGrooveSamples = new Queue<float>();
        private bool currentStrongBeatWindow;
        private string currentStrongBeatReason = "none";

        public event Action<BeatEventInfo> BeatChanged;
        public event Action<SegmentEventInfo> SegmentChanged;
        public event Action PlaybackCompleted;

        public SongAnalysisData SongData => songData;
        public bool IsPlaying => isPlaying;
        public int CurrentBeatIndex => currentBeatIndex;
        public int CurrentSegmentIndex => currentSegmentIndex;
        public int CurrentBarIndex => songData == null ? -1 : Mathf.FloorToInt(Mathf.Max(0, currentBeatIndex) / (float)Mathf.Max(1, songData.beatsPerBar));
        public SongSegmentData CurrentSegment => songData == null || songData.segments == null || songData.segments.Length == 0 || currentSegmentIndex < 0 || currentSegmentIndex >= songData.segments.Length
            ? null
            : songData.segments[currentSegmentIndex];
        public double SongDspStartTime => songDspStartTime;
        public MusicRhythmSnapshot CurrentRhythm => currentRhythm;

        public float CurrentSongTimeSec
        {
            get
            {
                if (!initialized)
                {
                    return 0f;
                }

                if (!isPlaying)
                {
                    return pausedSongTimeSec;
                }

                return Mathf.Clamp((float)(AudioSettings.dspTime - songDspStartTime), 0f, songData.durationSec);
            }
        }

        public void Initialize(SongAnalysisData data, AudioSource source, SongPlaybackProfile profile = null)
        {
            songData = data;
            playbackProfile = profile != null ? profile.ResolveDefaults() : SongPlaybackProfile.CreateDefault();
            audioSource = source;
            initialized = songData != null && audioSource != null;
            currentBeatIndex = -1;
            currentSegmentIndex = -1;
            pausedSongTimeSec = 0f;
            isPlaying = false;
            currentRhythm = default;
            recentBeatGrooveSamples.Clear();
            currentStrongBeatWindow = false;
            currentStrongBeatReason = "none";
        }

        public void PlayFromStart()
        {
            if (!initialized)
            {
                return;
            }

            currentBeatIndex = -1;
            currentSegmentIndex = -1;
            pausedSongTimeSec = 0f;
            recentBeatGrooveSamples.Clear();
            currentStrongBeatWindow = false;
            currentStrongBeatReason = "none";
            SchedulePlayback(0f, true);
        }

        public void PausePlayback()
        {
            if (!initialized || !isPlaying)
            {
                return;
            }

            pausedSongTimeSec = CurrentSongTimeSec;
            audioSource.Pause();
            isPlaying = false;
        }

        public void ResumePlayback()
        {
            if (!initialized || isPlaying)
            {
                return;
            }

            if (pausedSongTimeSec >= songData.durationSec - 0.01f)
            {
                return;
            }

            songDspStartTime = AudioSettings.dspTime - pausedSongTimeSec;
            audioSource.UnPause();
            isPlaying = true;
        }

        public void StopPlayback(bool resetToStart)
        {
            if (!initialized)
            {
                return;
            }

            audioSource.Stop();
            isPlaying = false;
            pausedSongTimeSec = resetToStart ? 0f : Mathf.Clamp(pausedSongTimeSec, 0f, songData.durationSec);
            if (resetToStart)
            {
                currentBeatIndex = -1;
                currentSegmentIndex = -1;
                currentRhythm = default;
                recentBeatGrooveSamples.Clear();
                currentStrongBeatWindow = false;
                currentStrongBeatReason = "none";
            }
        }

        public void TogglePlayPause()
        {
            if (!initialized)
            {
                return;
            }

            if (isPlaying)
            {
                PausePlayback();
                return;
            }

            if (pausedSongTimeSec >= songData.durationSec - 0.01f)
            {
                PlayFromStart();
                return;
            }

            ResumePlayback();
        }

        public double GetBeatDspTime(int beatIndex)
        {
            if (songData == null || beatIndex < 0)
            {
                return AudioSettings.dspTime;
            }

            return songDspStartTime + songData.GetBeatTime(beatIndex);
        }

        private void SchedulePlayback(float startAtSongTimeSec, bool restartState)
        {
            if (audioSource.clip == null)
            {
                return;
            }

            audioSource.Stop();
            audioSource.time = Mathf.Clamp(startAtSongTimeSec, 0f, audioSource.clip.length);
            songDspStartTime = AudioSettings.dspTime + StartLeadTimeSec - startAtSongTimeSec;
            audioSource.PlayScheduled(songDspStartTime);
            isPlaying = true;

            if (restartState)
            {
                currentBeatIndex = -1;
                currentSegmentIndex = -1;
            }
        }

        private void Update()
        {
            if (!initialized)
            {
                return;
            }

            var songTime = CurrentSongTimeSec;
            if (isPlaying)
            {
                ProcessBeatProgress(songTime);
                if (songTime >= songData.durationSec && audioSource.isPlaying == false)
                {
                    isPlaying = false;
                    pausedSongTimeSec = songData.durationSec;
                    PlaybackCompleted?.Invoke();
                }
            }

            UpdateRhythmSnapshot(songTime);
        }

        private void ProcessBeatProgress(float songTime)
        {
            if (songData == null)
            {
                return;
            }

            while (ShouldAdvanceBeat(songTime, currentBeatIndex + 1))
            {
                currentBeatIndex++;
                var segmentChanged = UpdateSegmentState(songTime, currentBeatIndex);

                var beatWindow = songData.GetBeatWindow(currentBeatIndex);
                var grooveAtBeat = songData.ResolveGrooveStrength(beatWindow.startSec);
                UpdateStrongBeatWindow(currentBeatIndex, beatWindow, CurrentSegment, grooveAtBeat, segmentChanged);
                UpdateRhythmSnapshot(songTime);
                AppendBeatGrooveSample(grooveAtBeat);
                var info = new BeatEventInfo(
                    currentBeatIndex,
                    CurrentBarIndex,
                    songTime,
                    beatWindow.startSec,
                    GetBeatDspTime(currentBeatIndex),
                    CurrentSegment,
                    beatWindow);

                BeatChanged?.Invoke(info);
            }

            UpdateSegmentState(songTime, currentBeatIndex);
        }

        private void UpdateRhythmSnapshot(float songTime)
        {
            if (songData == null)
            {
                currentRhythm = default;
                return;
            }

            var resolvedBeat = Mathf.Max(0, currentBeatIndex);
            var beatWindow = songData.GetBeatWindow(resolvedBeat);
            var perceivedBeatWindow = GetPerceivedBeatWindow(resolvedBeat);
            var duration = Mathf.Max(0.0001f, perceivedBeatWindow.durationSec);
            var beatPhase = Mathf.Clamp01((songTime - perceivedBeatWindow.startSec) / duration);
            var localBpm = beatWindow.localBpm > 0.01f ? beatWindow.localBpm : songData.bpm;
            var danceLocalBpm = perceivedBeatWindow.localBpm > 0.01f ? perceivedBeatWindow.localBpm : (localBpm / Mathf.Max(1, ResolveBeatGrouping()));
            var beatGrouping = ResolveBeatGrouping();
            var perceivedBeatIndex = resolvedBeat / beatGrouping;
            currentRhythm = new MusicRhythmSnapshot
            {
                SongTimeSec = songTime,
                BeatIndex = resolvedBeat,
                BarIndex = Mathf.FloorToInt(resolvedBeat / (float)Mathf.Max(1, songData.beatsPerBar)),
                SegmentIndex = currentSegmentIndex,
                BeatPhase = beatPhase,
                LocalBpm = localBpm,
                DanceLocalBpm = danceLocalBpm * ResolveTempoScale(),
                GrooveStrength = songData.ResolveGrooveStrength(songTime),
                IsDownbeat = beatWindow.isDownbeat,
                IsPerceivedDownbeat = IsPerceivedDownbeat(resolvedBeat),
                IsStrongBeatWindow = currentStrongBeatWindow,
                StrongBeatReason = currentStrongBeatReason,
                PerceivedBeatIndex = perceivedBeatIndex,
                BeatGrouping = beatGrouping,
                EnergyMode = ResolveEnergyMode(),
                BeatWindow = perceivedBeatWindow,
                Segment = CurrentSegment,
            };
        }

        private bool UpdateSegmentState(float songTime, int beatIndex)
        {
            if (songData.segments == null || songData.segments.Length == 0)
            {
                return false;
            }

            var resolvedBeatIndex = Mathf.Max(0, beatIndex);
            var nextSegmentIndex = currentSegmentIndex;
            for (var i = 0; i < songData.segments.Length; i++)
            {
                var segment = songData.segments[i];
                if (resolvedBeatIndex >= segment.startBeat && resolvedBeatIndex < segment.endBeatExclusive)
                {
                    nextSegmentIndex = i;
                    break;
                }
            }

            if (nextSegmentIndex < 0)
            {
                nextSegmentIndex = resolvedBeatIndex >= songData.segments[songData.segments.Length - 1].endBeatExclusive
                    ? songData.segments.Length - 1
                    : 0;
            }

            if (nextSegmentIndex == currentSegmentIndex)
            {
                return false;
            }

            currentSegmentIndex = nextSegmentIndex;
            SegmentChanged?.Invoke(new SegmentEventInfo(currentSegmentIndex, CurrentSegment, songTime));
            return true;
        }

        private bool ShouldAdvanceBeat(float songTime, int nextBeatIndex)
        {
            if (nextBeatIndex < 0)
            {
                return false;
            }

            var beatTime = songData.GetBeatTime(nextBeatIndex);
            if (beatTime > songData.durationSec + 0.01f)
            {
                return false;
            }

            return songTime >= beatTime;
        }

        private void UpdateStrongBeatWindow(int beatIndex, BeatWindowData beatWindow, SongSegmentData segment, float grooveStrength, bool segmentChanged)
        {
            currentStrongBeatWindow = false;
            currentStrongBeatReason = "none";

            if (segmentChanged)
            {
                currentStrongBeatWindow = true;
                currentStrongBeatReason = "segment_change";
                return;
            }

            if (!IsPerceivedDownbeat(beatIndex))
            {
                return;
            }

            var isHighEnergySegment = IsHighEnergySegment(segment);
            if (beatWindow != null)
            {
                if (isHighEnergySegment)
                {
                    currentStrongBeatWindow = true;
                    currentStrongBeatReason = "perceived_segment_downbeat";
                    return;
                }

                if (grooveStrength >= ResolveStrongBeatGrooveThreshold())
                {
                    currentStrongBeatWindow = true;
                    currentStrongBeatReason = "perceived_downbeat_groove";
                    return;
                }
            }

            if (IsGrooveSpike(beatIndex, beatWindow, grooveStrength))
            {
                currentStrongBeatWindow = true;
                currentStrongBeatReason = "groove_spike";
            }
        }

        private bool IsGrooveSpike(int beatIndex, BeatWindowData beatWindow, float grooveStrength)
        {
            if (!IsPerceivedBeatBoundary(beatIndex))
            {
                return false;
            }

            if (recentBeatGrooveSamples.Count == 0)
            {
                return false;
            }

            var averageGroove = 0f;
            foreach (var sample in recentBeatGrooveSamples)
            {
                averageGroove += sample;
            }

            averageGroove /= recentBeatGrooveSamples.Count;
            if (grooveStrength < averageGroove + GrooveSpikeThreshold)
            {
                return false;
            }

            var nearDownbeat = IsPerceivedDownbeat(beatIndex);
            if (!nearDownbeat && beatIndex > 0)
            {
                nearDownbeat = IsPerceivedDownbeat(beatIndex - 1);
            }

            return nearDownbeat;
        }

        private float ResolveTempoScale()
        {
            return playbackProfile != null ? playbackProfile.EffectiveTempoScale : 1f;
        }

        private int ResolveBeatGrouping()
        {
            return playbackProfile != null ? playbackProfile.EffectiveBeatGrouping : 1;
        }

        private string ResolveEnergyMode()
        {
            return playbackProfile != null ? playbackProfile.EffectiveEnergyMode : "balanced";
        }

        private float ResolveStrongBeatGrooveThreshold()
        {
            return string.Equals(ResolveEnergyMode(), "gentle", StringComparison.Ordinal)
                ? GentleStrongDownbeatGrooveThreshold
                : StrongDownbeatGrooveThreshold;
        }

        private bool IsPerceivedBeatBoundary(int beatIndex)
        {
            var beatGrouping = Mathf.Max(1, ResolveBeatGrouping());
            return beatIndex >= 0 && beatIndex % beatGrouping == 0;
        }

        private bool IsPerceivedDownbeat(int beatIndex)
        {
            if (beatIndex < 0 || !IsPerceivedBeatBoundary(beatIndex))
            {
                return false;
            }

            var beatGrouping = Mathf.Max(1, ResolveBeatGrouping());
            var perceivedBeatIndex = beatIndex / beatGrouping;
            return perceivedBeatIndex % Mathf.Max(1, songData != null ? songData.beatsPerBar : 4) == 0;
        }

        private BeatWindowData GetPerceivedBeatWindow(int beatIndex)
        {
            var baseWindow = songData.GetBeatWindow(Mathf.Max(0, beatIndex));
            var beatGrouping = ResolveBeatGrouping();
            if (beatGrouping <= 1 || songData == null)
            {
                return baseWindow;
            }

            var groupStartBeat = Mathf.Max(0, (Mathf.Max(0, beatIndex) / beatGrouping) * beatGrouping);
            var groupEndBeat = groupStartBeat + beatGrouping;
            var beatCount = songData.beatWindows != null && songData.beatWindows.Length > 0
                ? songData.beatWindows.Length
                : (songData.beats != null ? songData.beats.Length : 0);
            var startSec = songData.GetBeatTime(groupStartBeat);
            var endSec = groupEndBeat < beatCount
                ? songData.GetBeatTime(groupEndBeat)
                : songData.durationSec;
            var durationSec = Mathf.Max(0.2f, endSec - startSec);
            return new BeatWindowData
            {
                index = groupStartBeat / beatGrouping,
                startSec = startSec,
                endSec = endSec,
                durationSec = durationSec,
                localBpm = durationSec > 0.001f ? 60f / durationSec : baseWindow.localBpm,
                isDownbeat = IsPerceivedDownbeat(groupStartBeat),
            };
        }

        private void AppendBeatGrooveSample(float grooveStrength)
        {
            recentBeatGrooveSamples.Enqueue(grooveStrength);
            while (recentBeatGrooveSamples.Count > GrooveSpikeWindowSize)
            {
                recentBeatGrooveSamples.Dequeue();
            }
        }

        private static bool IsHighEnergySegment(SongSegmentData segment)
        {
            if (segment == null || string.IsNullOrEmpty(segment.label))
            {
                return false;
            }

            var label = segment.label.ToLowerInvariant();
            return label == "chorus" || label == "instrumental";
        }
    }
}
