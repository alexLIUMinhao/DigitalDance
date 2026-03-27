using DanceDemo;
using NUnit.Framework;
using System.IO;
using System.Linq;
using UnityEngine;

namespace DanceDemoTests
{
    public class DancePlaybackProfileTests
    {
        private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

        [Test]
        public void SongPlaybackProfile_ResolvesInvalidValuesToDefaults()
        {
            var profile = new SongPlaybackProfile
            {
                tempoScale = 0f,
                beatGrouping = 0,
                energyMode = "unknown",
            };

            var resolved = profile.ResolveDefaults();

            Assert.AreEqual(1f, resolved.EffectiveTempoScale);
            Assert.AreEqual(1, resolved.EffectiveBeatGrouping);
            Assert.AreEqual("balanced", resolved.EffectiveEnergyMode);
        }

        [Test]
        public void SongCatalogEntry_UsesPlaybackProfileDefaultsWhenMissing()
        {
            var entry = new SongCatalogEntry
            {
                songId = "audio1_mp3",
                displayName = "audio1",
            };

            var resolved = entry.EffectivePlaybackProfile;

            Assert.AreEqual(1f, resolved.EffectiveTempoScale);
            Assert.AreEqual(1, resolved.EffectiveBeatGrouping);
            Assert.AreEqual("balanced", resolved.EffectiveEnergyMode);
        }

        [Test]
        public void ChoreographyEngine_GentleModePrefersLowEnergyLoopOverAccentOrHighEnergy()
        {
            var manifest = new MotionManifestData
            {
                defaultPhraseBeats = 8,
                clips = new[]
                {
                    new MotionClipEntry
                    {
                        clipId = "low_loop",
                        resourcePath = "Dance/Clips/low_loop",
                        energyBand = "low_energy",
                        nativeBpm = 96f,
                        speedMin = 0.72f,
                        speedMax = 1.02f,
                        retimeProfile = "gentle",
                        varietyGroup = "low",
                        entryOffsetsBeats = new[] { 0 },
                        sliceBeatsOptions = new[] { 4, 8 },
                        accentBias = 1f,
                    },
                    new MotionClipEntry
                    {
                        clipId = "high_loop",
                        resourcePath = "Dance/Clips/high_loop",
                        energyBand = "high_energy",
                        nativeBpm = 132f,
                        speedMin = 0.9f,
                        speedMax = 1.18f,
                        retimeProfile = "punchy",
                        varietyGroup = "high",
                        entryOffsetsBeats = new[] { 0 },
                        sliceBeatsOptions = new[] { 4, 8 },
                        accentBias = 1f,
                    },
                },
            };

            var engine = new ChoreographyEngine(manifest);
            var context = new DanceSelectionContext
            {
                SongId = "audio1_mp3",
                CurrentBeatIndex = 0,
                CurrentBarIndex = 0,
                CurrentSegment = new SongSegmentData
                {
                    segmentId = "verse_0",
                    label = "verse",
                    startBeat = 0,
                    endBeatExclusive = 32,
                    energy = 0.3f,
                },
                SegmentChanged = true,
                Forced = false,
                PhraseIndex = 0,
                LocalBpm = 148f,
                DanceLocalBpm = 74f,
                GrooveStrength = 0.24f,
                IsDownbeat = true,
                IsPerceivedDownbeat = true,
                AllowAccent = false,
                IsStrongBeatWindow = true,
                StrongBeatReason = "start",
                BeatsSinceLastSwitch = 0,
                PerceivedBeatIndex = 0,
                BeatGrouping = 2,
                EnergyMode = "gentle",
            };

            var selection = engine.SelectNextClip(context);

            Assert.NotNull(selection.SelectedVariant);
            Assert.NotNull(selection.SelectedClip);
            Assert.AreEqual("low_loop", selection.SelectedClip.clipId);
            Assert.AreEqual("loop", selection.SelectedVariant.EffectiveRole);
        }

        [Test]
        public void SongCatalog_Audio1PlaybackProfile_IsGentleManualBeatMode()
        {
            var catalogPath = Path.Combine(ProjectRoot, "Assets/StreamingAssets/DanceData/song_catalog.json");
            var catalog = JsonUtility.FromJson<SongCatalogData>(File.ReadAllText(catalogPath));
            var entry = catalog.songs.FirstOrDefault(song => song.songId == "audio1_mp3");

            Assert.NotNull(entry, "audio1_mp3 entry is missing from song_catalog.json");
            Assert.AreEqual(1, entry.EffectivePlaybackProfile.EffectiveBeatGrouping);
            Assert.AreEqual("gentle", entry.EffectivePlaybackProfile.EffectiveEnergyMode);
            Assert.AreEqual(1f, entry.EffectivePlaybackProfile.EffectiveTempoScale);
        }

        [Test]
        public void Audio1SongAnalysis_UsesManualHalfTimeBeatGrid()
        {
            var songPath = Path.Combine(ProjectRoot, "Assets/StreamingAssets/DanceData/Songs/audio1_mp3.json");
            var analysis = JsonUtility.FromJson<SongAnalysisData>(File.ReadAllText(songPath));

            Assert.NotNull(analysis);
            Assert.Greater(analysis.bpm, 60f);
            Assert.Less(analysis.bpm, 100f, "audio1 should use the slower manual beat grid, not the fast auto BPM.");
            Assert.AreEqual(294, analysis.beats.Length, "audio1 manual beat override should halve the original fast beat grid.");
        }

        [Test]
        public void MotionManifest_HasAtLeastThreeLowEnergyClips()
        {
            var manifestPath = Path.Combine(ProjectRoot, "Assets/StreamingAssets/DanceData/motion_manifest.json");
            var manifest = JsonUtility.FromJson<MotionManifestData>(File.ReadAllText(manifestPath));
            var lowEnergyCount = manifest.clips.Count(clip => clip.energyBand == "low_energy");

            Assert.GreaterOrEqual(lowEnergyCount, 3, "motion_manifest.json should keep at least three low_energy clips available.");
        }

        [Test]
        public void Audio1SongAnalysis_BeatsAreStrictlyIncreasing()
        {
            var songPath = Path.Combine(ProjectRoot, "Assets/StreamingAssets/DanceData/Songs/audio1_mp3.json");
            var analysis = JsonUtility.FromJson<SongAnalysisData>(File.ReadAllText(songPath));

            Assert.NotNull(analysis);
            Assert.NotNull(analysis.beats);
            Assert.Greater(analysis.beats.Length, 1);
            for (var index = 0; index < analysis.beats.Length - 1; index++)
            {
                Assert.Greater(analysis.beats[index + 1], analysis.beats[index], "audio1 beat timeline must be strictly increasing.");
            }
        }

        [Test]
        public void Audio1SongAnalysis_BeatWindowsHavePositiveDuration()
        {
            var songPath = Path.Combine(ProjectRoot, "Assets/StreamingAssets/DanceData/Songs/audio1_mp3.json");
            var analysis = JsonUtility.FromJson<SongAnalysisData>(File.ReadAllText(songPath));

            Assert.NotNull(analysis);
            Assert.NotNull(analysis.beatWindows);
            Assert.Greater(analysis.beatWindows.Length, 0);
            foreach (var beatWindow in analysis.beatWindows)
            {
                Assert.Greater(beatWindow.durationSec, 0f, "audio1 beatWindows must all have positive duration.");
            }
        }
    }
}
