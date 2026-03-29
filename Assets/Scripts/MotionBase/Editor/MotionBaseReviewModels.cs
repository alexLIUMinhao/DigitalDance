using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace MotionBase.Editor
{
    [Serializable]
    public class MotionBaseAssetRootsConfig
    {
        public int schemaVersion = 1;
        public MotionBaseAssetRoots roots = new MotionBaseAssetRoots();
    }

    [Serializable]
    public class MotionBaseAssetRoots
    {
        public string rawFbx;
        public string sourceVideo;
        public string extractedMotion;
        public string approvedFbx;
        public string previewCache;
    }

    [Serializable]
    public class MotionBaseMetricsSummary
    {
        public int frameCount;
        public float durationSec;
        public float fps;
        public bool mixamoCompatibleGuess;
        public float loopDeltaScore;
        public float rootTranslationDeltaMeters;
        public float leftFootDeltaMeters;
        public float rightFootDeltaMeters;
    }

    [Serializable]
    public class MotionBaseReviewFeedData
    {
        public int schemaVersion = 1;
        public string generatedAtUtc;
        public List<MotionBaseReviewFeedEntry> entries = new List<MotionBaseReviewFeedEntry>();
    }

    [Serializable]
    public class MotionBaseReviewFeedEntry
    {
        public string candidateId;
        public string jobId;
        public string motionId;
        public string displayName;
        public string sourceLane;
        public string stage;
        public string sourceVideoRelPath;
        public string candidateFbxRelPath;
        public string targetStyleFamily;
        public string targetStyleSubstyle;
        public string expectedMetaAction;
        public string transitionProfile;
        public string operatorNotes;
        public MotionBaseMetricsSummary metricsSummary = new MotionBaseMetricsSummary();
    }

    [Serializable]
    public class MotionBaseCandidateReviewData
    {
        public int schemaVersion = 1;
        public List<MotionBaseCandidateReviewEntry> entries = new List<MotionBaseCandidateReviewEntry>();
    }

    [Serializable]
    public class MotionBaseCandidateReviewEntry
    {
        public string candidateId;
        public string jobId;
        public string motionId;
        public string displayName;
        public string sourceLane;
        public string sourceVideoRelPath;
        public string candidateFbxRelPath;
        public string targetStyleFamily;
        public string targetStyleSubstyle;
        public string expectedMetaAction;
        public string transitionProfile;
        public string energyBand;
        public List<string> preferredSegments = new List<string>();
        public float nativeBpm = 120f;
        public int phraseBeats = 8;
        public List<int> entryOffsetsBeats = new List<int>();
        public List<int> sliceBeatsOptions = new List<int>();
        public string varietyGroup;
        public string role;
        public string qualityTier = "review_hold";
        public string licenseTier = "prototype_only";
        public string decisionStatus = "pending";
        public string reviewStatus = string.Empty;
        public string loopSeam = "manual_fix_needed";
        public string footStability = "monitor";
        public string styleClarity = "mixed";
        public string tempoTolerance = "medium";
        public string reviewer = string.Empty;
        public List<string> issues = new List<string>();
        public string notes = string.Empty;
        public MotionBaseMetricsSummary metricsSummary = new MotionBaseMetricsSummary();
        public string savedAtUtc = string.Empty;
    }

    [Serializable]
    public class MotionBaseIntakeQueueData
    {
        public int schemaVersion = 1;
        public List<MotionBaseIntakeJob> jobs = new List<MotionBaseIntakeJob>();
    }

    [Serializable]
    public class MotionBaseIntakeJob
    {
        public string jobId;
        public string sourceLane;
        public string sourceAssetKind;
        public string sourceAssetRelPath;
        public string targetStyleFamily;
        public string targetStyleSubstyle;
        public string expectedMetaAction;
        public string proposedMotionId;
        public string displayName;
        public string stage;
        public MotionBaseArtifactRelPaths artifactRelPaths = new MotionBaseArtifactRelPaths();
        public string reviewFeedRelPath;
        public string operatorNotes;
        public MotionBaseSourceFingerprint sourceFingerprint = new MotionBaseSourceFingerprint();
        public MotionBasePrecheckSummary precheckSummary = new MotionBasePrecheckSummary();
        public string discoveredAtUtc;
        public string updatedAtUtc;
    }

    [Serializable]
    public class MotionBaseArtifactRelPaths
    {
        public string providerExportRelPath;
        public string candidateMetricsRelPath;
        public List<MotionBaseCandidateSlice> candidateSlices = new List<MotionBaseCandidateSlice>();
    }

    [Serializable]
    public class MotionBaseCandidateSlice
    {
        public string candidateId;
        public string candidateFbxRelPath;
        public string displayName;
        public int phraseBeats;
        public int entryOffsetBeats;
        public int sliceIndex;
        public string transitionProfile;
        public string energyBand;
        public string role;
        public float nativeBpm;
    }

    [Serializable]
    public class MotionBaseSourceFingerprint
    {
        public long sizeBytes;
        public string modifiedAtUtc;
    }

    [Serializable]
    public class MotionBasePrecheckSummary
    {
        public string path;
        public string checkedAtUtc;
        public string status;
        public List<string> issues = new List<string>();
        public List<string> warnings = new List<string>();
        public List<string> manualChecks = new List<string>();
        public int width;
        public int height;
        public float fps;
        public float durationSec;
        public string formatName;
        public long sizeBytes;
    }

    public static class MotionBaseReviewProjectPaths
    {
        public const string WillaAvatarAssetPath = "Assets/Art/Avatars/willa.vrm";
        public const string ReviewCacheAssetPath = "Assets/MotionBaseReviewCache";

        public static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        public static string ReviewFeedPath => Path.Combine(ProjectRoot, "motion_base", "review", "review_feed.json");
        public static string CandidateReviewPath => Path.Combine(ProjectRoot, "motion_base", "review", "candidate_review.json");
        public static string IntakeQueuePath => Path.Combine(ProjectRoot, "motion_base", "intake", "intake_queue.json");
        public static string AssetRootsPath => Path.Combine(ProjectRoot, "motion_base", "config", "asset_roots.local.json");

        public static string EnsureReviewCacheFolder()
        {
            if (!AssetDatabase.IsValidFolder(ReviewCacheAssetPath))
            {
                if (!AssetDatabase.IsValidFolder("Assets"))
                {
                    throw new InvalidOperationException("Unity Assets folder is missing.");
                }

                AssetDatabase.CreateFolder("Assets", "MotionBaseReviewCache");
                AssetDatabase.CreateFolder(ReviewCacheAssetPath, "Candidates");
            }
            else if (!AssetDatabase.IsValidFolder(Path.Combine(ReviewCacheAssetPath, "Candidates").Replace("\\", "/")))
            {
                AssetDatabase.CreateFolder(ReviewCacheAssetPath, "Candidates");
            }

            AssetDatabase.Refresh();
            return ReviewCacheAssetPath;
        }
    }
}
