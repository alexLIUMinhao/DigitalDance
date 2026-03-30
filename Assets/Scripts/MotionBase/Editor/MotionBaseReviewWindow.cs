using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.Video;
using UnityEngine.Animations;

namespace MotionBase.Editor
{
    public class MotionBaseReviewWindow : EditorWindow
    {
        private enum StudioTab
        {
            SourceIntake,
            CandidateReview,
        }

        private enum CameraPreset
        {
            FullBody,
            Feet,
            UpperBody,
        }

        private readonly string[] decisionOptions = { "pending", "approved", "hold", "rejected" };
        private readonly string[] reviewStatusOptions = { "", "approved", "hold", "rejected" };
        private readonly string[] loopSeamOptions = { "manual_fix_needed", "clean", "accent_only" };
        private readonly string[] footStabilityOptions = { "stable", "monitor", "unstable" };
        private readonly string[] styleClarityOptions = { "clear", "mixed", "unclear" };
        private readonly string[] tempoToleranceOptions = { "narrow", "medium", "wide" };
        private readonly string[] qualityTierOptions = { "review_hold", "production", "rejected" };
        private readonly string[] licenseTierOptions = { "prototype_only", "replace_before_ship", "commercial_safe" };

        private MotionBaseSourceCatalogData sourceCatalogData = new MotionBaseSourceCatalogData();
        private MotionBaseReviewFeedData reviewFeed = new MotionBaseReviewFeedData();
        private MotionBaseCandidateReviewData candidateReviewData = new MotionBaseCandidateReviewData();
        private MotionBaseIntakeQueueData intakeQueueData = new MotionBaseIntakeQueueData();
        private MotionBaseAssetRootsConfig assetRootsConfig = new MotionBaseAssetRootsConfig();

        private readonly Dictionary<string, MotionBaseSourceCatalogEntry> sourceById = new Dictionary<string, MotionBaseSourceCatalogEntry>(StringComparer.Ordinal);
        private readonly Dictionary<string, MotionBaseCandidateReviewEntry> reviewByCandidateId = new Dictionary<string, MotionBaseCandidateReviewEntry>(StringComparer.Ordinal);
        private readonly Dictionary<string, MotionBaseIntakeJob> jobById = new Dictionary<string, MotionBaseIntakeJob>(StringComparer.Ordinal);

        private PreviewRenderUtility previewUtility;
        private GameObject previewAvatar;
        private Animator previewAnimator;
        private PlayableGraph previewGraph;
        private AnimationClipPlayable previewClipPlayable;

        private GameObject videoPlayerObject;
        private VideoPlayer videoPlayer;
        private RenderTexture videoTexture;

        private MotionBaseIntakeJob selectedSourceJob;
        private MotionBaseSourceCatalogEntry selectedSourceCatalog;
        private int selectedSourceIndex = -1;
        private MotionBaseReviewFeedEntry selectedFeedEntry;
        private MotionBaseCandidateReviewEntry selectedReviewEntry;
        private int selectedIndex = -1;

        private string importedClipAssetPath;
        private AnimationClip importedClip;
        private string currentVideoAbsolutePath;
        private string issueDraft = string.Empty;
        private Vector2 sourceQueueScroll;
        private Vector2 sourceDetailsScroll;
        private Vector2 queueScroll;
        private Vector2 detailsScroll;
        private bool isPlaying;
        private double lastEditorTime;
        private float previewTime;
        private float playbackSpeed = 1f;
        private CameraPreset cameraPreset = CameraPreset.FullBody;
        private StudioTab activeTab = StudioTab.SourceIntake;

        [MenuItem("Tools/Motion Base/Studio")]
        [MenuItem("Tools/Motion Base/Review Queue")]
        public static void Open()
        {
            var window = GetWindow<MotionBaseReviewWindow>("Motion Base Studio");
            window.minSize = new Vector2(1280f, 760f);
            window.Show();
        }

        private void OnEnable()
        {
            CreatePreviewResources();
            LoadAllData();
            EditorApplication.update += OnEditorUpdate;
            lastEditorTime = EditorApplication.timeSinceStartup;
        }

        private void OnDisable()
        {
            EditorApplication.update -= OnEditorUpdate;
            DisposePreviewGraph();
            CleanupPreviewResources();
        }

        private void CreatePreviewResources()
        {
            if (previewUtility == null)
            {
                previewUtility = new PreviewRenderUtility();
                previewUtility.cameraFieldOfView = 30f;
                previewUtility.lights[0].intensity = 1.25f;
                previewUtility.lights[0].transform.rotation = Quaternion.Euler(30f, 30f, 0f);
                previewUtility.lights[1].intensity = 1.1f;
                previewUtility.lights[1].transform.rotation = Quaternion.Euler(340f, 218f, 177f);
            }

            if (videoPlayerObject == null)
            {
                videoPlayerObject = new GameObject("MotionBaseReviewVideoPlayer");
                videoPlayerObject.hideFlags = HideFlags.HideAndDontSave;
                videoPlayer = videoPlayerObject.AddComponent<VideoPlayer>();
                videoPlayer.playOnAwake = false;
                videoPlayer.isLooping = true;
                videoPlayer.audioOutputMode = VideoAudioOutputMode.None;
                videoPlayer.renderMode = VideoRenderMode.RenderTexture;
            }

            if (videoTexture == null)
            {
                videoTexture = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32)
                {
                    hideFlags = HideFlags.HideAndDontSave,
                    name = "MotionBaseReviewVideoTexture",
                };
                videoTexture.Create();
                videoPlayer.targetTexture = videoTexture;
            }
        }

        private void CleanupPreviewResources()
        {
            if (previewAvatar != null)
            {
                DestroyImmediate(previewAvatar);
                previewAvatar = null;
                previewAnimator = null;
            }

            if (previewUtility != null)
            {
                previewUtility.Cleanup();
                previewUtility = null;
            }

            if (videoPlayerObject != null)
            {
                DestroyImmediate(videoPlayerObject);
                videoPlayerObject = null;
                videoPlayer = null;
            }

            if (videoTexture != null)
            {
                videoTexture.Release();
                DestroyImmediate(videoTexture);
                videoTexture = null;
            }
        }

        private void OnEditorUpdate()
        {
            var now = EditorApplication.timeSinceStartup;
            var delta = Mathf.Max(0f, (float)(now - lastEditorTime));
            lastEditorTime = now;

            if (isPlaying)
            {
                previewTime += delta * playbackSpeed;
                var duration = ResolvePreviewDuration();
                if (duration > 0.001f)
                {
                    previewTime = Mathf.Repeat(previewTime, duration);
                }
                else
                {
                    previewTime = 0f;
                }
            }

            SyncPreviewPlayback();
            Repaint();
        }

        private void LoadAllData()
        {
            assetRootsConfig = LoadJson<MotionBaseAssetRootsConfig>(MotionBaseReviewProjectPaths.AssetRootsPath) ?? new MotionBaseAssetRootsConfig();
            sourceCatalogData = LoadJson<MotionBaseSourceCatalogData>(MotionBaseReviewProjectPaths.SourceCatalogPath) ?? new MotionBaseSourceCatalogData();
            reviewFeed = LoadJson<MotionBaseReviewFeedData>(MotionBaseReviewProjectPaths.ReviewFeedPath) ?? new MotionBaseReviewFeedData();
            candidateReviewData = LoadJson<MotionBaseCandidateReviewData>(MotionBaseReviewProjectPaths.CandidateReviewPath) ?? new MotionBaseCandidateReviewData();
            intakeQueueData = LoadJson<MotionBaseIntakeQueueData>(MotionBaseReviewProjectPaths.IntakeQueuePath) ?? new MotionBaseIntakeQueueData();

            sourceCatalogData.sources ??= new List<MotionBaseSourceCatalogEntry>();
            reviewFeed.entries ??= new List<MotionBaseReviewFeedEntry>();
            candidateReviewData.entries ??= new List<MotionBaseCandidateReviewEntry>();
            intakeQueueData.jobs ??= new List<MotionBaseIntakeJob>();

            sourceById.Clear();
            foreach (var entry in sourceCatalogData.sources.Where(source => source != null && !string.IsNullOrEmpty(source.sourceId)))
            {
                sourceById[entry.sourceId] = entry;
            }

            reviewByCandidateId.Clear();
            foreach (var entry in candidateReviewData.entries.Where(candidate => candidate != null && !string.IsNullOrEmpty(candidate.candidateId)))
            {
                entry.issues ??= new List<string>();
                entry.preferredSegments ??= new List<string>();
                entry.entryOffsetsBeats ??= new List<int>();
                entry.sliceBeatsOptions ??= new List<int>();
                reviewByCandidateId[entry.candidateId] = entry;
            }

            jobById.Clear();
            foreach (var job in intakeQueueData.jobs.Where(candidate => candidate != null && !string.IsNullOrEmpty(candidate.jobId)))
            {
                job.artifactRelPaths ??= new MotionBaseArtifactRelPaths();
                job.artifactRelPaths.candidateSlices ??= new List<MotionBaseCandidateSlice>();
                job.precheckSummary ??= new MotionBasePrecheckSummary();
                job.sourceProvenance ??= new MotionBaseSourceProvenance();
                job.sourceReview ??= new MotionBaseSourceReview();
                jobById[job.jobId] = job;
            }

            var sourceJobs = SourceJobs();
            if (sourceJobs.Count == 0)
            {
                selectedSourceIndex = -1;
                selectedSourceJob = null;
                selectedSourceCatalog = null;
            }
            else
            {
                if (selectedSourceIndex < 0 || selectedSourceIndex >= sourceJobs.Count)
                {
                    selectedSourceIndex = 0;
                }
                SelectSourceEntry(selectedSourceIndex);
            }

            if (reviewFeed.entries.Count == 0)
            {
                selectedIndex = -1;
                selectedFeedEntry = null;
                selectedReviewEntry = null;
                importedClip = null;
            }
            else
            {
                if (selectedIndex < 0 || selectedIndex >= reviewFeed.entries.Count)
                {
                    selectedIndex = 0;
                }
                SelectEntry(selectedIndex);
            }
        }

        private static T LoadJson<T>(string path) where T : class
        {
            if (!File.Exists(path))
            {
                return null;
            }

            var json = File.ReadAllText(path);
            if (string.IsNullOrWhiteSpace(json))
            {
                return null;
            }

            return JsonUtility.FromJson<T>(json);
        }

        private static void SaveJson<T>(string path, T payload)
        {
            var json = JsonUtility.ToJson(payload, true);
            File.WriteAllText(path, json + Environment.NewLine);
        }

        private List<MotionBaseIntakeJob> SourceJobs()
        {
            return intakeQueueData.jobs
                .Where(job => job != null && string.Equals(job.sourceAssetKind, "video", StringComparison.Ordinal))
                .OrderBy(job => job.displayName ?? string.Empty, StringComparer.Ordinal)
                .ToList();
        }

        private void SelectSourceEntry(int index)
        {
            var sourceJobs = SourceJobs();
            if (index < 0 || index >= sourceJobs.Count)
            {
                return;
            }

            selectedSourceIndex = index;
            selectedSourceJob = sourceJobs[index];
            selectedSourceCatalog = ResolveSourceCatalogForJob(selectedSourceJob);
            previewTime = 0f;
            PrepareSourceVideo();
        }

        private MotionBaseSourceCatalogEntry ResolveSourceCatalogForJob(MotionBaseIntakeJob job)
        {
            if (job == null)
            {
                return null;
            }

            var sourceId = job.sourceProvenance != null ? job.sourceProvenance.sourceId : string.Empty;
            if (!string.IsNullOrEmpty(sourceId) && sourceById.TryGetValue(sourceId, out var entry))
            {
                return entry;
            }

            return sourceCatalogData.sources.FirstOrDefault(source =>
                source != null &&
                string.Equals(source.localVideoRelPath, job.sourceAssetRelPath, StringComparison.Ordinal));
        }

        private void SelectEntry(int index)
        {
            if (index < 0 || index >= reviewFeed.entries.Count)
            {
                return;
            }

            selectedIndex = index;
            selectedFeedEntry = reviewFeed.entries[index];
            reviewByCandidateId.TryGetValue(selectedFeedEntry.candidateId, out selectedReviewEntry);
            if (selectedReviewEntry == null)
            {
                selectedReviewEntry = new MotionBaseCandidateReviewEntry
                {
                    candidateId = selectedFeedEntry.candidateId,
                    jobId = selectedFeedEntry.jobId,
                    motionId = selectedFeedEntry.motionId,
                    displayName = selectedFeedEntry.displayName,
                    sourceLane = selectedFeedEntry.sourceLane,
                    sourceVideoRelPath = selectedFeedEntry.sourceVideoRelPath,
                    candidateFbxRelPath = selectedFeedEntry.candidateFbxRelPath,
                    targetStyleFamily = selectedFeedEntry.targetStyleFamily,
                    targetStyleSubstyle = selectedFeedEntry.targetStyleSubstyle,
                    expectedMetaAction = selectedFeedEntry.expectedMetaAction,
                    transitionProfile = selectedFeedEntry.transitionProfile,
                    metricsSummary = selectedFeedEntry.metricsSummary ?? new MotionBaseMetricsSummary(),
                    issues = new List<string>(),
                    preferredSegments = new List<string>(),
                    entryOffsetsBeats = new List<int>(),
                    sliceBeatsOptions = new List<int>(),
                };
                candidateReviewData.entries.Add(selectedReviewEntry);
                reviewByCandidateId[selectedReviewEntry.candidateId] = selectedReviewEntry;
            }

            issueDraft = string.Join("\n", selectedReviewEntry.issues ?? new List<string>());
            previewTime = 0f;
            PrepareImportedClip();
            PrepareSourceVideo();
        }

        private void PrepareImportedClip()
        {
            importedClip = null;
            importedClipAssetPath = null;
            DisposePreviewGraph();
            DestroyPreviewAvatar();

            if (selectedReviewEntry == null || string.IsNullOrEmpty(selectedReviewEntry.candidateFbxRelPath))
            {
                return;
            }

            var approvedRoot = assetRootsConfig.roots != null ? assetRootsConfig.roots.approvedFbx : null;
            if (string.IsNullOrEmpty(approvedRoot))
            {
                return;
            }

            var externalPath = Path.Combine(approvedRoot, selectedReviewEntry.candidateFbxRelPath.Replace("/", Path.DirectorySeparatorChar.ToString()));
            if (!File.Exists(externalPath))
            {
                return;
            }

            var cacheRoot = MotionBaseReviewProjectPaths.EnsureReviewCacheFolder();
            var cacheAssetPath = Path.Combine(cacheRoot, "Candidates", $"{selectedReviewEntry.candidateId}.fbx").Replace("\\", "/");
            var cacheAbsolutePath = Path.Combine(MotionBaseReviewProjectPaths.ProjectRoot, cacheAssetPath);
            Directory.CreateDirectory(Path.GetDirectoryName(cacheAbsolutePath) ?? MotionBaseReviewProjectPaths.ProjectRoot);
            File.Copy(externalPath, cacheAbsolutePath, true);

            AssetDatabase.ImportAsset(cacheAssetPath, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
            var importer = AssetImporter.GetAtPath(cacheAssetPath) as ModelImporter;
            if (importer != null)
            {
                importer.importAnimation = true;
                importer.animationType = ModelImporterAnimationType.Human;
                importer.importCameras = false;
                importer.importLights = false;
                importer.materialImportMode = ModelImporterMaterialImportMode.None;
                importer.SaveAndReimport();
            }

            importedClipAssetPath = cacheAssetPath;
            importedClip = AssetDatabase.LoadAllAssetsAtPath(cacheAssetPath)
                .OfType<AnimationClip>()
                .FirstOrDefault(clip => clip != null && !clip.name.StartsWith("__preview__", StringComparison.OrdinalIgnoreCase))
                ?? AssetDatabase.LoadAllAssetsAtPath(cacheAssetPath).OfType<AnimationClip>().FirstOrDefault();

            if (importedClip == null)
            {
                return;
            }

            CreatePreviewAvatar();
            RebuildPlayableGraph();
        }

        private void CreatePreviewAvatar()
        {
            DestroyPreviewAvatar();
            var avatarPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(MotionBaseReviewProjectPaths.WillaAvatarAssetPath);
            if (avatarPrefab == null || previewUtility == null)
            {
                return;
            }

            previewAvatar = Instantiate(avatarPrefab);
            previewAvatar.hideFlags = HideFlags.HideAndDontSave;
            previewUtility.AddSingleGO(previewAvatar);
            previewAnimator = previewAvatar.GetComponent<Animator>() ?? previewAvatar.GetComponentInChildren<Animator>(true);
        }

        private void DestroyPreviewAvatar()
        {
            if (previewAvatar != null)
            {
                DestroyImmediate(previewAvatar);
                previewAvatar = null;
                previewAnimator = null;
            }
        }

        private void DisposePreviewGraph()
        {
            if (previewGraph.IsValid())
            {
                previewGraph.Destroy();
            }
        }

        private void RebuildPlayableGraph()
        {
            DisposePreviewGraph();
            if (previewAnimator == null || importedClip == null)
            {
                return;
            }

            previewGraph = PlayableGraph.Create("MotionBaseReviewGraph");
            previewGraph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
            var output = AnimationPlayableOutput.Create(previewGraph, "MotionBaseReview", previewAnimator);
            previewClipPlayable = AnimationClipPlayable.Create(previewGraph, importedClip);
            previewClipPlayable.SetSpeed(0d);
            output.SetSourcePlayable(previewClipPlayable);
            previewGraph.Play();
            EvaluatePreviewGraph();
        }

        private string CurrentSourceVideoRelPath()
        {
            if (activeTab == StudioTab.SourceIntake)
            {
                return selectedSourceJob != null ? selectedSourceJob.sourceAssetRelPath : string.Empty;
            }

            return selectedReviewEntry != null ? selectedReviewEntry.sourceVideoRelPath : string.Empty;
        }

        private void PrepareSourceVideo()
        {
            currentVideoAbsolutePath = string.Empty;
            if (videoPlayer == null)
            {
                return;
            }

            videoPlayer.Stop();
            videoPlayer.clip = null;
            videoPlayer.source = VideoSource.Url;

            var sourceVideoRelPath = CurrentSourceVideoRelPath();
            if (string.IsNullOrEmpty(sourceVideoRelPath))
            {
                return;
            }

            var videoRoot = assetRootsConfig.roots != null ? assetRootsConfig.roots.sourceVideo : null;
            if (string.IsNullOrEmpty(videoRoot))
            {
                return;
            }

            currentVideoAbsolutePath = Path.Combine(videoRoot, sourceVideoRelPath.Replace("/", Path.DirectorySeparatorChar.ToString()));
            if (!File.Exists(currentVideoAbsolutePath))
            {
                currentVideoAbsolutePath = string.Empty;
                return;
            }

            videoPlayer.url = currentVideoAbsolutePath;
            videoPlayer.isLooping = true;
            videoPlayer.playbackSpeed = playbackSpeed;
            videoPlayer.Prepare();
        }

        private float ResolvePreviewDuration()
        {
            if (activeTab == StudioTab.SourceIntake)
            {
                if (videoPlayer != null && videoPlayer.isPrepared && videoPlayer.length > 0.001d)
                {
                    return (float)videoPlayer.length;
                }
            }

            if (importedClip != null && importedClip.length > 0.001f)
            {
                return importedClip.length;
            }

            if (videoPlayer != null && videoPlayer.isPrepared && videoPlayer.length > 0.001d)
            {
                return (float)videoPlayer.length;
            }

            return 0f;
        }

        private void SyncPreviewPlayback()
        {
            EvaluatePreviewGraph();

            if (videoPlayer == null || string.IsNullOrEmpty(currentVideoAbsolutePath))
            {
                return;
            }

            if (!videoPlayer.isPrepared)
            {
                return;
            }

            if (isPlaying)
            {
                if (!videoPlayer.isPlaying)
                {
                    videoPlayer.Play();
                }
            }
            else if (videoPlayer.isPlaying)
            {
                videoPlayer.Pause();
            }

            videoPlayer.playbackSpeed = playbackSpeed;
            if (Math.Abs(videoPlayer.time - previewTime) > 0.08d)
            {
                videoPlayer.time = previewTime;
            }
        }

        private void EvaluatePreviewGraph()
        {
            if (!previewGraph.IsValid() || importedClip == null)
            {
                return;
            }

            var clipDuration = Mathf.Max(0.001f, importedClip.length);
            var clampedTime = Mathf.Repeat(previewTime, clipDuration);
            previewClipPlayable.SetTime(clampedTime);
            previewGraph.Evaluate(0f);
        }

        private void OnGUI()
        {
            DrawToolbar();
            DrawTabStrip();

            if (activeTab == StudioTab.SourceIntake)
            {
                DrawSourceIntakeTab();
                return;
            }

            if (reviewFeed.entries == null || reviewFeed.entries.Count == 0)
            {
                EditorGUILayout.HelpBox("No review feed entries found. Run build_review_queue.py first.", MessageType.Info);
                return;
            }

            EditorGUILayout.BeginHorizontal();
            DrawQueueColumn();
            DrawPreviewColumn();
            DrawDetailsColumn();
            EditorGUILayout.EndHorizontal();
        }

        private void DrawToolbar()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            if (GUILayout.Button("Refresh", EditorStyles.toolbarButton, GUILayout.Width(72f)))
            {
                LoadAllData();
            }

            var saveLabel = activeTab == StudioTab.SourceIntake ? "Save Source" : "Save Review";
            if (GUILayout.Button(saveLabel, EditorStyles.toolbarButton, GUILayout.Width(96f)))
            {
                if (activeTab == StudioTab.SourceIntake)
                {
                    SaveCurrentSourceReview();
                }
                else
                {
                    SaveCurrentReview();
                }
            }

            GUILayout.Space(8f);
            if (activeTab == StudioTab.SourceIntake)
            {
                if (GUILayout.Button("Approve For Extraction", EditorStyles.toolbarButton, GUILayout.Width(148f)))
                {
                    SetSourceDecision("approved");
                }
                if (GUILayout.Button("Hold", EditorStyles.toolbarButton, GUILayout.Width(60f)))
                {
                    SetSourceDecision("hold");
                }
                if (GUILayout.Button("Reject", EditorStyles.toolbarButton, GUILayout.Width(68f)))
                {
                    SetSourceDecision("rejected");
                }
            }
            else
            {
                if (GUILayout.Button("Approve", EditorStyles.toolbarButton, GUILayout.Width(72f)))
                {
                    SetDecision("approved");
                }
                if (GUILayout.Button("Hold", EditorStyles.toolbarButton, GUILayout.Width(60f)))
                {
                    SetDecision("hold");
                }
                if (GUILayout.Button("Reject", EditorStyles.toolbarButton, GUILayout.Width(68f)))
                {
                    SetDecision("rejected");
                }
            }

            GUILayout.FlexibleSpace();
            var entryCount = activeTab == StudioTab.SourceIntake ? SourceJobs().Count : reviewFeed.entries.Count;
            GUILayout.Label($"Entries: {entryCount}", EditorStyles.miniLabel);
            EditorGUILayout.EndHorizontal();
        }

        private void DrawTabStrip()
        {
            EditorGUILayout.BeginHorizontal();
            var nextTab = GUILayout.Toolbar((int)activeTab, new[] { "Source Intake", "Candidate Review" }, GUILayout.Height(24f));
            if (nextTab != (int)activeTab)
            {
                activeTab = (StudioTab)nextTab;
                previewTime = 0f;
                PrepareSourceVideo();
            }
            EditorGUILayout.EndHorizontal();
            GUILayout.Space(4f);
        }

        private void DrawSourceIntakeTab()
        {
            var sourceJobs = SourceJobs();
            if (sourceJobs.Count == 0)
            {
                EditorGUILayout.HelpBox("No source-video jobs found. Run fetch_source_candidates.py, sync_intake_queue.py, and precheck_sources.py first.", MessageType.Info);
                return;
            }

            EditorGUILayout.BeginHorizontal();
            DrawSourceQueueColumn(sourceJobs);
            DrawSourcePreviewColumn();
            DrawSourceDetailsColumn();
            EditorGUILayout.EndHorizontal();
        }

        private void DrawSourceQueueColumn(List<MotionBaseIntakeJob> sourceJobs)
        {
            EditorGUILayout.BeginVertical(GUILayout.Width(300f));
            GUILayout.Label("Source Intake", EditorStyles.boldLabel);
            sourceQueueScroll = EditorGUILayout.BeginScrollView(sourceQueueScroll, GUILayout.ExpandHeight(true));
            for (var index = 0; index < sourceJobs.Count; index++)
            {
                var job = sourceJobs[index];
                var selected = index == selectedSourceIndex;
                var decision = job.sourceReview != null ? job.sourceReview.decisionStatus : "pending";
                var label = $"{job.displayName}\n{job.targetStyleFamily} / {job.expectedMetaAction} / {decision}";
                var style = new GUIStyle(EditorStyles.miniButton)
                {
                    alignment = TextAnchor.MiddleLeft,
                    wordWrap = true,
                    fixedHeight = 48f,
                };
                if (GUILayout.Toggle(selected, label, style, GUILayout.ExpandWidth(true)))
                {
                    if (!selected)
                    {
                        SelectSourceEntry(index);
                    }
                }
            }
            EditorGUILayout.EndScrollView();
            EditorGUILayout.EndVertical();
        }

        private void DrawSourcePreviewColumn()
        {
            EditorGUILayout.BeginVertical(GUILayout.MinWidth(520f), GUILayout.ExpandWidth(true));
            GUILayout.Label("Source Preview", EditorStyles.boldLabel);
            DrawPlaybackControls();

            var rect = GUILayoutUtility.GetRect(600f, 520f, GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));
            DrawVideoPreview(rect);
            EditorGUILayout.EndVertical();
        }

        private void DrawSourceDetailsColumn()
        {
            EditorGUILayout.BeginVertical(GUILayout.Width(380f));
            GUILayout.Label("Source Details", EditorStyles.boldLabel);
            sourceDetailsScroll = EditorGUILayout.BeginScrollView(sourceDetailsScroll, GUILayout.ExpandHeight(true));

            if (selectedSourceJob == null)
            {
                EditorGUILayout.HelpBox("No source-video job selected.", MessageType.Info);
                EditorGUILayout.EndScrollView();
                EditorGUILayout.EndVertical();
                return;
            }

            selectedSourceJob.sourceReview ??= new MotionBaseSourceReview();
            selectedSourceJob.sourceProvenance ??= new MotionBaseSourceProvenance();
            selectedSourceCatalog = ResolveSourceCatalogForJob(selectedSourceJob);

            EditorGUILayout.LabelField("Source Video", selectedSourceJob.sourceAssetRelPath ?? string.Empty, EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Stage", selectedSourceJob.stage ?? string.Empty);
            EditorGUILayout.LabelField("Display Name", selectedSourceJob.displayName ?? string.Empty);

            GUILayout.Space(8f);
            GUILayout.Label("Provenance", EditorStyles.boldLabel);
            DrawMetric("Provider", selectedSourceJob.sourceProvenance.provider ?? string.Empty);
            DrawMetric("Remote Asset", selectedSourceJob.sourceProvenance.remoteAssetId ?? string.Empty);
            DrawMetric("Creator", selectedSourceJob.sourceProvenance.creatorName ?? string.Empty);
            DrawMetric("License", selectedSourceJob.sourceProvenance.licenseName ?? string.Empty);
            DrawMetric("Query", selectedSourceJob.sourceProvenance.query ?? string.Empty);
            DrawMetric("Downloaded", selectedSourceCatalog != null ? selectedSourceCatalog.downloadedAtUtc ?? string.Empty : string.Empty);

            if (!string.IsNullOrEmpty(selectedSourceJob.sourceProvenance.sourcePageUrl))
            {
                EditorGUILayout.SelectableLabel(selectedSourceJob.sourceProvenance.sourcePageUrl, EditorStyles.textField, GUILayout.Height(36f));
            }

            GUILayout.Space(8f);
            GUILayout.Label("Precheck", EditorStyles.boldLabel);
            DrawMetric("Status", selectedSourceJob.precheckSummary.status ?? string.Empty);
            DrawMetric("Resolution", $"{selectedSourceJob.precheckSummary.width} x {selectedSourceJob.precheckSummary.height}");
            DrawMetric("FPS", $"{selectedSourceJob.precheckSummary.fps:0.0}");
            DrawMetric("Duration", $"{selectedSourceJob.precheckSummary.durationSec:0.00}s");
            if (selectedSourceJob.precheckSummary.warnings != null && selectedSourceJob.precheckSummary.warnings.Count > 0)
            {
                EditorGUILayout.HelpBox($"Warnings: {string.Join(", ", selectedSourceJob.precheckSummary.warnings)}", MessageType.Warning);
            }
            if (selectedSourceJob.precheckSummary.issues != null && selectedSourceJob.precheckSummary.issues.Count > 0)
            {
                EditorGUILayout.HelpBox($"Issues: {string.Join(", ", selectedSourceJob.precheckSummary.issues)}", MessageType.Error);
            }

            GUILayout.Space(8f);
            GUILayout.Label("Source Review", EditorStyles.boldLabel);
            selectedSourceJob.sourceReview.decisionStatus = PopupString("Decision", selectedSourceJob.sourceReview.decisionStatus, decisionOptions);
            selectedSourceJob.sourceReview.reviewer = EditorGUILayout.TextField("Reviewer", selectedSourceJob.sourceReview.reviewer ?? string.Empty);
            selectedSourceJob.sourceReview.notes = EditorGUILayout.TextField("Review Notes", selectedSourceJob.sourceReview.notes ?? string.Empty);

            GUILayout.Space(8f);
            GUILayout.Label("Metadata Override", EditorStyles.boldLabel);
            selectedSourceJob.targetStyleFamily = EditorGUILayout.TextField("Style Family", selectedSourceJob.targetStyleFamily ?? string.Empty);
            selectedSourceJob.targetStyleSubstyle = EditorGUILayout.TextField("Style Substyle", selectedSourceJob.targetStyleSubstyle ?? string.Empty);
            selectedSourceJob.expectedMetaAction = EditorGUILayout.TextField("Meta Action", selectedSourceJob.expectedMetaAction ?? string.Empty);
            selectedSourceJob.proposedMotionId = EditorGUILayout.TextField("Proposed Motion Id", selectedSourceJob.proposedMotionId ?? string.Empty);
            selectedSourceJob.operatorNotes = EditorGUILayout.TextField("Operator Notes", selectedSourceJob.operatorNotes ?? string.Empty);

            EditorGUILayout.EndScrollView();
            EditorGUILayout.EndVertical();
        }

        private void DrawQueueColumn()
        {
            EditorGUILayout.BeginVertical(GUILayout.Width(280f));
            GUILayout.Label("Review Queue", EditorStyles.boldLabel);
            queueScroll = EditorGUILayout.BeginScrollView(queueScroll, GUILayout.ExpandHeight(true));
            for (var index = 0; index < reviewFeed.entries.Count; index++)
            {
                var entry = reviewFeed.entries[index];
                var selected = index == selectedIndex;
                var review = reviewByCandidateId.TryGetValue(entry.candidateId, out var reviewEntry) ? reviewEntry : null;
                var state = review != null ? review.decisionStatus : "pending";
                var label = $"{entry.displayName}\n{entry.targetStyleFamily} / {entry.expectedMetaAction} / {state}";
                var style = new GUIStyle(EditorStyles.miniButton)
                {
                    alignment = TextAnchor.MiddleLeft,
                    wordWrap = true,
                    fixedHeight = 48f,
                };
                if (GUILayout.Toggle(selected, label, style, GUILayout.ExpandWidth(true)))
                {
                    if (!selected)
                    {
                        SelectEntry(index);
                    }
                }
            }
            EditorGUILayout.EndScrollView();
            EditorGUILayout.EndVertical();
        }

        private void DrawPreviewColumn()
        {
            EditorGUILayout.BeginVertical(GUILayout.MinWidth(520f), GUILayout.ExpandWidth(true));
            GUILayout.Label("Preview", EditorStyles.boldLabel);

            DrawPlaybackControls();

            var rect = GUILayoutUtility.GetRect(600f, 520f, GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));
            var leftRect = new Rect(rect.x, rect.y, rect.width * 0.5f - 4f, rect.height);
            var rightRect = new Rect(rect.x + rect.width * 0.5f + 4f, rect.y, rect.width * 0.5f - 4f, rect.height);

            DrawVideoPreview(leftRect);
            DrawAvatarPreview(rightRect);

            EditorGUILayout.EndVertical();
        }

        private void DrawPlaybackControls()
        {
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(isPlaying ? "Pause" : "Play", GUILayout.Width(72f)))
            {
                isPlaying = !isPlaying;
            }

            if (GUILayout.Button("Jump Start", GUILayout.Width(88f)))
            {
                previewTime = 0f;
            }

            if (GUILayout.Button("Jump End", GUILayout.Width(82f)))
            {
                var duration = ResolvePreviewDuration();
                previewTime = Mathf.Max(0f, duration - 0.35f);
            }

            playbackSpeed = GUILayout.Toolbar(playbackSpeed switch
            {
                <= 0.95f => 0,
                >= 1.05f => 2,
                _ => 1,
            }, new[] { "0.9x", "1.0x", "1.1x" }, GUILayout.Width(180f)) switch
            {
                0 => 0.9f,
                2 => 1.1f,
                _ => 1.0f,
            };

            if (activeTab == StudioTab.CandidateReview)
            {
                cameraPreset = (CameraPreset)EditorGUILayout.EnumPopup(cameraPreset, GUILayout.Width(120f));
            }
            EditorGUILayout.EndHorizontal();

            var durationSec = ResolvePreviewDuration();
            var sliderLimit = Mathf.Max(0.1f, durationSec > 0f ? durationSec : 10f);
            previewTime = EditorGUILayout.Slider("Time", previewTime, 0f, sliderLimit);
        }

        private void DrawVideoPreview(Rect rect)
        {
            GUI.Box(rect, GUIContent.none);
            if (videoTexture != null && videoPlayer != null && videoPlayer.isPrepared)
            {
                GUI.DrawTexture(rect, videoTexture, ScaleMode.ScaleToFit, true);
            }
            else
            {
                var missingMessage = activeTab == StudioTab.SourceIntake
                    ? "No source video resolved for this intake job."
                    : "No source video resolved for this candidate.";
                EditorGUI.HelpBox(rect, string.IsNullOrEmpty(currentVideoAbsolutePath) ? missingMessage : "Source video is loading or unavailable.", MessageType.Info);
            }

            GUI.Label(new Rect(rect.x + 8f, rect.y + 8f, rect.width - 16f, 20f), "Source Video", EditorStyles.boldLabel);
        }

        private void DrawAvatarPreview(Rect rect)
        {
            GUI.Box(rect, GUIContent.none);
            if (previewUtility == null || previewAvatar == null || importedClip == null)
            {
                EditorGUI.HelpBox(rect, "No candidate FBX is loaded into the review cache.", MessageType.Info);
                return;
            }

            previewUtility.BeginPreview(rect, GUIStyle.none);
            ConfigurePreviewCamera();
            previewUtility.Render();
            var texture = previewUtility.EndPreview();
            GUI.DrawTexture(rect, texture, ScaleMode.StretchToFill, false);
            GUI.Label(new Rect(rect.x + 8f, rect.y + 8f, rect.width - 16f, 20f), "Avatar Preview", EditorStyles.boldLabel);
        }

        private void ConfigurePreviewCamera()
        {
            if (previewUtility == null || previewAvatar == null)
            {
                return;
            }

            var bounds = CalculateAvatarBounds(previewAvatar);
            var center = bounds.center;
            var size = bounds.size;
            var height = Mathf.Max(1.5f, size.y);
            var width = Mathf.Max(0.8f, size.x);
            var target = center;
            var position = center + new Vector3(0.05f, height * 0.2f, height * 1.7f);

            switch (cameraPreset)
            {
                case CameraPreset.Feet:
                    target = new Vector3(center.x, bounds.min.y + 0.25f, center.z);
                    position = target + new Vector3(0.05f, 0.25f, Mathf.Max(1.2f, width * 1.8f));
                    break;
                case CameraPreset.UpperBody:
                    target = new Vector3(center.x, center.y + height * 0.22f, center.z);
                    position = target + new Vector3(0.05f, 0.12f, Mathf.Max(1.1f, width * 1.6f));
                    break;
                default:
                    target = center + new Vector3(0f, height * 0.08f, 0f);
                    position = target + new Vector3(0.08f, height * 0.15f, Mathf.Max(2.1f, height * 1.55f));
                    break;
            }

            previewUtility.camera.transform.position = position;
            previewUtility.camera.transform.LookAt(target);
            previewUtility.camera.nearClipPlane = 0.01f;
            previewUtility.camera.farClipPlane = 100f;
        }

        private static Bounds CalculateAvatarBounds(GameObject root)
        {
            var renderers = root.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length == 0)
            {
                return new Bounds(root.transform.position + Vector3.up, new Vector3(1f, 2f, 1f));
            }

            var bounds = renderers[0].bounds;
            for (var index = 1; index < renderers.Length; index++)
            {
                bounds.Encapsulate(renderers[index].bounds);
            }
            return bounds;
        }

        private void DrawDetailsColumn()
        {
            EditorGUILayout.BeginVertical(GUILayout.Width(360f));
            GUILayout.Label("Review Details", EditorStyles.boldLabel);
            detailsScroll = EditorGUILayout.BeginScrollView(detailsScroll, GUILayout.ExpandHeight(true));

            if (selectedFeedEntry == null || selectedReviewEntry == null)
            {
                EditorGUILayout.HelpBox("No review candidate selected.", MessageType.Info);
                EditorGUILayout.EndScrollView();
                EditorGUILayout.EndVertical();
                return;
            }

            EditorGUILayout.LabelField("Candidate", selectedFeedEntry.displayName, EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Motion Id", selectedReviewEntry.motionId);
            EditorGUILayout.LabelField("Style", $"{selectedReviewEntry.targetStyleFamily} / {selectedReviewEntry.targetStyleSubstyle}");
            EditorGUILayout.LabelField("Meta Action", selectedReviewEntry.expectedMetaAction);
            EditorGUILayout.LabelField("Transition", selectedReviewEntry.transitionProfile);
            EditorGUILayout.LabelField("Candidate FBX", selectedReviewEntry.candidateFbxRelPath);
            if (!string.IsNullOrEmpty(selectedReviewEntry.sourceVideoRelPath))
            {
                EditorGUILayout.LabelField("Source Video", selectedReviewEntry.sourceVideoRelPath);
            }

            GUILayout.Space(8f);
            GUILayout.Label("Static Metrics", EditorStyles.boldLabel);
            DrawMetric("Frames", selectedReviewEntry.metricsSummary.frameCount.ToString());
            DrawMetric("Duration", $"{selectedReviewEntry.metricsSummary.durationSec:0.00}s");
            DrawMetric("FPS", $"{selectedReviewEntry.metricsSummary.fps:0.0}");
            DrawMetric("Mixamo Guess", selectedReviewEntry.metricsSummary.mixamoCompatibleGuess ? "yes" : "no");
            DrawMetric("Loop Delta", $"{selectedReviewEntry.metricsSummary.loopDeltaScore:0.000}");
            DrawMetric("Root Delta", $"{selectedReviewEntry.metricsSummary.rootTranslationDeltaMeters:0.000}m");
            DrawMetric("Left Foot", $"{selectedReviewEntry.metricsSummary.leftFootDeltaMeters:0.000}m");
            DrawMetric("Right Foot", $"{selectedReviewEntry.metricsSummary.rightFootDeltaMeters:0.000}m");

            GUILayout.Space(8f);
            GUILayout.Label("Decision", EditorStyles.boldLabel);
            selectedReviewEntry.decisionStatus = PopupString("Decision", selectedReviewEntry.decisionStatus, decisionOptions);
            selectedReviewEntry.reviewStatus = PopupString("Review Status", selectedReviewEntry.reviewStatus, reviewStatusOptions);
            selectedReviewEntry.loopSeam = PopupString("Loop Seam", selectedReviewEntry.loopSeam, loopSeamOptions);
            selectedReviewEntry.footStability = PopupString("Foot Stability", selectedReviewEntry.footStability, footStabilityOptions);
            selectedReviewEntry.styleClarity = PopupString("Style Clarity", selectedReviewEntry.styleClarity, styleClarityOptions);
            selectedReviewEntry.tempoTolerance = PopupString("Tempo Tolerance", selectedReviewEntry.tempoTolerance, tempoToleranceOptions);
            selectedReviewEntry.qualityTier = PopupString("Quality Tier", selectedReviewEntry.qualityTier, qualityTierOptions);
            selectedReviewEntry.licenseTier = PopupString("License Tier", selectedReviewEntry.licenseTier, licenseTierOptions);
            selectedReviewEntry.reviewer = EditorGUILayout.TextField("Reviewer", selectedReviewEntry.reviewer ?? string.Empty);
            selectedReviewEntry.notes = EditorGUILayout.TextField("Notes", selectedReviewEntry.notes ?? string.Empty);

            issueDraft = EditorGUILayout.TextArea(issueDraft, GUILayout.MinHeight(96f));
            EditorGUILayout.HelpBox("Enter one issue per line. Leave blank if the candidate is clean.", MessageType.None);

            GUILayout.Space(8f);
            GUILayout.Label("Motion Metadata", EditorStyles.boldLabel);
            selectedReviewEntry.displayName = EditorGUILayout.TextField("Display Name", selectedReviewEntry.displayName ?? string.Empty);
            selectedReviewEntry.energyBand = EditorGUILayout.TextField("Energy Band", selectedReviewEntry.energyBand ?? string.Empty);
            selectedReviewEntry.nativeBpm = EditorGUILayout.FloatField("Native BPM", selectedReviewEntry.nativeBpm);
            selectedReviewEntry.phraseBeats = EditorGUILayout.IntField("Phrase Beats", selectedReviewEntry.phraseBeats);
            selectedReviewEntry.varietyGroup = EditorGUILayout.TextField("Variety Group", selectedReviewEntry.varietyGroup ?? string.Empty);
            selectedReviewEntry.role = EditorGUILayout.TextField("Role", selectedReviewEntry.role ?? string.Empty);
            selectedReviewEntry.preferredSegments = EditStringList("Preferred Segments", selectedReviewEntry.preferredSegments);
            selectedReviewEntry.entryOffsetsBeats = EditIntList("Entry Offsets", selectedReviewEntry.entryOffsetsBeats);
            selectedReviewEntry.sliceBeatsOptions = EditIntList("Slice Beats", selectedReviewEntry.sliceBeatsOptions);

            EditorGUILayout.EndScrollView();
            EditorGUILayout.EndVertical();
        }

        private static void DrawMetric(string label, string value)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, GUILayout.Width(120f));
            EditorGUILayout.LabelField(value);
            EditorGUILayout.EndHorizontal();
        }

        private string PopupString(string label, string currentValue, string[] options)
        {
            var index = Array.IndexOf(options, currentValue ?? string.Empty);
            if (index < 0)
            {
                index = 0;
            }
            var nextIndex = EditorGUILayout.Popup(label, index, options);
            return options[Mathf.Clamp(nextIndex, 0, options.Length - 1)];
        }

        private static List<string> EditStringList(string label, List<string> values)
        {
            values ??= new List<string>();
            var raw = EditorGUILayout.TextField(label, string.Join(",", values));
            return raw.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(item => item.Trim()).Where(item => !string.IsNullOrEmpty(item)).ToList();
        }

        private static List<int> EditIntList(string label, List<int> values)
        {
            values ??= new List<int>();
            var raw = EditorGUILayout.TextField(label, string.Join(",", values));
            var result = new List<int>();
            foreach (var token in raw.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(token.Trim(), out var parsed))
                {
                    result.Add(parsed);
                }
            }
            return result;
        }

        private void SetDecision(string decision)
        {
            if (selectedReviewEntry == null)
            {
                return;
            }

            selectedReviewEntry.decisionStatus = decision;
            selectedReviewEntry.reviewStatus = decision == "pending" ? string.Empty : decision;
            if (decision == "approved" && selectedReviewEntry.qualityTier == "review_hold")
            {
                selectedReviewEntry.qualityTier = "production";
            }
            SaveCurrentReview();
        }

        private void SetSourceDecision(string decision)
        {
            if (selectedSourceJob == null)
            {
                return;
            }

            selectedSourceJob.sourceReview ??= new MotionBaseSourceReview();
            selectedSourceJob.sourceReview.decisionStatus = decision;
            SaveCurrentSourceReview();
        }

        private void SaveCurrentSourceReview()
        {
            if (selectedSourceJob == null)
            {
                return;
            }

            selectedSourceJob.sourceReview ??= new MotionBaseSourceReview();
            selectedSourceJob.sourceProvenance ??= new MotionBaseSourceProvenance();
            UpdateJobStageFromSourceReview(selectedSourceJob);
            selectedSourceJob.updatedAtUtc = DateTime.UtcNow.ToString("O");
            SaveJson(MotionBaseReviewProjectPaths.IntakeQueuePath, intakeQueueData);
            AssetDatabase.Refresh();
        }

        private void SaveCurrentReview()
        {
            if (selectedReviewEntry == null)
            {
                return;
            }

            selectedReviewEntry.issues = issueDraft.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(item => item.Trim())
                .Where(item => !string.IsNullOrEmpty(item))
                .Distinct(StringComparer.Ordinal)
                .ToList();
            selectedReviewEntry.savedAtUtc = DateTime.UtcNow.ToString("O");

            reviewByCandidateId[selectedReviewEntry.candidateId] = selectedReviewEntry;
            candidateReviewData.entries = reviewByCandidateId.Values.OrderBy(entry => entry.candidateId, StringComparer.Ordinal).ToList();
            SaveJson(MotionBaseReviewProjectPaths.CandidateReviewPath, candidateReviewData);

            UpdateJobStageFromReviews(selectedReviewEntry.jobId);
            SaveJson(MotionBaseReviewProjectPaths.IntakeQueuePath, intakeQueueData);
            AssetDatabase.Refresh();
        }

        private static void UpdateJobStageFromSourceReview(MotionBaseIntakeJob job)
        {
            if (job == null)
            {
                return;
            }

            var currentStage = job.stage ?? string.Empty;
            if (currentStage == "extracted" || currentStage == "retarget_ready" || currentStage == "sliced" || currentStage == "review_ready" || currentStage == "approved")
            {
                return;
            }

            var decision = job.sourceReview != null ? job.sourceReview.decisionStatus : "pending";
            switch (decision)
            {
                case "approved":
                    job.stage = "extraction_pending";
                    break;
                case "hold":
                    job.stage = "hold";
                    break;
                case "rejected":
                    job.stage = "rejected";
                    break;
                default:
                    job.stage = string.Equals(job.precheckSummary != null ? job.precheckSummary.status : string.Empty, "pass", StringComparison.Ordinal)
                        ? "precheck_passed"
                        : string.IsNullOrEmpty(currentStage) ? "discovered" : currentStage;
                    break;
            }
        }

        private void UpdateJobStageFromReviews(string jobId)
        {
            if (string.IsNullOrEmpty(jobId) || !jobById.TryGetValue(jobId, out var job))
            {
                return;
            }

            var reviews = candidateReviewData.entries.Where(entry => entry != null && entry.jobId == jobId).ToList();
            if (reviews.Any(entry => entry.decisionStatus == "approved"))
            {
                job.stage = "approved";
            }
            else if (reviews.Count > 0 && reviews.All(entry => entry.decisionStatus == "rejected"))
            {
                job.stage = "rejected";
            }
            else if (reviews.Any(entry => entry.decisionStatus == "hold" || entry.decisionStatus == "rejected"))
            {
                job.stage = "hold";
            }
            else
            {
                job.stage = "review_ready";
            }

            job.reviewFeedRelPath = "motion_base/review/review_feed.json";
            job.updatedAtUtc = DateTime.UtcNow.ToString("O");
        }
    }
}
