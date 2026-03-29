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

        private MotionBaseReviewFeedData reviewFeed = new MotionBaseReviewFeedData();
        private MotionBaseCandidateReviewData candidateReviewData = new MotionBaseCandidateReviewData();
        private MotionBaseIntakeQueueData intakeQueueData = new MotionBaseIntakeQueueData();
        private MotionBaseAssetRootsConfig assetRootsConfig = new MotionBaseAssetRootsConfig();

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

        private MotionBaseReviewFeedEntry selectedFeedEntry;
        private MotionBaseCandidateReviewEntry selectedReviewEntry;
        private int selectedIndex = -1;

        private string importedClipAssetPath;
        private AnimationClip importedClip;
        private string currentVideoAbsolutePath;
        private string issueDraft = string.Empty;
        private Vector2 queueScroll;
        private Vector2 detailsScroll;
        private bool isPlaying;
        private double lastEditorTime;
        private float previewTime;
        private float playbackSpeed = 1f;
        private CameraPreset cameraPreset = CameraPreset.FullBody;

        [MenuItem("Tools/Motion Base/Review Queue")]
        public static void Open()
        {
            var window = GetWindow<MotionBaseReviewWindow>("Motion Base Review");
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
            reviewFeed = LoadJson<MotionBaseReviewFeedData>(MotionBaseReviewProjectPaths.ReviewFeedPath) ?? new MotionBaseReviewFeedData();
            candidateReviewData = LoadJson<MotionBaseCandidateReviewData>(MotionBaseReviewProjectPaths.CandidateReviewPath) ?? new MotionBaseCandidateReviewData();
            intakeQueueData = LoadJson<MotionBaseIntakeQueueData>(MotionBaseReviewProjectPaths.IntakeQueuePath) ?? new MotionBaseIntakeQueueData();

            reviewFeed.entries ??= new List<MotionBaseReviewFeedEntry>();
            candidateReviewData.entries ??= new List<MotionBaseCandidateReviewEntry>();
            intakeQueueData.jobs ??= new List<MotionBaseIntakeJob>();

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
                jobById[job.jobId] = job;
            }

            if (reviewFeed.entries.Count == 0)
            {
                selectedIndex = -1;
                selectedFeedEntry = null;
                selectedReviewEntry = null;
                importedClip = null;
                return;
            }

            if (selectedIndex < 0 || selectedIndex >= reviewFeed.entries.Count)
            {
                selectedIndex = 0;
            }

            SelectEntry(selectedIndex);
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

            if (selectedReviewEntry == null || string.IsNullOrEmpty(selectedReviewEntry.sourceVideoRelPath))
            {
                return;
            }

            var videoRoot = assetRootsConfig.roots != null ? assetRootsConfig.roots.sourceVideo : null;
            if (string.IsNullOrEmpty(videoRoot))
            {
                return;
            }

            currentVideoAbsolutePath = Path.Combine(videoRoot, selectedReviewEntry.sourceVideoRelPath.Replace("/", Path.DirectorySeparatorChar.ToString()));
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

            if (GUILayout.Button("Save Review", EditorStyles.toolbarButton, GUILayout.Width(90f)))
            {
                SaveCurrentReview();
            }

            GUILayout.Space(8f);
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

            GUILayout.FlexibleSpace();
            GUILayout.Label($"Entries: {reviewFeed.entries.Count}", EditorStyles.miniLabel);
            EditorGUILayout.EndHorizontal();
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

            cameraPreset = (CameraPreset)EditorGUILayout.EnumPopup(cameraPreset, GUILayout.Width(120f));
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
                EditorGUI.HelpBox(rect, string.IsNullOrEmpty(currentVideoAbsolutePath) ? "No source video resolved for this candidate." : "Source video is loading or unavailable.", MessageType.Info);
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
