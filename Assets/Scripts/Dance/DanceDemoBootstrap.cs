using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UniGLTF;
using UniVRM10;

namespace DanceDemo
{
    public class DanceDemoBootstrap : MonoBehaviour
    {
        private enum PlaybackState
        {
            Idle,
            Playing,
            Paused,
            StoppedAtEnd,
        }

        private const string SongCatalogPath = "DanceData/song_catalog.json";
        private const string SongAnalysisPath = "DanceData/song_analysis.json";
        private const string MotionManifestPath = "DanceData/motion_manifest.json";
        private const string AvatarName = "willa";
        private const string AvatarStreamingPath = "Avatars/willa.vrm";
        private const string FallbackAudioResourcePath = "Dance/Audio/debug_click_track";
        private const string RuntimeCameraName = "DanceRuntimeCamera";
        private const string SceneMainCameraName = "Main Camera";
        private const string RuntimeLightName = "DanceRuntimeKeyLight";
        private const string RuntimeMarkerRootName = "DanceRuntimeDebugMarkers";
        private const string AvatarFallbackShaderName = "Standard";
        private const float AvatarFloorPadding = 0.03f;
        private const bool ShowAvatarDebugMarkers = false;
        private static readonly Vector3 FixedCameraPosition = new Vector3(0.15f, 1.56f, 5.94f);
        private static readonly Vector3 FixedCameraEuler = new Vector3(8f, 180f, 0f);
        private static readonly Vector3 FixedMainCameraPosition = new Vector3(0.15f, 2.91f, 5.63f);
        private static readonly Vector3 FixedMainCameraEuler = new Vector3(20f, 180f, 0f);

        private SongAnalysisData songData;
        private SongCatalogData songCatalog;
        private MotionManifestData manifest;
        private MusicConductor conductor;
        private ChoreographyEngine choreography;
        private CharacterDanceController characterController;
        private DanceDebugUI debugUi;
        private AudioSource audioSource;
        private Animator avatarAnimator;
        private readonly DanceDebugState debugState = new DanceDebugState();
        private readonly List<string> warnings = new List<string>();
        private readonly List<SongCatalogEntry> availableSongs = new List<SongCatalogEntry>();
        private Dictionary<string, AnimationClip> clipLookup = new Dictionary<string, AnimationClip>();
        private MotionClipEntry activeClipEntry;
        private DanceVariantCandidate activeVariant;
        private int activeVariantActivatedBeatIndex = -1;
        private DanceSelectionResult lastSelection;
        private bool pendingSegmentChange;
        private bool forceSwitchOnNextBeat;
        private bool isReloading;
        private Camera targetCamera;
        private float cameraReframeTimer;
        private const float CameraReframeDuration = 2.5f;
        private bool cameraFallbackApplied;
        private float cameraVisibilityProbeDelay = 0.15f;
        private AvatarVisibilityMetrics lastVisibilityMetrics;
        private Material avatarFallbackMaterial;
        private float avatarBaseRootY;
        private bool avatarBaseRootYInitialized;
        private PlaybackState playbackState = PlaybackState.Idle;
        private string lastVisibleClipName;
        private string selectedSongId;
        private string loadedSongId;
        private SongPlaybackProfile currentPlaybackProfile = SongPlaybackProfile.CreateDefault();
        private bool pendingStartAfterReload;

        private struct AvatarVisibilityMetrics
        {
            public bool InFrustum;
            public bool CenterVisible;
            public float ViewportCoverage;
            public float Distance;
            public string CameraName;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void CreateRuntimeBootstrap()
        {
            var existing = FindObjectOfType<DanceDemoBootstrap>();
            if (existing != null)
            {
                return;
            }

            var go = new GameObject("DanceDemoBootstrap");
            DontDestroyOnLoad(go);
            go.AddComponent<DanceDemoBootstrap>();
        }

        private IEnumerator Start()
        {
            yield return null;
            yield return LoadAndStart();
        }

        private IEnumerator LoadAndStart()
        {
            var preferredSongId = selectedSongId;
            isReloading = true;
            conductor?.StopPlayback(true);
            characterController?.StopAndResetPose();
            warnings.Clear();
            ClearPlaybackRuntimeState();
            lastVisibleClipName = null;
            loadedSongId = null;
            pendingStartAfterReload = false;
            playbackState = PlaybackState.Idle;
            debugState.RuntimeMessage = "Loading data...";

            EnsureComponents();
            yield return LoadJsonData(preferredSongId);

            var songErrors = DanceDataValidator.ValidateSong(songData);
            var manifestErrors = DanceDataValidator.ValidateManifest(manifest);

            if (songErrors.Count > 0 || manifestErrors.Count > 0)
            {
                var allErrors = songErrors.Concat(manifestErrors).ToArray();
                debugState.RuntimeMessage = string.Join(" | ", allErrors);
                Debug.LogError(debugState.RuntimeMessage);
                isReloading = false;
                yield break;
            }

            yield return EnsureAvatarAnimator();
            if (avatarAnimator == null)
            {
                debugState.RuntimeMessage = "Could not find the avatar Animator in the loaded scene.";
                Debug.LogError(debugState.RuntimeMessage);
                isReloading = false;
                yield break;
            }

            clipLookup = DanceAssetResolver.ResolveClips(manifest, warning => warnings.Add(warning));
            manifest = FilterManifestToResolvedClips(manifest, clipLookup, warning => warnings.Add(warning));
            if (clipLookup.Count == 0)
            {
                debugState.RuntimeMessage = "No playable AnimationClips resolved from Resources/Dance/Clips.";
                Debug.LogError(debugState.RuntimeMessage);
                isReloading = false;
                yield break;
            }

            var audioClip = DanceAssetResolver.ResolveAudioClip(songData, FallbackAudioResourcePath, warning => warnings.Add(warning));
            if (audioClip == null)
            {
                audioClip = DebugAudioClipFactory.CreateClickTrack(songData);
                warnings.Add("Using generated click track because no AudioClip was found in Resources.");
            }

            audioSource.clip = audioClip;
            Debug.Log(string.Format("Dance demo loaded audio '{0}' ({1:0.00}s) with {2} motion clips.", audioClip.name, audioClip.length, clipLookup.Count));

            characterController = avatarAnimator.GetComponent<CharacterDanceController>();
            if (characterController == null)
            {
                characterController = avatarAnimator.gameObject.AddComponent<CharacterDanceController>();
            }

            NormalizeAvatarStagePlacement();
            DiagnoseAndRepairAvatarRenderers();
            CreateAvatarDebugMarkers();
            ConfigureSceneMainCamera();
            EnsureRuntimeLighting();
            FrameMainCameraToAvatar();
            conductor.Initialize(songData, audioSource, currentPlaybackProfile);
            choreography = new ChoreographyEngine(manifest);

            conductor.BeatChanged -= HandleBeatChanged;
            conductor.BeatChanged += HandleBeatChanged;
            conductor.SegmentChanged -= HandleSegmentChanged;
            conductor.SegmentChanged += HandleSegmentChanged;
            conductor.PlaybackCompleted -= HandlePlaybackCompleted;
            conductor.PlaybackCompleted += HandlePlaybackCompleted;

            debugUi?.SetSongOptions(availableSongs, selectedSongId);
            EnterIdleForCurrentSong(warnings.Count > 0 ? string.Join(" | ", warnings.ToArray()) : "Ready. Press Start.");
            isReloading = false;
        }

        private void EnsureComponents()
        {
            audioSource = GetComponent<AudioSource>();
            if (audioSource == null)
            {
                audioSource = gameObject.AddComponent<AudioSource>();
            }

            audioSource.playOnAwake = false;
            audioSource.loop = false;

            conductor = GetComponent<MusicConductor>();
            if (conductor == null)
            {
                conductor = gameObject.AddComponent<MusicConductor>();
            }

            debugUi = GetComponent<DanceDebugUI>();
            if (debugUi == null)
            {
                debugUi = gameObject.AddComponent<DanceDebugUI>();
                debugUi.Initialize();
                debugUi.StartPlaybackRequested += HandleStartPlaybackRequested;
                debugUi.PausePlaybackRequested += HandlePausePlaybackRequested;
                debugUi.SongSelectionChanged += HandleSongSelectionChanged;
                debugUi.ForceSwitchRequested += HandleForceSwitchRequested;
                debugUi.ReloadDataRequested += HandleReloadDataRequested;
                debugUi.QuitRequested += HandleQuitRequested;
            }
        }

        private IEnumerator LoadJsonData(string preferredSongId)
        {
            songCatalog = null;
            manifest = null;
            songData = null;
            availableSongs.Clear();
            string catalogError = null;
            string manifestError = null;

            if (CatalogExists())
            {
                yield return DanceDataLoader.LoadJson<SongCatalogData>(
                    SongCatalogPath,
                    data => songCatalog = data,
                    error => catalogError = error);

                if (!string.IsNullOrEmpty(catalogError))
                {
                    warnings.Add(catalogError);
                }
            }

            yield return DanceDataLoader.LoadJson<MotionManifestData>(
                MotionManifestPath,
                data => manifest = data,
                error => manifestError = error);

            if (!string.IsNullOrEmpty(manifestError))
            {
                warnings.Add(manifestError);
            }

            yield return LoadSelectedSongData(preferredSongId);
        }

        private IEnumerator LoadSelectedSongData(string preferredSongId)
        {
            if (songCatalog != null)
            {
                foreach (var error in DanceDataValidator.ValidateSongCatalog(songCatalog))
                {
                    warnings.Add(error);
                }

                availableSongs.AddRange(FilterAvailableSongs(songCatalog));
            }

            if (availableSongs.Count > 0)
            {
                selectedSongId = ResolvePreferredSongId(preferredSongId);
                var selectedEntry = GetSelectedSongEntry();
                if (selectedEntry != null)
                {
                    SongAnalysisData loadedSong = null;
                    string songError = null;
                    yield return DanceDataLoader.LoadJson<SongAnalysisData>(
                        selectedEntry.analysisPath,
                        data => loadedSong = data,
                        error => songError = error);

                    if (!string.IsNullOrEmpty(songError))
                    {
                        warnings.Add(songError);
                    }

                    if (loadedSong != null)
                    {
                        loadedSong.songId = string.IsNullOrEmpty(selectedEntry.songId) ? loadedSong.songId : selectedEntry.songId;
                        loadedSong.displayName = string.IsNullOrEmpty(selectedEntry.displayName) ? loadedSong.displayName : selectedEntry.displayName;
                        loadedSong.audioResourcePath = string.IsNullOrEmpty(selectedEntry.audioResourcePath) ? loadedSong.audioResourcePath : selectedEntry.audioResourcePath;
                        songData = loadedSong;
                        loadedSongId = selectedSongId;
                        currentPlaybackProfile = selectedEntry.EffectivePlaybackProfile;
                        yield break;
                    }
                }

                warnings.Add("Selected catalog song could not be loaded. Falling back to legacy song_analysis.json.");
            }

            yield return DanceDataLoader.LoadJson<SongAnalysisData>(
                SongAnalysisPath,
                data => songData = data,
                error => warnings.Add(error));

            if (songData != null)
            {
                var fallbackEntry = new SongCatalogEntry
                {
                    songId = songData.songId,
                    displayName = songData.displayName,
                    analysisPath = SongAnalysisPath,
                    audioResourcePath = songData.audioResourcePath,
                };
                availableSongs.Clear();
                availableSongs.Add(fallbackEntry);
                selectedSongId = fallbackEntry.songId;
                loadedSongId = fallbackEntry.songId;
                currentPlaybackProfile = SongPlaybackProfile.CreateDefault();
            }
        }

        private Animator ResolveAvatarAnimator()
        {
            var avatarRoot = GameObject.Find(AvatarName);
            if (avatarRoot != null)
            {
                var namedAnimator = SelectBestAnimator(avatarRoot.GetComponentsInChildren<Animator>(true));
                if (namedAnimator != null)
                {
                    return namedAnimator;
                }
            }

            var animators = FindObjectsOfType<Animator>(true);
            var willaAnimators = animators.Where(candidate =>
            {
                return candidate != null && candidate.transform.root.name.ToLowerInvariant().Contains("willa");
            }).ToArray();

            var bestWillaAnimator = SelectBestAnimator(willaAnimators);
            if (bestWillaAnimator != null)
            {
                return bestWillaAnimator;
            }

            return SelectBestAnimator(animators);
        }

        private IEnumerator EnsureAvatarAnimator()
        {
            avatarAnimator = ResolveAvatarAnimator();
            if (IsUsableAvatarAnimator(avatarAnimator))
            {
                yield break;
            }

            if (avatarAnimator != null)
            {
                warnings.Add("Scene avatar reference was invalid. Loading willa.vrm from StreamingAssets instead.");
                Debug.LogWarning(string.Format(
                    "Avatar animator '{0}' under root '{1}' was invalid. Falling back to runtime VRM loading.",
                    avatarAnimator.name,
                    avatarAnimator.transform.root.name));
            }

            var avatarPath = Path.Combine(Application.streamingAssetsPath, AvatarStreamingPath);
            if (!File.Exists(avatarPath))
            {
                Debug.LogError("Runtime avatar file was not found: " + avatarPath);
                avatarAnimator = ResolveAvatarAnimator();
                yield break;
            }

            var loadTask = Vrm10.LoadPathAsync(
                avatarPath,
                canLoadVrm0X: true,
                showMeshes: true,
                awaitCaller: new ImmediateCaller(),
                materialGenerator: new BuiltInVrm10MaterialDescriptorGenerator());
            while (!loadTask.IsCompleted)
            {
                yield return null;
            }

            if (loadTask.IsFaulted)
            {
                Debug.LogException(loadTask.Exception);
                avatarAnimator = ResolveAvatarAnimator();
                yield break;
            }

            var instance = loadTask.Result;
            if (instance != null)
            {
                instance.name = AvatarName;
                instance.transform.position = Vector3.zero;
                instance.transform.rotation = Quaternion.identity;
                avatarAnimator = instance.GetComponent<Animator>() ?? instance.GetComponentInChildren<Animator>(true);
                Debug.Log("Loaded runtime avatar from " + avatarPath);
            }
        }

        private static Animator SelectBestAnimator(IEnumerable<Animator> animators)
        {
            Animator best = null;
            var bestRendererCount = -1;

            foreach (var candidate in animators)
            {
                if (candidate == null)
                {
                    continue;
                }

                var rendererCount = GetAvatarRenderers(candidate.transform.root.gameObject).Length;
                if (rendererCount > bestRendererCount)
                {
                    best = candidate;
                    bestRendererCount = rendererCount;
                }
            }

            return best;
        }

        private static bool IsUsableAvatarAnimator(Animator animator)
        {
            if (animator == null)
            {
                return false;
            }

            var root = animator.transform.root;
            if (root == null)
            {
                return false;
            }

            if (root.name.Contains("Missing Prefab"))
            {
                return false;
            }

            var renderers = GetAvatarRenderers(root.gameObject);
            if (renderers.Length == 0)
            {
                return false;
            }

            if (renderers.Length == 1 && renderers[0] != null && renderers[0].name == "Plane")
            {
                return false;
            }

            return true;
        }

        private void FrameMainCameraToAvatar()
        {
            ApplyAdaptiveFallbackCameraFrame();
            cameraFallbackApplied = true;
            cameraVisibilityProbeDelay = 0f;
            cameraReframeTimer = 0f;
        }

        private void ApplyCameraFrameToAvatar()
        {
            targetCamera = ResolveCamera();
            if (targetCamera == null || avatarAnimator == null)
            {
                return;
            }

            targetCamera.fieldOfView = 60f;
            targetCamera.cullingMask = ~0;
            targetCamera.transform.position = FixedCameraPosition;
            targetCamera.transform.rotation = Quaternion.Euler(FixedCameraEuler);
            targetCamera.transform.localScale = Vector3.one;
            targetCamera.nearClipPlane = 0.3f;
            targetCamera.farClipPlane = 1000f;
        }

        private void ApplyAdaptiveFallbackCameraFrame()
        {
            targetCamera = ResolveCamera();
            if (targetCamera == null || avatarAnimator == null)
            {
                return;
            }

            var bounds = ResolveAvatarBounds(avatarAnimator.transform.root.gameObject);
            var focusPoint = new Vector3(
                bounds.center.x,
                Mathf.Lerp(bounds.min.y, bounds.max.y, 0.50f),
                bounds.center.z);
            var avatarHeight = Mathf.Max(1.4f, bounds.size.y);
            var avatarWidth = Mathf.Max(0.9f, bounds.size.x);
            var desiredScreenFraction = 0.54f;
            var fallbackFov = 40f;
            var verticalDistance = avatarHeight / (2f * Mathf.Tan(fallbackFov * 0.5f * Mathf.Deg2Rad) * desiredScreenFraction);
            var horizontalFov = Camera.VerticalToHorizontalFieldOfView(fallbackFov, Mathf.Max(1f, targetCamera.aspect));
            var horizontalDistance = avatarWidth / (2f * Mathf.Tan(horizontalFov * 0.5f * Mathf.Deg2Rad) * 0.34f);
            var distance = Mathf.Max(3.8f, verticalDistance, horizontalDistance);
            var fallbackPosition = focusPoint + new Vector3(0.10f, avatarHeight * 0.18f, distance);

            targetCamera.fieldOfView = fallbackFov;
            targetCamera.transform.position = fallbackPosition;
            targetCamera.transform.rotation = Quaternion.LookRotation(focusPoint - fallbackPosition, Vector3.up);
            targetCamera.transform.localScale = Vector3.one;
            targetCamera.nearClipPlane = 0.2f;
            targetCamera.farClipPlane = 1000f;
            cameraFallbackApplied = true;
            Debug.Log(string.Format("Applied fallback runtime camera at {0} looking at {1} for avatar height {2:0.00}.", fallbackPosition, focusPoint, avatarHeight));
        }

        private void ConfigureSceneMainCamera()
        {
            var sceneMainCamera = GameObject.Find(SceneMainCameraName);
            if (sceneMainCamera == null)
            {
                return;
            }

            sceneMainCamera.transform.position = FixedMainCameraPosition;
            sceneMainCamera.transform.rotation = Quaternion.Euler(FixedMainCameraEuler);
            sceneMainCamera.transform.localScale = Vector3.one;
        }

        private void NormalizeAvatarStagePlacement()
        {
            if (avatarAnimator == null)
            {
                return;
            }

            var root = avatarAnimator.transform.root;
            var bounds = ResolveAvatarBounds(root.gameObject);
            var groundAnchorY = ResolveAvatarGroundAnchorY(root, bounds);
            root.position = new Vector3(0f, root.position.y - groundAnchorY, 0f);
            root.rotation = Quaternion.Euler(0f, 0f, 0f);
            var correctedBounds = ResolveAvatarBounds(root.gameObject);
            if (correctedBounds.min.y < AvatarFloorPadding)
            {
                root.position += Vector3.up * (AvatarFloorPadding - correctedBounds.min.y);
                correctedBounds = ResolveAvatarBounds(root.gameObject);
            }
            avatarBaseRootY = root.position.y;
            avatarBaseRootYInitialized = true;
            Debug.Log(string.Format(
                "Normalized avatar root to {0} with bounds center {1}, ground anchor {2:0.###}, post minY {3:0.###}.",
                root.position,
                correctedBounds.center,
                groundAnchorY,
                correctedBounds.min.y));
        }

        private float ResolveAvatarGroundAnchorY(Transform root, Bounds bounds)
        {
            if (avatarAnimator == null || !avatarAnimator.isHuman)
            {
                return bounds.min.y;
            }

            var candidateBones = new[]
            {
                HumanBodyBones.LeftToes,
                HumanBodyBones.RightToes,
                HumanBodyBones.LeftFoot,
                HumanBodyBones.RightFoot,
                HumanBodyBones.LeftLowerLeg,
                HumanBodyBones.RightLowerLeg,
            };

            var foundBone = false;
            var minBoneY = float.MaxValue;
            foreach (var bone in candidateBones)
            {
                var boneTransform = avatarAnimator.GetBoneTransform(bone);
                if (boneTransform == null || !boneTransform.IsChildOf(root))
                {
                    continue;
                }

                foundBone = true;
                minBoneY = Mathf.Min(minBoneY, boneTransform.position.y);
            }

            if (!foundBone)
            {
                return bounds.min.y;
            }

            // Keep the soles slightly above the floor to avoid the mesh clipping into the plane.
            return minBoneY - 0.015f;
        }

        private void MaintainAvatarAboveFloor()
        {
            if (avatarAnimator == null)
            {
                return;
            }

            var root = avatarAnimator.transform.root;
            if (avatarBaseRootYInitialized)
            {
                root.position = new Vector3(0f, avatarBaseRootY, 0f);
            }

            var bounds = ResolveAvatarBounds(root.gameObject);
            var groundAnchorY = ResolveAvatarGroundAnchorY(root, bounds);
            var floorReferenceY = Mathf.Min(bounds.min.y, groundAnchorY);
            if (floorReferenceY >= AvatarFloorPadding)
            {
                return;
            }

            root.position += Vector3.up * (AvatarFloorPadding - floorReferenceY);
        }

        private void DiagnoseAndRepairAvatarRenderers()
        {
            if (avatarAnimator == null)
            {
                return;
            }

            var avatarRoot = avatarAnimator.transform.root.gameObject;
            var renderers = GetAvatarRenderers(avatarRoot);
            if (renderers.Length == 0)
            {
                Debug.LogWarning(string.Format(
                    "Avatar renderer diagnostics: no renderers found under root '{0}' using animator '{1}'. Child count: {2}.",
                    avatarRoot.name,
                    avatarAnimator.name,
                    avatarRoot.GetComponentsInChildren<Transform>(true).Length));
                return;
            }

            var fallbackMaterial = GetOrCreateAvatarFallbackMaterial();
            Debug.Log(string.Format(
                "Avatar renderer diagnostics: root '{0}', animator '{1}', renderer count {2}.",
                avatarRoot.name,
                avatarAnimator.name,
                renderers.Length));

            foreach (var renderer in renderers)
            {
                if (renderer == null)
                {
                    continue;
                }

                renderer.enabled = true;
                renderer.forceRenderingOff = false;
                renderer.receiveShadows = false;
                renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;

                var skinnedMeshRenderer = renderer as SkinnedMeshRenderer;
                if (skinnedMeshRenderer != null)
                {
                    skinnedMeshRenderer.updateWhenOffscreen = true;
                }

                var materials = renderer.sharedMaterials;
                if (materials == null || materials.Length == 0)
                {
                    if (fallbackMaterial != null)
                    {
                        renderer.sharedMaterial = fallbackMaterial;
                    }

                    Debug.LogWarning(string.Format("Avatar renderer fallback: '{0}' had no materials.", renderer.name));
                    continue;
                }

                var repaired = false;
                var repairedMaterials = new Material[materials.Length];
                for (var i = 0; i < materials.Length; i++)
                {
                    repairedMaterials[i] = CreateVisibleMaterial(materials[i], fallbackMaterial, renderer.name, i, ref repaired);
                }

                if (repaired)
                {
                    renderer.sharedMaterials = repairedMaterials;
                }
            }
        }

        private void CreateAvatarDebugMarkers()
        {
            var existingRoot = GameObject.Find(RuntimeMarkerRootName);
            if (existingRoot != null)
            {
                Destroy(existingRoot);
            }

            if (avatarAnimator == null || !ShowAvatarDebugMarkers)
            {
                return;
            }

            var bounds = ResolveAvatarBounds(avatarAnimator.transform.root.gameObject);
            var markerRoot = new GameObject(RuntimeMarkerRootName);

            CreateMarker(markerRoot.transform, "AvatarCenterMarker", PrimitiveType.Sphere, bounds.center, new Vector3(0.16f, 0.16f, 0.16f), Color.red);
            CreateMarker(
                markerRoot.transform,
                "AvatarFeetMarker",
                PrimitiveType.Cube,
                new Vector3(bounds.center.x, bounds.min.y + 0.05f, bounds.center.z),
                new Vector3(0.18f, 0.1f, 0.18f),
                Color.green);

            var renderers = GetAvatarRenderers(avatarAnimator.transform.root.gameObject);
            Debug.Log(string.Format("Avatar debug markers created. Renderer count: {0}. Bounds center: {1}, min: {2}, max: {3}.", renderers.Length, bounds.center, bounds.min, bounds.max));
        }

        private static void CreateMarker(Transform parent, string name, PrimitiveType primitiveType, Vector3 position, Vector3 scale, Color color)
        {
            var marker = GameObject.CreatePrimitive(primitiveType);
            marker.name = name;
            marker.transform.SetParent(parent, false);
            marker.transform.position = position;
            marker.transform.localScale = scale;

            var collider = marker.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }

            var renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
            {
                var material = new Material(Shader.Find("Standard"));
                material.color = color;
                material.EnableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", color * 1.4f);
                renderer.material = material;
                renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
        }

        private void EnsureRuntimeLighting()
        {
            var existingLight = GameObject.Find(RuntimeLightName);
            Light lightComponent;
            if (existingLight != null)
            {
                lightComponent = existingLight.GetComponent<Light>();
            }
            else
            {
                var lightGo = new GameObject(RuntimeLightName, typeof(Light));
                lightComponent = lightGo.GetComponent<Light>();
            }

            if (lightComponent == null)
            {
                return;
            }

            lightComponent.type = LightType.Directional;
            lightComponent.intensity = 0.38f;
            lightComponent.color = new Color(0.98f, 0.96f, 0.93f, 1f);
            lightComponent.shadows = LightShadows.Soft;
            lightComponent.transform.rotation = Quaternion.Euler(28f, -38f, 0f);
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.18f, 0.2f, 0.24f, 1f);
        }

        private Camera ResolveCamera()
        {
            var existingRuntimeCamera = GameObject.Find(RuntimeCameraName);
            if (existingRuntimeCamera != null)
            {
                var runtimeCamera = existingRuntimeCamera.GetComponent<Camera>();
                if (runtimeCamera != null)
                {
                    EnableExclusiveCamera(runtimeCamera);
                    return runtimeCamera;
                }
            }

            var runtimeCameraGo = new GameObject(RuntimeCameraName, typeof(Camera), typeof(AudioListener));
            var createdCamera = runtimeCameraGo.GetComponent<Camera>();
            createdCamera.tag = "MainCamera";
            createdCamera.clearFlags = CameraClearFlags.Skybox;
            createdCamera.depth = 100f;
            EnableExclusiveCamera(createdCamera);
            return createdCamera;
        }

        private void EnableExclusiveCamera(Camera activeCamera)
        {
            var allCameras = FindObjectsOfType<Camera>(true);
            foreach (var candidate in allCameras)
            {
                if (candidate == null)
                {
                    continue;
                }

                candidate.enabled = candidate == activeCamera;
                if (candidate != activeCamera && candidate.CompareTag("MainCamera"))
                {
                    candidate.tag = "Untagged";
                }
            }

            activeCamera.enabled = true;
            if (!activeCamera.CompareTag("MainCamera"))
            {
                activeCamera.tag = "MainCamera";
            }
        }

        private static Bounds ResolveAvatarBounds(GameObject avatarRoot)
        {
            var renderers = GetAvatarRenderers(avatarRoot);
            if (renderers.Length == 0)
            {
                return new Bounds(avatarRoot.transform.position + Vector3.up, new Vector3(1f, 2f, 1f));
            }

            var bounds = renderers[0].bounds;
            for (var i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }

            return bounds;
        }

        private static Renderer[] GetAvatarRenderers(GameObject avatarRoot)
        {
            var renderers = avatarRoot.GetComponentsInChildren<Renderer>(true);
            var filtered = new List<Renderer>(renderers.Length);
            foreach (var renderer in renderers)
            {
                if (renderer == null)
                {
                    continue;
                }

                if (string.Equals(renderer.name, "Plane", System.StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                filtered.Add(renderer);
            }

            return filtered.ToArray();
        }

        private static MotionManifestData FilterManifestToResolvedClips(MotionManifestData sourceManifest, Dictionary<string, AnimationClip> resolvedClips, System.Action<string> onWarning)
        {
            if (sourceManifest == null || sourceManifest.clips == null)
            {
                return sourceManifest;
            }

            var filtered = new List<MotionClipEntry>(sourceManifest.clips.Length);
            foreach (var clip in sourceManifest.clips)
            {
                if (clip == null || string.IsNullOrEmpty(clip.clipId))
                {
                    continue;
                }

                if (resolvedClips != null && resolvedClips.ContainsKey(clip.clipId))
                {
                    filtered.Add(clip);
                }
                else
                {
                    onWarning?.Invoke("Removing unresolved or unsupported clip from runtime manifest: " + clip.clipId);
                }
            }

            return new MotionManifestData
            {
                libraryId = sourceManifest.libraryId,
                defaultPhraseBeats = sourceManifest.defaultPhraseBeats,
                clips = filtered.ToArray(),
            };
        }

        private bool CatalogExists()
        {
            var absolutePath = Path.Combine(Application.streamingAssetsPath, SongCatalogPath);
            return File.Exists(absolutePath);
        }

        private List<SongCatalogEntry> FilterAvailableSongs(SongCatalogData catalog)
        {
            var validSongs = new List<SongCatalogEntry>();
            if (catalog == null || catalog.songs == null)
            {
                return validSongs;
            }

            var seenIds = new HashSet<string>();
            foreach (var song in catalog.songs)
            {
                if (song == null || string.IsNullOrEmpty(song.songId))
                {
                    continue;
                }

                if (!seenIds.Add(song.songId))
                {
                    continue;
                }

                if (string.IsNullOrEmpty(song.analysisPath) || string.IsNullOrEmpty(song.audioResourcePath))
                {
                    continue;
                }

                var analysisAbsolutePath = Path.Combine(Application.streamingAssetsPath, song.analysisPath);
                if (!File.Exists(analysisAbsolutePath))
                {
                    warnings.Add("Skipping song with missing analysis file: " + song.songId);
                    continue;
                }

                validSongs.Add(song);
            }

            return validSongs;
        }

        private string ResolvePreferredSongId(string preferredSongId)
        {
            if (!string.IsNullOrEmpty(preferredSongId) && availableSongs.Any(song => string.Equals(song.songId, preferredSongId, System.StringComparison.Ordinal)))
            {
                return preferredSongId;
            }

            if (songCatalog != null && !string.IsNullOrEmpty(songCatalog.defaultSongId))
            {
                var defaultEntry = availableSongs.FirstOrDefault(song => string.Equals(song.songId, songCatalog.defaultSongId, System.StringComparison.Ordinal));
                if (defaultEntry != null)
                {
                    return defaultEntry.songId;
                }
            }

            return availableSongs.Count > 0 ? availableSongs[0].songId : null;
        }

        private SongCatalogEntry GetSelectedSongEntry()
        {
            return availableSongs.FirstOrDefault(song => string.Equals(song.songId, selectedSongId, System.StringComparison.Ordinal));
        }

        private IEnumerator LoadSongEntryForSelection(SongCatalogEntry entry)
        {
            if (entry == null)
            {
                yield break;
            }

            SongAnalysisData loadedSong = null;
            string songError = null;
            yield return DanceDataLoader.LoadJson<SongAnalysisData>(
                entry.analysisPath,
                data => loadedSong = data,
                error => songError = error);

            if (!string.IsNullOrEmpty(songError))
            {
                warnings.Add(songError);
                yield break;
            }

            if (loadedSong == null)
            {
                warnings.Add("Could not parse song analysis for " + entry.songId);
                yield break;
            }

            loadedSong.songId = string.IsNullOrEmpty(entry.songId) ? loadedSong.songId : entry.songId;
            loadedSong.displayName = string.IsNullOrEmpty(entry.displayName) ? loadedSong.displayName : entry.displayName;
            loadedSong.audioResourcePath = string.IsNullOrEmpty(entry.audioResourcePath) ? loadedSong.audioResourcePath : entry.audioResourcePath;
            songData = loadedSong;
            loadedSongId = entry.songId;
            currentPlaybackProfile = entry.EffectivePlaybackProfile;
        }

        private void EnterIdleForCurrentSong(string statusMessage)
        {
            conductor?.StopPlayback(true);
            characterController?.StopAndResetPose();
            ClearPlaybackRuntimeState();
            playbackState = PlaybackState.Idle;
            debugState.RuntimeMessage = statusMessage;
        }

        private void PrepareControllerForPlayback()
        {
            if (avatarAnimator == null)
            {
                return;
            }

            characterController = avatarAnimator.GetComponent<CharacterDanceController>();
            if (characterController == null)
            {
                characterController = avatarAnimator.gameObject.AddComponent<CharacterDanceController>();
            }

            characterController.StopAndResetPose();
            characterController.Initialize(avatarAnimator, clipLookup, 0.38f);
        }

        private float ResolveInitialDanceBpm()
        {
            if (songData == null)
            {
                return 120f;
            }

            var beatGrouping = Mathf.Max(1, currentPlaybackProfile != null ? currentPlaybackProfile.EffectiveBeatGrouping : 1);
            var tempoScale = currentPlaybackProfile != null ? currentPlaybackProfile.EffectiveTempoScale : 1f;
            return (songData.bpm / beatGrouping) * tempoScale;
        }

        private void StartPlaybackFromBeginning()
        {
            if (conductor == null || avatarAnimator == null || string.IsNullOrEmpty(loadedSongId) || !string.Equals(loadedSongId, selectedSongId, System.StringComparison.Ordinal))
            {
                Debug.LogWarning(string.Format(
                    "StartPlaybackFromBeginning aborted. conductor={0} avatarAnimator={1} loadedSongId={2} selectedSongId={3}",
                    conductor != null,
                    avatarAnimator != null,
                    string.IsNullOrEmpty(loadedSongId) ? "<null>" : loadedSongId,
                    string.IsNullOrEmpty(selectedSongId) ? "<null>" : selectedSongId));
                return;
            }

            ClearPlaybackRuntimeState();
            PrepareControllerForPlayback();
            conductor.PlayFromStart();
            ScheduleInitialClip(conductor.SongDspStartTime);
            playbackState = PlaybackState.Playing;
            Debug.Log("Playback started from beginning for song " + loadedSongId + ".");
        }

        private void ClearPlaybackRuntimeState()
        {
            activeClipEntry = null;
            activeVariant = null;
            activeVariantActivatedBeatIndex = -1;
            lastSelection = null;
            pendingSegmentChange = false;
            forceSwitchOnNextBeat = false;
        }

        private void ScheduleInitialClip(double startDspTime)
        {
            if (manifest == null || manifest.clips == null || manifest.clips.Length == 0)
            {
                Debug.LogWarning("ScheduleInitialClip skipped because manifest has no clips.");
                return;
            }

            var currentSegment = songData.GetSegmentForBeat(0);
            lastSelection = choreography.SelectNextClip(new DanceSelectionContext
            {
                SongId = songData != null ? songData.songId : "song",
                CurrentBeatIndex = 0,
                CurrentBarIndex = 0,
                CurrentSegment = currentSegment,
                CurrentClipId = null,
                CurrentVariantId = null,
                CurrentVarietyGroup = null,
                CurrentRole = null,
                SegmentChanged = true,
                Forced = false,
                LastActivatedBeatIndex = -1,
                PhraseIndex = 0,
                LocalBpm = songData != null ? songData.bpm : 120f,
                DanceLocalBpm = ResolveInitialDanceBpm(),
                GrooveStrength = songData != null ? songData.ResolveGrooveStrength(0f) : 0.5f,
                IsDownbeat = true,
                IsPerceivedDownbeat = true,
                AllowAccent = false,
                IsStrongBeatWindow = true,
                StrongBeatReason = "start",
                BeatsSinceLastSwitch = 0,
                PerceivedBeatIndex = 0,
                BeatGrouping = currentPlaybackProfile.EffectiveBeatGrouping,
                EnergyMode = currentPlaybackProfile.EffectiveEnergyMode,
            });

            if (lastSelection.SelectedVariant == null || lastSelection.SelectedClip == null)
            {
                Debug.LogWarning("ScheduleInitialClip failed to pick an initial variant.");
                return;
            }

            var clip = characterController.GetClip(lastSelection.SelectedClip);
            if (clip == null)
            {
                warnings.Add("Initial clip could not be resolved: " + lastSelection.SelectedClip.clipId);
                Debug.LogWarning("ScheduleInitialClip resolved a variant but clip lookup returned null for " + lastSelection.SelectedClip.clipId + ".");
                return;
            }

            characterController.QueueClip(lastSelection.SelectedVariant, clip, startDspTime);
            activeClipEntry = lastSelection.SelectedClip;
            activeVariant = lastSelection.SelectedVariant;
            activeVariantActivatedBeatIndex = 0;
            choreography.NotifyVariantActivated(activeVariant, 0);
            Debug.Log(string.Format(
                "Scheduled initial variant '{0}' clip '{1}' at dsp {2:0.000}.",
                activeVariant.variantId,
                activeClipEntry.clipId,
                startDspTime));
        }

        private void HandleBeatChanged(BeatEventInfo info)
        {
            if (isReloading)
            {
                return;
            }

            var currentSpeed = characterController != null ? characterController.CurrentPlaybackSpeed : 1f;
            var targetSpeed = characterController != null ? characterController.TargetPlaybackSpeed : 1f;
            var clampHits = characterController != null ? characterController.ConsecutiveClampHitCount : 0;
            var retimeStress = Mathf.Abs(targetSpeed - currentSpeed);
            var playingVariant = characterController != null ? characterController.ActiveVariant : activeVariant;
            var playingEntry = characterController != null ? characterController.ActiveEntry : activeClipEntry;
            var rhythm = conductor != null ? conductor.CurrentRhythm : default;
            var beatGrouping = Mathf.Max(1, rhythm.BeatGrouping);
            var perceivedBeatIndex = rhythm.PerceivedBeatIndex;
            var phraseIndex = manifest != null && manifest.defaultPhraseBeats > 0
                ? Mathf.Max(0, perceivedBeatIndex / manifest.defaultPhraseBeats)
                : Mathf.Max(0, perceivedBeatIndex / 8);
            var grooveStrength = conductor != null ? rhythm.GrooveStrength : (songData != null ? songData.ResolveGrooveStrength(info.SongTimeSec) : 0.5f);
            var isDownbeat = conductor != null ? rhythm.IsDownbeat : (info.BeatWindow != null
                ? info.BeatWindow.isDownbeat
                : info.BeatIndex % Mathf.Max(1, songData != null ? songData.beatsPerBar : 4) == 0);
            var isPerceivedDownbeat = conductor != null && rhythm.IsPerceivedDownbeat;
            var strongBeatWindow = conductor != null && rhythm.IsStrongBeatWindow;
            var strongBeatReason = conductor != null ? rhythm.StrongBeatReason : "none";
            var rawLocalBpm = conductor != null && rhythm.LocalBpm > 0.01f ? rhythm.LocalBpm : songData.bpm;
            var danceLocalBpm = conductor != null && rhythm.DanceLocalBpm > 0.01f ? rhythm.DanceLocalBpm : rawLocalBpm;
            var beatsSinceLastSwitch = activeVariantActivatedBeatIndex >= 0 ? Mathf.Max(0, (info.BeatIndex - activeVariantActivatedBeatIndex) / beatGrouping) : 0;
            var gentleMode = string.Equals(rhythm.EnergyMode, "gentle", System.StringComparison.OrdinalIgnoreCase);
            var allowAccent = pendingSegmentChange || (!gentleMode && strongBeatWindow && beatsSinceLastSwitch >= 6 && grooveStrength >= 0.72f);
            var context = new DanceSelectionContext
            {
                SongId = songData != null ? songData.songId : "song",
                CurrentBeatIndex = info.BeatIndex,
                CurrentBarIndex = info.BarIndex,
                CurrentSegment = info.ActiveSegment,
                CurrentClipId = playingEntry != null ? playingEntry.clipId : null,
                CurrentVariantId = playingVariant != null ? playingVariant.variantId : null,
                CurrentVarietyGroup = playingVariant != null ? playingVariant.EffectiveVarietyGroup : null,
                CurrentRole = playingVariant != null ? playingVariant.EffectiveRole : null,
                SegmentChanged = pendingSegmentChange,
                Forced = forceSwitchOnNextBeat,
                LastActivatedBeatIndex = activeVariantActivatedBeatIndex,
                PhraseIndex = phraseIndex,
                LocalBpm = rawLocalBpm,
                DanceLocalBpm = danceLocalBpm,
                CurrentSpeed = currentSpeed,
                TargetSpeed = targetSpeed,
                RetimeStress = retimeStress,
                ClampHitCount = clampHits,
                GrooveStrength = grooveStrength,
                IsDownbeat = isDownbeat,
                IsPerceivedDownbeat = isPerceivedDownbeat,
                AllowAccent = allowAccent,
                IsStrongBeatWindow = strongBeatWindow,
                StrongBeatReason = strongBeatReason,
                BeatsSinceLastSwitch = beatsSinceLastSwitch,
                PerceivedBeatIndex = perceivedBeatIndex,
                BeatGrouping = beatGrouping,
                EnergyMode = conductor != null ? rhythm.EnergyMode : currentPlaybackProfile.EffectiveEnergyMode,
            };
            var shouldEvaluate = choreography.ShouldForceReselect(context, playingVariant, manifest.defaultPhraseBeats);
            if (shouldEvaluate)
            {
                var selection = choreography.SelectNextClip(context);

                lastSelection = selection;
                if (selection.SelectedVariant != null && selection.SelectedClip != null)
                {
                    var isSameVariant =
                        playingVariant != null &&
                        string.Equals(playingVariant.variantId, selection.SelectedVariant.variantId, System.StringComparison.Ordinal);
                    var isSameClip =
                        playingEntry != null &&
                        string.Equals(playingEntry.clipId, selection.SelectedClip.clipId, System.StringComparison.Ordinal);
                    if (isSameVariant || (isSameClip && !selection.SelectedVariant.IsAccent))
                    {
                        pendingSegmentChange = false;
                        forceSwitchOnNextBeat = false;
                        return;
                    }

                    var scheduledBeat = info.BeatIndex + beatGrouping;
                    var scheduledDsp = conductor.GetBeatDspTime(scheduledBeat);
                    var clip = characterController.GetClip(selection.SelectedClip);
                    if (clip != null)
                    {
                        characterController.QueueClip(selection.SelectedVariant, clip, scheduledDsp);
                        activeClipEntry = selection.SelectedClip;
                        activeVariant = selection.SelectedVariant;
                        activeVariantActivatedBeatIndex = scheduledBeat;
                        choreography.NotifyVariantActivated(selection.SelectedVariant, scheduledBeat);
                    }
                }
            }

            pendingSegmentChange = false;
            forceSwitchOnNextBeat = false;
        }

        private void HandleSegmentChanged(SegmentEventInfo info)
        {
            pendingSegmentChange = true;
        }

        private void HandleStartPlaybackRequested()
        {
            Debug.Log("UI start requested. Current playback state: " + playbackState);
            if (isReloading)
            {
                pendingStartAfterReload = true;
                debugState.RuntimeMessage = "Loading selected song. Start queued.";
                Debug.Log("Start request queued while reloading song data.");
                return;
            }

            if (conductor == null || characterController == null)
            {
                Debug.LogWarning("Start request ignored because conductor or controller is missing.");
                return;
            }

            if (!string.Equals(loadedSongId, selectedSongId, System.StringComparison.Ordinal))
            {
                StartCoroutine(SwitchSongSelectionCoroutine(selectedSongId, true));
                return;
            }

            switch (playbackState)
            {
                case PlaybackState.Paused:
                    characterController.ResumePlayback();
                    conductor.ResumePlayback();
                    playbackState = PlaybackState.Playing;
                    debugState.RuntimeMessage = "Playback resumed.";
                    break;
                case PlaybackState.Idle:
                case PlaybackState.StoppedAtEnd:
                    StartPlaybackFromBeginning();
                    debugState.RuntimeMessage = "Playback restarted from beginning.";
                    break;
                case PlaybackState.Playing:
                    debugState.RuntimeMessage = "Playback already running.";
                    break;
            }
        }

        private void HandlePausePlaybackRequested()
        {
            Debug.Log("UI pause requested. Current playback state: " + playbackState);
            if (isReloading || conductor == null || characterController == null)
            {
                return;
            }

            switch (playbackState)
            {
                case PlaybackState.Playing:
                    conductor.PausePlayback();
                    characterController.PausePlayback();
                    playbackState = PlaybackState.Paused;
                    debugState.RuntimeMessage = "Paused.";
                    break;
                case PlaybackState.Paused:
                    debugState.RuntimeMessage = "Already paused.";
                    break;
                default:
                    debugState.RuntimeMessage = "Playback is not running.";
                    break;
            }
        }

        private void HandleSongSelectionChanged(string songId)
        {
            Debug.Log("Bootstrap received song selection: " + songId + ", current selected: " + selectedSongId + ", state: " + playbackState);
            if (isReloading || string.IsNullOrEmpty(songId) || string.Equals(songId, selectedSongId, System.StringComparison.Ordinal))
            {
                return;
            }

            StartCoroutine(SwitchSongSelectionCoroutine(songId, false));
        }

        private IEnumerator SwitchSongSelectionCoroutine(string songId, bool autoStartAfterLoad)
        {
            var selectedEntry = availableSongs.FirstOrDefault(song => string.Equals(song.songId, songId, System.StringComparison.Ordinal));
            if (selectedEntry == null)
            {
                yield break;
            }

            isReloading = true;
            warnings.Clear();
            selectedSongId = songId;
            lastVisibleClipName = null;
            EnterIdleForCurrentSong("Loading selected song...");
            songData = null;
            loadedSongId = null;
            yield return LoadSongEntryForSelection(selectedEntry);

            if (songData == null)
            {
                debugState.RuntimeMessage = "Failed to load selected song.";
                debugUi?.SetSongOptions(availableSongs, selectedSongId);
                pendingStartAfterReload = false;
                isReloading = false;
                yield break;
            }

            var songErrors = DanceDataValidator.ValidateSong(songData);
            if (songErrors.Count > 0)
            {
                debugState.RuntimeMessage = string.Join(" | ", songErrors.ToArray());
                debugUi?.SetSongOptions(availableSongs, selectedSongId);
                pendingStartAfterReload = false;
                isReloading = false;
                yield break;
            }

            var audioClip = DanceAssetResolver.ResolveAudioClip(songData, FallbackAudioResourcePath, warning => warnings.Add(warning));
            if (audioClip == null)
            {
                audioClip = DebugAudioClipFactory.CreateClickTrack(songData);
                warnings.Add("Using generated click track because no AudioClip was found in Resources.");
            }

            audioSource.clip = audioClip;
            Debug.Log(string.Format("Dance demo selected audio '{0}' ({1:0.00}s).", audioClip.name, audioClip.length));
            conductor.Initialize(songData, audioSource, currentPlaybackProfile);
            debugUi?.SetSongOptions(availableSongs, selectedSongId);
            var shouldAutoStart = autoStartAfterLoad || pendingStartAfterReload;
            pendingStartAfterReload = false;

            if (shouldAutoStart)
            {
                isReloading = false;
                StartPlaybackFromBeginning();
                debugState.RuntimeMessage = "Playback restarted from beginning.";
            }
            else
            {
                EnterIdleForCurrentSong("Song changed. Press Start.");
                isReloading = false;
            }
        }

        private void HandlePlaybackCompleted()
        {
            if (isReloading || conductor == null)
            {
                return;
            }

            lastVisibleClipName = characterController != null && !string.IsNullOrEmpty(characterController.CurrentClipId)
                ? characterController.CurrentClipId
                : lastVisibleClipName;
            characterController?.StopAndResetPose();
            ClearPlaybackRuntimeState();
            playbackState = PlaybackState.StoppedAtEnd;
            debugState.RuntimeMessage = "Playback complete. Press Start.";
        }

        private void HandleForceSwitchRequested()
        {
            if (playbackState != PlaybackState.Playing)
            {
                return;
            }

            forceSwitchOnNextBeat = true;
        }

        private void HandleReloadDataRequested()
        {
            if (isReloading)
            {
                return;
            }

            StartCoroutine(LoadAndStart());
        }

        private void HandleQuitRequested()
        {
            Application.Quit();
        }

        private string ResolvePlaybackStateLabel()
        {
            switch (playbackState)
            {
                case PlaybackState.Playing:
                    return "playing";
                case PlaybackState.Paused:
                    return "paused";
                case PlaybackState.StoppedAtEnd:
                    return "stopped_at_end";
                default:
                    return "idle";
            }
        }

        private string ResolveSelectedSongDisplayName()
        {
            var selectedEntry = GetSelectedSongEntry();
            if (selectedEntry != null)
            {
                return selectedEntry.EffectiveDisplayName;
            }

            return songData != null && !string.IsNullOrEmpty(songData.displayName) ? songData.displayName : "n/a";
        }

        private string ResolveCompactCurrentClip()
        {
            if (!string.IsNullOrEmpty(debugState.CurrentClipName))
            {
                return debugState.CurrentClipName;
            }

            return string.IsNullOrEmpty(lastVisibleClipName) ? "n/a" : lastVisibleClipName;
        }

        private string ResolveCompactNextClip()
        {
            if (playbackState == PlaybackState.Playing)
            {
                return string.IsNullOrEmpty(debugState.PendingClipName) ? "Hold" : debugState.PendingClipName;
            }

            return string.Empty;
        }

        private string ResolveCompactReason()
        {
            switch (playbackState)
            {
                case PlaybackState.Paused:
                    return "Paused";
                case PlaybackState.StoppedAtEnd:
                    return "Playback complete";
                case PlaybackState.Idle:
                    return !string.IsNullOrEmpty(debugState.RuntimeMessage) ? SummarizeReason(debugState.RuntimeMessage, 54) : "Ready";
                default:
                    var raw = !string.IsNullOrEmpty(debugState.DiversityReason) ? debugState.DiversityReason : debugState.NextReason;
                    if (string.IsNullOrEmpty(raw))
                    {
                        return "Holding pattern";
                    }

                    return SummarizeReason(raw, 54);
            }
        }

        private string ResolveCompactStrongBeat()
        {
            if (playbackState != PlaybackState.Playing)
            {
                return "no";
            }

            if (!debugState.StrongBeatWindow)
            {
                return "no";
            }

            return string.IsNullOrEmpty(debugState.StrongBeatReason)
                ? "yes"
                : "yes (" + debugState.StrongBeatReason + ")";
        }

        private static string SummarizeReason(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value;
            }

            return value.Substring(0, Mathf.Max(0, maxLength - 1)).TrimEnd() + "…";
        }

        private void Update()
        {
            if (cameraReframeTimer > 0f)
            {
                cameraReframeTimer -= Time.deltaTime;
                cameraVisibilityProbeDelay -= Time.deltaTime;
                if (!cameraFallbackApplied && cameraVisibilityProbeDelay > 0f)
                {
                    ApplyCameraFrameToAvatar();
                }
            }

            if (playbackState == PlaybackState.Playing && conductor != null && characterController != null)
            {
                var rhythm = conductor.CurrentRhythm;
                var danceLocalBpm = rhythm.DanceLocalBpm > 0.01f ? rhythm.DanceLocalBpm : (rhythm.LocalBpm > 0.01f ? rhythm.LocalBpm : songData.bpm);
                characterController.UpdatePlaybackSpeed(danceLocalBpm, rhythm.GrooveStrength, rhythm.IsPerceivedDownbeat, rhythm.BeatPhase, rhythm.EnergyMode);
            }

            lastVisibilityMetrics = EvaluateAvatarVisibility();
            if (!cameraFallbackApplied && cameraReframeTimer > 0f && cameraVisibilityProbeDelay <= 0f)
            {
                var coverageTooSmall = lastVisibilityMetrics.ViewportCoverage < 0.035f;
                if (!lastVisibilityMetrics.InFrustum || !lastVisibilityMetrics.CenterVisible || coverageTooSmall)
                {
                    ApplyAdaptiveFallbackCameraFrame();
                    lastVisibilityMetrics = EvaluateAvatarVisibility();
                }
            }

            debugState.SongName = songData != null ? songData.displayName : "n/a";
            debugState.SelectedSongName = ResolveSelectedSongDisplayName();
            debugState.AvailableSongCount = availableSongs.Count;
            debugState.PlaybackHint = debugState.RuntimeMessage;
            debugState.IsBusy = isReloading;
            debugState.ActiveCameraName = lastVisibilityMetrics.CameraName;
            debugState.SongTimeSec = conductor != null ? conductor.CurrentSongTimeSec : 0f;
            debugState.DurationSec = songData != null ? songData.durationSec : 0f;
            debugState.Bpm = songData != null ? songData.bpm : 0f;
            debugState.LocalBpm = conductor != null && conductor.CurrentRhythm.LocalBpm > 0.01f
                ? conductor.CurrentRhythm.LocalBpm
                : (songData != null ? songData.bpm : 0f);
            debugState.DanceBpm = conductor != null && conductor.CurrentRhythm.DanceLocalBpm > 0.01f
                ? conductor.CurrentRhythm.DanceLocalBpm
                : ResolveInitialDanceBpm();
            debugState.CurrentSpeed = characterController != null ? characterController.CurrentPlaybackSpeed : 1f;
            debugState.TargetSpeed = characterController != null ? characterController.TargetPlaybackSpeed : 1f;
            debugState.GrooveStrength = conductor != null ? conductor.CurrentRhythm.GrooveStrength : 0f;
            debugState.StrongBeatWindow = conductor != null && conductor.CurrentRhythm.IsStrongBeatWindow;
            debugState.StrongBeatReason = conductor != null ? conductor.CurrentRhythm.StrongBeatReason : "none";
            debugState.BeatGrouping = conductor != null && conductor.CurrentRhythm.BeatGrouping > 0
                ? conductor.CurrentRhythm.BeatGrouping
                : currentPlaybackProfile.EffectiveBeatGrouping;
            debugState.EnergyMode = conductor != null && !string.IsNullOrEmpty(conductor.CurrentRhythm.EnergyMode)
                ? conductor.CurrentRhythm.EnergyMode
                : currentPlaybackProfile.EffectiveEnergyMode;
            debugState.AvatarInView = lastVisibilityMetrics.InFrustum;
            debugState.AvatarCenterVisible = lastVisibilityMetrics.CenterVisible;
            debugState.AvatarViewportCoverage = lastVisibilityMetrics.ViewportCoverage;
            debugState.AvatarDistance = lastVisibilityMetrics.Distance;
            if (targetCamera != null)
            {
                debugState.CameraPosition = targetCamera.transform.position;
                debugState.CameraRotationEuler = targetCamera.transform.eulerAngles;
                debugState.CameraFieldOfView = targetCamera.fieldOfView;
            }

            if (avatarAnimator != null)
            {
                var avatarRoot = avatarAnimator.transform.root;
                var avatarBounds = ResolveAvatarBounds(avatarRoot.gameObject);
                debugState.AvatarRootPosition = avatarRoot.position;
                debugState.AvatarBoundsCenter = avatarBounds.center;
                debugState.AvatarBoundsSize = avatarBounds.size;
            }

            debugState.BeatIndex = conductor != null ? conductor.CurrentBeatIndex : -1;
            debugState.BarIndex = conductor != null ? conductor.CurrentBarIndex : -1;
            debugState.SegmentLabel = conductor != null && conductor.CurrentSegment != null ? conductor.CurrentSegment.label : "n/a";
            debugState.CurrentClipName = characterController != null ? characterController.CurrentClipId : null;
            debugState.PendingClipName = characterController != null ? characterController.PendingClipId : null;
            if (!string.IsNullOrEmpty(debugState.CurrentClipName))
            {
                lastVisibleClipName = debugState.CurrentClipName;
            }
            debugState.CurrentVariantId = characterController != null ? characterController.CurrentVariantId : (activeVariant != null ? activeVariant.variantId : null);
            debugState.PendingVariantId = characterController != null ? characterController.PendingVariantId : null;
            debugState.NextReason = lastSelection != null ? lastSelection.Reason : null;
            debugState.DiversityReason = lastSelection != null ? lastSelection.DiversityReason : null;
            debugState.CandidateCount = lastSelection != null ? lastSelection.CandidateCount : 0;
            debugState.IsPlaying = playbackState == PlaybackState.Playing;
            debugState.PlaybackStateLabel = ResolvePlaybackStateLabel();
            debugState.RetimeClampHit = characterController != null && characterController.IsRetimeClampHit;
            var debugVariant = characterController != null && characterController.ActiveVariant != null ? characterController.ActiveVariant : activeVariant;
            var debugClip = characterController != null && characterController.ActiveEntry != null ? characterController.ActiveEntry : activeClipEntry;
            debugState.NativeBpm = debugClip != null ? debugClip.EffectiveNativeBpm : 0f;
            debugState.NativePhraseBeats = debugClip != null ? debugClip.EffectiveNativePhraseBeats(manifest != null ? manifest.defaultPhraseBeats : 8) : 0;
            debugState.CurrentVariantRole = debugVariant != null ? debugVariant.EffectiveRole : null;
            debugState.CurrentVarietyGroup = debugVariant != null ? debugVariant.EffectiveVarietyGroup : null;
            debugState.VariantStartOffsetBeats = debugVariant != null ? debugVariant.startOffsetBeats : 0;
            debugState.VariantSliceBeats = debugVariant != null ? debugVariant.sliceBeats : 0;
            debugState.BeatsSinceLastSwitch = activeVariantActivatedBeatIndex >= 0 && conductor != null
                ? Mathf.Max(0, (conductor.CurrentBeatIndex - activeVariantActivatedBeatIndex) / Mathf.Max(1, conductor.CurrentRhythm.BeatGrouping))
                : 0;
            debugState.SwitchPressure = choreography != null
                ? choreography.EstimateSwitchPressure(
                    new DanceSelectionContext
                    {
                        SongId = songData != null ? songData.songId : "song",
                        CurrentBeatIndex = conductor != null ? conductor.CurrentBeatIndex : 0,
                        CurrentBarIndex = conductor != null ? conductor.CurrentBarIndex : 0,
                        CurrentSegment = conductor != null ? conductor.CurrentSegment : null,
                        CurrentClipId = debugClip != null ? debugClip.clipId : null,
                        CurrentVariantId = debugVariant != null ? debugVariant.variantId : null,
                        CurrentVarietyGroup = debugVariant != null ? debugVariant.EffectiveVarietyGroup : null,
                        CurrentRole = debugVariant != null ? debugVariant.EffectiveRole : null,
                        SegmentChanged = pendingSegmentChange,
                        Forced = forceSwitchOnNextBeat,
                        LastActivatedBeatIndex = activeVariantActivatedBeatIndex,
                        PhraseIndex = manifest != null && manifest.defaultPhraseBeats > 0 && conductor != null ? Mathf.Max(0, conductor.CurrentRhythm.PerceivedBeatIndex / manifest.defaultPhraseBeats) : 0,
                        LocalBpm = conductor != null ? conductor.CurrentRhythm.LocalBpm : 0f,
                        DanceLocalBpm = conductor != null ? conductor.CurrentRhythm.DanceLocalBpm : 0f,
                        CurrentSpeed = characterController != null ? characterController.CurrentPlaybackSpeed : 1f,
                        TargetSpeed = characterController != null ? characterController.TargetPlaybackSpeed : 1f,
                        RetimeStress = characterController != null ? Mathf.Abs(characterController.TargetPlaybackSpeed - characterController.CurrentPlaybackSpeed) : 0f,
                        ClampHitCount = characterController != null ? characterController.ConsecutiveClampHitCount : 0,
                        GrooveStrength = conductor != null ? conductor.CurrentRhythm.GrooveStrength : 0f,
                        IsDownbeat = conductor != null && conductor.CurrentRhythm.IsDownbeat,
                        IsPerceivedDownbeat = conductor != null && conductor.CurrentRhythm.IsPerceivedDownbeat,
                        AllowAccent = conductor != null && conductor.CurrentRhythm.IsStrongBeatWindow && !string.Equals(conductor.CurrentRhythm.EnergyMode, "gentle", System.StringComparison.OrdinalIgnoreCase) && debugState.BeatsSinceLastSwitch >= 6,
                        IsStrongBeatWindow = conductor != null && conductor.CurrentRhythm.IsStrongBeatWindow,
                        StrongBeatReason = conductor != null ? conductor.CurrentRhythm.StrongBeatReason : "none",
                        BeatsSinceLastSwitch = debugState.BeatsSinceLastSwitch,
                        PerceivedBeatIndex = conductor != null ? conductor.CurrentRhythm.PerceivedBeatIndex : 0,
                        BeatGrouping = conductor != null ? conductor.CurrentRhythm.BeatGrouping : currentPlaybackProfile.EffectiveBeatGrouping,
                        EnergyMode = conductor != null ? conductor.CurrentRhythm.EnergyMode : currentPlaybackProfile.EffectiveEnergyMode,
                    },
                    debugVariant,
                    manifest != null ? manifest.defaultPhraseBeats : 8)
                : 0f;
            debugState.CurrentClipTimeSec = characterController != null ? characterController.CurrentClipTimeSec : 0f;
            debugState.CurrentClipIsHumanMotion = characterController != null && characterController.CurrentClipIsHumanMotion;
            debugState.CompactCurrentClip = ResolveCompactCurrentClip();
            debugState.CompactNextClip = ResolveCompactNextClip();
            debugState.CompactReason = ResolveCompactReason();
            debugState.CompactStrongBeat = ResolveCompactStrongBeat();
            debugState.CompactSegment = debugState.SegmentLabel;

            if (debugUi != null)
            {
                debugUi.Render(debugState);
            }
        }

        private void LateUpdate()
        {
            MaintainAvatarAboveFloor();
        }

        private AvatarVisibilityMetrics EvaluateAvatarVisibility()
        {
            var metrics = new AvatarVisibilityMetrics
            {
                CameraName = targetCamera != null ? targetCamera.name : "n/a",
            };

            if (targetCamera == null || avatarAnimator == null)
            {
                return metrics;
            }

            var bounds = ResolveAvatarBounds(avatarAnimator.transform.root.gameObject);
            metrics.Distance = Vector3.Distance(targetCamera.transform.position, bounds.center);

            var planes = GeometryUtility.CalculateFrustumPlanes(targetCamera);
            metrics.InFrustum = GeometryUtility.TestPlanesAABB(planes, bounds);

            var viewportCenter = targetCamera.WorldToViewportPoint(bounds.center);
            metrics.CenterVisible =
                viewportCenter.z > 0f &&
                viewportCenter.x >= 0f && viewportCenter.x <= 1f &&
                viewportCenter.y >= 0f && viewportCenter.y <= 1f;

            var corners = GetBoundsCorners(bounds);
            var minX = 1f;
            var maxX = 0f;
            var minY = 1f;
            var maxY = 0f;
            var hasProjectedPoint = false;

            for (var i = 0; i < corners.Length; i++)
            {
                var viewport = targetCamera.WorldToViewportPoint(corners[i]);
                if (viewport.z <= 0f)
                {
                    continue;
                }

                hasProjectedPoint = true;
                minX = Mathf.Min(minX, Mathf.Clamp01(viewport.x));
                maxX = Mathf.Max(maxX, Mathf.Clamp01(viewport.x));
                minY = Mathf.Min(minY, Mathf.Clamp01(viewport.y));
                maxY = Mathf.Max(maxY, Mathf.Clamp01(viewport.y));
            }

            if (hasProjectedPoint)
            {
                metrics.ViewportCoverage = Mathf.Clamp01((maxX - minX) * (maxY - minY));
            }

            return metrics;
        }

        private static Vector3[] GetBoundsCorners(Bounds bounds)
        {
            var min = bounds.min;
            var max = bounds.max;
            return new[]
            {
                new Vector3(min.x, min.y, min.z),
                new Vector3(min.x, min.y, max.z),
                new Vector3(min.x, max.y, min.z),
                new Vector3(min.x, max.y, max.z),
                new Vector3(max.x, min.y, min.z),
                new Vector3(max.x, min.y, max.z),
                new Vector3(max.x, max.y, min.z),
                new Vector3(max.x, max.y, max.z),
            };
        }

        private Material GetOrCreateAvatarFallbackMaterial()
        {
            if (avatarFallbackMaterial != null)
            {
                return avatarFallbackMaterial;
            }

            var shader = Shader.Find(AvatarFallbackShaderName);
            if (shader == null)
            {
                Debug.LogError("Avatar fallback shader could not be found.");
                return null;
            }

            avatarFallbackMaterial = new Material(shader);
            avatarFallbackMaterial.name = "DanceAvatarFallbackMaterial";
            avatarFallbackMaterial.color = new Color(0.88f, 0.83f, 0.78f, 1f);
            return avatarFallbackMaterial;
        }

        private Material CreateVisibleMaterial(Material source, Material fallbackMaterial, string rendererName, int materialIndex, ref bool repaired)
        {
            if (fallbackMaterial == null)
            {
                return source;
            }

            if (source == null)
            {
                repaired = true;
                Debug.LogWarning(string.Format("Avatar renderer fallback: '{0}' material slot {1} was null.", rendererName, materialIndex));
                return fallbackMaterial;
            }

            var shader = source.shader;
            var shaderName = shader != null ? shader.name : "null";
            var isSupported = shader != null && shader.isSupported;
            Debug.Log(string.Format(
                "Avatar renderer material: renderer '{0}', slot {1}, material '{2}', shader '{3}', supported {4}.",
                rendererName,
                materialIndex,
                source.name,
                shaderName,
                isSupported));

            var needsFallback =
                shader == null ||
                !isSupported ||
                shaderName == "Hidden/InternalErrorShader";

            if (!needsFallback)
            {
                return source;
            }

            repaired = true;
            var visibleMaterial = new Material(fallbackMaterial);
            visibleMaterial.name = string.Format("{0}_Fallback_{1}", rendererName, materialIndex);

            if (source.HasProperty("_Color"))
            {
                visibleMaterial.color = source.color;
            }

            if (source.HasProperty("_MainTex"))
            {
                visibleMaterial.mainTexture = source.mainTexture;
            }

            Debug.LogWarning(string.Format(
                "Avatar renderer fallback: renderer '{0}', slot {1}, shader '{2}' was unsupported. Replaced with '{3}'.",
                rendererName,
                materialIndex,
                shaderName,
                AvatarFallbackShaderName));
            return visibleMaterial;
        }
    }

    public static class DebugAudioClipFactory
    {
        public static AudioClip CreateClickTrack(SongAnalysisData songData)
        {
            var sampleRate = 44100;
            var durationSec = songData != null && songData.durationSec > 0f ? songData.durationSec : 16f;
            var sampleCount = Mathf.CeilToInt(durationSec * sampleRate);
            var samples = new float[sampleCount];
            var beats = songData != null && songData.beats != null && songData.beats.Length > 0
                ? songData.beats
                : new[] { 0f, 0.5f, 1f, 1.5f };

            for (var i = 0; i < beats.Length; i++)
            {
                var start = Mathf.Clamp(Mathf.RoundToInt(beats[i] * sampleRate), 0, sampleCount - 1);
                var clickLength = Mathf.Min(Mathf.RoundToInt(0.045f * sampleRate), sampleCount - start);
                for (var j = 0; j < clickLength; j++)
                {
                    var t = j / (float)sampleRate;
                    var envelope = 1f - (j / (float)clickLength);
                    var frequency = i % 4 == 0 ? 1320f : 880f;
                    samples[start + j] += Mathf.Sin(t * Mathf.PI * 2f * frequency) * envelope * 0.18f;
                }
            }

            var clip = AudioClip.Create("generated_click_track", sampleCount, 1, sampleRate, false);
            clip.SetData(samples, 0);
            return clip;
        }
    }
}
