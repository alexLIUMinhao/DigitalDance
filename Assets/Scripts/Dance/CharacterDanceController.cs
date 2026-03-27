using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

namespace DanceDemo
{
    public class CharacterDanceController : MonoBehaviour
    {
        private struct SlotState
        {
            public AnimationClipPlayable Playable;
            public AnimationClip Clip;
            public MotionClipEntry Entry;
            public DanceVariantCandidate Variant;
        }

        private struct PendingTransition
        {
            public DanceVariantCandidate Variant;
            public MotionClipEntry Entry;
            public AnimationClip Clip;
            public double DspStartTime;
            public float StartOffsetTimeSec;
        }

        [SerializeField] private float crossfadeDuration = 0.38f;
        [SerializeField] private float speedSmoothing = 7.5f;
        [SerializeField] private float grooveInfluence = 0.14f;

        private Animator animator;
        private PlayableGraph graph;
        private AnimationMixerPlayable mixer;
        private SlotState slotA;
        private SlotState slotB;
        private Dictionary<string, AnimationClip> clipLookup;
        private int activeSlotIndex = -1;
        private PendingTransition? pendingTransition;
        private bool isFading;
        private float fadeElapsed;
        private int fadeFromIndex;
        private int fadeToIndex;
        private string currentClipId;
        private string currentVariantId;
        private float currentPlaybackSpeed = 1f;
        private float targetPlaybackSpeed = 1f;
        private bool retimeClampHit;
        private int consecutiveClampHitCount;
        private bool isPlaybackPaused;

        public string CurrentClipId => currentClipId;
        public string CurrentVariantId => currentVariantId;
        public string PendingClipId => pendingTransition.HasValue && pendingTransition.Value.Entry != null ? pendingTransition.Value.Entry.clipId : null;
        public string PendingVariantId => pendingTransition.HasValue && pendingTransition.Value.Variant != null ? pendingTransition.Value.Variant.variantId : null;
        public float CurrentPlaybackSpeed => currentPlaybackSpeed;
        public float TargetPlaybackSpeed => targetPlaybackSpeed;
        public bool IsRetimeClampHit => retimeClampHit;
        public int ConsecutiveClampHitCount => consecutiveClampHitCount;
        public MotionClipEntry ActiveEntry => activeSlotIndex < 0 ? null : GetSlot(activeSlotIndex).Entry;
        public DanceVariantCandidate ActiveVariant => activeSlotIndex < 0 ? null : GetSlot(activeSlotIndex).Variant;
        public float SpeedSmoothing => speedSmoothing;
        public float GrooveInfluence => grooveInfluence;
        public float CurrentClipTimeSec => activeSlotIndex < 0 ? 0f : (float)GetSlot(activeSlotIndex).Playable.GetTime();
        public bool CurrentClipIsHumanMotion => activeSlotIndex >= 0 && GetSlot(activeSlotIndex).Clip != null && GetSlot(activeSlotIndex).Clip.humanMotion;

        public void Initialize(Animator targetAnimator, Dictionary<string, AnimationClip> lookup, float fadeDuration)
        {
            DestroyGraph();

            animator = targetAnimator;
            clipLookup = lookup ?? new Dictionary<string, AnimationClip>();
            crossfadeDuration = Mathf.Max(0.01f, fadeDuration);
            activeSlotIndex = -1;
            pendingTransition = null;
            isFading = false;
            currentClipId = null;
            currentVariantId = null;
            currentPlaybackSpeed = 1f;
            targetPlaybackSpeed = 1f;
            retimeClampHit = false;
            consecutiveClampHitCount = 0;
            isPlaybackPaused = false;
            slotA = default;
            slotB = default;

            animator.applyRootMotion = false;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            graph = PlayableGraph.Create("DancePlayableGraph");
            graph.SetTimeUpdateMode(DirectorUpdateMode.GameTime);
            mixer = AnimationMixerPlayable.Create(graph, 2);
            var output = AnimationPlayableOutput.Create(graph, "DanceAnimation", animator);
            output.SetSourcePlayable(mixer);
            mixer.SetInputWeight(0, 0f);
            mixer.SetInputWeight(1, 0f);
            graph.Play();
        }

        public void PausePlayback()
        {
            if (!graph.IsValid() || isPlaybackPaused)
            {
                return;
            }

            pendingTransition = null;
            isPlaybackPaused = true;
            graph.Stop();
        }

        public void ResumePlayback()
        {
            if (!graph.IsValid() || !isPlaybackPaused)
            {
                return;
            }

            isPlaybackPaused = false;
            graph.Play();
            ApplySpeedToSlots(currentPlaybackSpeed);
        }

        public void StopAndResetPose()
        {
            pendingTransition = null;
            isFading = false;
            fadeElapsed = 0f;
            fadeFromIndex = 0;
            fadeToIndex = 0;
            activeSlotIndex = -1;
            currentClipId = null;
            currentVariantId = null;
            currentPlaybackSpeed = 1f;
            targetPlaybackSpeed = 1f;
            retimeClampHit = false;
            consecutiveClampHitCount = 0;
            isPlaybackPaused = false;
            slotA = default;
            slotB = default;

            DestroyGraph();

            if (animator != null)
            {
                animator.Rebind();
                animator.Update(0f);
                animator.applyRootMotion = false;
                animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            }
        }

        public AnimationClip GetClip(MotionClipEntry entry)
        {
            if (entry == null || clipLookup == null || string.IsNullOrEmpty(entry.clipId))
            {
                return null;
            }

            clipLookup.TryGetValue(entry.clipId, out var clip);
            return clip;
        }

        public void QueueClip(DanceVariantCandidate variant, AnimationClip clip, double dspStartTime)
        {
            var entry = variant != null ? variant.sourceClip : null;
            if (entry == null || clip == null)
            {
                Debug.LogWarning("QueueClip skipped because variant or clip was null.");
                return;
            }

            pendingTransition = new PendingTransition
            {
                Variant = variant,
                Entry = entry,
                Clip = clip,
                DspStartTime = dspStartTime,
                StartOffsetTimeSec = variant.ResolveStartOffsetTimeSec(),
            };
            Debug.Log(string.Format(
                "Queued clip '{0}' variant '{1}' for dsp {2:0.000} with offset {3:0.00}s.",
                entry.clipId,
                variant.variantId,
                dspStartTime,
                variant.ResolveStartOffsetTimeSec()));
        }

        public void UpdatePlaybackSpeed(float danceLocalBpm, float grooveStrength, bool isPerceivedDownbeat, float beatPhase, string energyMode)
        {
            if (isPlaybackPaused)
            {
                return;
            }

            var entry = ActiveEntry;
            if (entry == null || activeSlotIndex < 0)
            {
                currentPlaybackSpeed = Mathf.Lerp(currentPlaybackSpeed, 1f, Time.deltaTime * speedSmoothing);
                targetPlaybackSpeed = 1f;
                retimeClampHit = false;
                consecutiveClampHitCount = 0;
                return;
            }

            var nativeBpm = entry.EffectiveNativeBpm;
            var resolvedEnergyMode = ResolveEnergyMode(energyMode);
            var modeGrooveInfluence = grooveInfluence;
            var pulseScale = 1f;
            var smoothingScale = 1f;
            switch (resolvedEnergyMode)
            {
                case "gentle":
                    modeGrooveInfluence *= 0.24f;
                    pulseScale = 0.01f;
                    smoothingScale = 1.85f;
                    break;
                case "driving":
                    modeGrooveInfluence *= 1.12f;
                    pulseScale = 1.18f;
                    smoothingScale = 0.92f;
                    break;
            }

            var baseTempoRatio = danceLocalBpm / nativeBpm;
            var grooveOffset = (grooveStrength - 0.5f) * modeGrooveInfluence * 2.35f;
            var beatPulse = Mathf.Clamp01(1f - (beatPhase * 1.65f));
            var accentStrength = (isPerceivedDownbeat ? 0.085f : 0.045f) * pulseScale * Mathf.Lerp(0.45f, 1f, grooveStrength);
            var pulseMultiplier = 1f + (beatPulse * accentStrength);
            var rawSpeed = baseTempoRatio * (1f + grooveOffset) * pulseMultiplier;
            ResolveEffectiveSpeedRange(entry, resolvedEnergyMode, out var minSpeed, out var maxSpeed);
            targetPlaybackSpeed = Mathf.Clamp(rawSpeed, minSpeed, maxSpeed);
            retimeClampHit = rawSpeed < minSpeed - 0.005f || rawSpeed > maxSpeed + 0.005f;
            if (retimeClampHit)
            {
                consecutiveClampHitCount++;
            }
            else
            {
                consecutiveClampHitCount = Mathf.Max(0, consecutiveClampHitCount - 1);
            }

            var smoothing = speedSmoothing * smoothingScale * (isPerceivedDownbeat ? 1.15f : 0.85f);
            currentPlaybackSpeed = Mathf.Lerp(currentPlaybackSpeed, targetPlaybackSpeed, 1f - Mathf.Exp(-smoothing * Time.deltaTime));
            ApplySpeedToSlots(currentPlaybackSpeed);
        }

        private static string ResolveEnergyMode(string energyMode)
        {
            if (string.IsNullOrEmpty(energyMode))
            {
                return "balanced";
            }

            switch (energyMode.ToLowerInvariant())
            {
                case "gentle":
                case "driving":
                    return energyMode.ToLowerInvariant();
                default:
                    return "balanced";
            }
        }

        private static void ResolveEffectiveSpeedRange(MotionClipEntry entry, string energyMode, out float minSpeed, out float maxSpeed)
        {
            minSpeed = entry.EffectiveSpeedMin;
            maxSpeed = entry.EffectiveSpeedMax;

            switch (energyMode)
            {
                case "gentle":
                    minSpeed = Mathf.Min(minSpeed, 0.68f);
                    maxSpeed = Mathf.Min(maxSpeed, 1.0f);
                    break;
                case "driving":
                    maxSpeed = Mathf.Max(maxSpeed, 1.2f);
                    break;
            }
        }

        private void Update()
        {
            if (isPlaybackPaused)
            {
                return;
            }

            if (pendingTransition.HasValue && AudioSettings.dspTime >= pendingTransition.Value.DspStartTime)
            {
                var transition = pendingTransition.Value;
                StartTransition(transition.Variant, transition.Entry, transition.Clip, transition.StartOffsetTimeSec);
                pendingTransition = null;
            }

            if (isFading)
            {
                fadeElapsed += Time.deltaTime;
                var t = Mathf.Clamp01(fadeElapsed / crossfadeDuration);
                mixer.SetInputWeight(fadeFromIndex, 1f - t);
                mixer.SetInputWeight(fadeToIndex, t);
                if (t >= 1f)
                {
                    isFading = false;
                    mixer.SetInputWeight(fadeFromIndex, 0f);
                    mixer.SetInputWeight(fadeToIndex, 1f);
                }
            }

            LoopSlot(ref slotA);
            LoopSlot(ref slotB);
        }

        private void StartTransition(DanceVariantCandidate variant, MotionClipEntry entry, AnimationClip clip, float startOffsetTimeSec)
        {
            if (!graph.IsValid())
            {
                return;
            }

            if (activeSlotIndex == -1)
            {
                ConfigureSlot(0, variant, entry, clip, startOffsetTimeSec);
                mixer.SetInputWeight(0, 1f);
                mixer.SetInputWeight(1, 0f);
                activeSlotIndex = 0;
                currentClipId = entry.clipId;
                currentVariantId = variant != null ? variant.variantId : null;
                return;
            }

            var nextSlot = activeSlotIndex == 0 ? 1 : 0;
            ConfigureSlot(nextSlot, variant, entry, clip, startOffsetTimeSec);

            fadeFromIndex = activeSlotIndex;
            fadeToIndex = nextSlot;
            fadeElapsed = 0f;
            isFading = true;
            activeSlotIndex = nextSlot;
            currentClipId = entry.clipId;
            currentVariantId = variant != null ? variant.variantId : null;
        }

        private void ConfigureSlot(int slotIndex, DanceVariantCandidate variant, MotionClipEntry entry, AnimationClip clip, float startOffsetTimeSec)
        {
            var state = GetSlot(slotIndex);
            if (state.Playable.IsValid())
            {
                graph.Disconnect(mixer, slotIndex);
                state.Playable.Destroy();
            }

            state.Playable = AnimationClipPlayable.Create(graph, clip);
            state.Playable.SetApplyFootIK(false);
            state.Playable.SetApplyPlayableIK(false);
            var initialTime = ResolveInitialTime(clip, startOffsetTimeSec);
            state.Playable.SetTime(initialTime);
            state.Playable.SetSpeed(currentPlaybackSpeed);
            state.Playable.SetDuration(entry.loopable ? double.MaxValue : clip.length);
            state.Clip = clip;
            state.Entry = entry;
            state.Variant = variant;
            clip.wrapMode = entry.loopable ? WrapMode.Loop : WrapMode.Once;
            Debug.Log(string.Format(
                "Configured dance slot {0} with clip '{1}' variant '{2}' start {3:0.00}s (len {4:0.00}s, humanMotion {5}, legacy {6}).",
                slotIndex,
                clip.name,
                variant != null ? variant.variantId : "base",
                initialTime,
                clip.length,
                clip.humanMotion,
                clip.legacy));

            graph.Connect(state.Playable, 0, mixer, slotIndex);
            mixer.SetInputWeight(slotIndex, 0f);
            SetSlot(slotIndex, state);
        }

        private void ApplySpeedToSlots(float speed)
        {
            if (slotA.Playable.IsValid())
            {
                slotA.Playable.SetSpeed(speed);
            }

            if (slotB.Playable.IsValid())
            {
                slotB.Playable.SetSpeed(speed);
            }
        }

        private SlotState GetSlot(int slotIndex)
        {
            return slotIndex == 0 ? slotA : slotB;
        }

        private void SetSlot(int slotIndex, SlotState state)
        {
            if (slotIndex == 0)
            {
                slotA = state;
            }
            else
            {
                slotB = state;
            }
        }

        private static void LoopSlot(ref SlotState slot)
        {
            if (!slot.Playable.IsValid() || slot.Clip == null || slot.Clip.length <= 0.01f)
            {
                return;
            }

            var time = slot.Playable.GetTime();
            if (time >= slot.Clip.length)
            {
                slot.Playable.SetTime(time % slot.Clip.length);
                slot.Playable.SetDone(false);
            }
        }

        private static double ResolveInitialTime(AnimationClip clip, float startOffsetTimeSec)
        {
            if (clip == null || clip.length <= 0.01f)
            {
                return 0d;
            }

            return Mathf.Repeat(startOffsetTimeSec, clip.length);
        }

        private void DestroyGraph()
        {
            if (!graph.IsValid())
            {
                return;
            }

            graph.Destroy();
        }

        private void OnDestroy()
        {
            DestroyGraph();
        }
    }
}
