using System;
using System.Collections.Generic;
using UnityEngine;

namespace DanceDemo
{
    public class ChoreographyEngine
    {
        private const int MinimumHoldBeatsBeforeStrongSwitch = 6;
        private const float TempoFitGuardThreshold = 0.82f;
        private const int MaxRecentVariantHistory = 8;
        private const int MaxRecentClipHistory = 8;
        private const int MaxRecentGroupHistory = 6;
        private const int MaxRecentRoleHistory = 6;

        private readonly MotionManifestData manifest;
        private readonly List<DanceVariantCandidate> variantPool = new List<DanceVariantCandidate>();
        private readonly List<string> recentVariantHistory = new List<string>();
        private readonly List<string> recentClipHistory = new List<string>();
        private readonly List<string> recentGroupHistory = new List<string>();
        private readonly List<string> recentRoleHistory = new List<string>();
        private readonly Dictionary<string, int> lastActivationBeatByVariant = new Dictionary<string, int>();
        private readonly Dictionary<string, int> lastActivationBeatByClip = new Dictionary<string, int>();
        private readonly Dictionary<string, int> lastActivationBeatByGroup = new Dictionary<string, int>();

        public ChoreographyEngine(MotionManifestData manifestData)
        {
            manifest = manifestData;
            BuildVariantPool();
        }

        public void NotifyVariantActivated(DanceVariantCandidate candidate, int beatIndex)
        {
            if (candidate == null || candidate.sourceClip == null || string.IsNullOrEmpty(candidate.SourceClipId))
            {
                return;
            }

            lastActivationBeatByVariant[candidate.variantId] = beatIndex;
            lastActivationBeatByClip[candidate.SourceClipId] = beatIndex;
            lastActivationBeatByGroup[candidate.EffectiveVarietyGroup] = beatIndex;

            AppendHistory(recentVariantHistory, candidate.variantId, MaxRecentVariantHistory);
            AppendHistory(recentClipHistory, candidate.SourceClipId, MaxRecentClipHistory);
            AppendHistory(recentGroupHistory, candidate.EffectiveVarietyGroup, MaxRecentGroupHistory);
            AppendHistory(recentRoleHistory, candidate.EffectiveRole, MaxRecentRoleHistory);
        }

        public bool ShouldForceReselect(DanceSelectionContext context, DanceVariantCandidate activeVariant, int defaultPhraseBeats)
        {
            if (context.Forced || context.SegmentChanged || activeVariant == null)
            {
                return true;
            }

            var beatsSinceActivation = Mathf.Max(0, context.BeatsSinceLastSwitch);
            var minimumHoldBeats = ResolveMinimumHoldBeats(context.EnergyMode);
            var onPerceivedBoundary = IsPerceivedBeatBoundary(context);
            var sliceBeats = activeVariant.sliceBeats > 0
                ? activeVariant.sliceBeats
                : Mathf.Max(1, activeVariant.sourceClip != null ? activeVariant.sourceClip.EffectiveNativePhraseBeats(defaultPhraseBeats) : defaultPhraseBeats);

            if (context.IsStrongBeatWindow && onPerceivedBoundary && beatsSinceActivation >= minimumHoldBeats)
            {
                return true;
            }

            if (activeVariant.IsAccent)
            {
                if (beatsSinceActivation >= sliceBeats)
                {
                    return true;
                }
            }
            else
            {
                if (IsGentleMode(context.EnergyMode))
                {
                    if (context.IsStrongBeatWindow && onPerceivedBoundary && beatsSinceActivation >= Mathf.Max(minimumHoldBeats, sliceBeats))
                    {
                        return true;
                    }
                }
                else
                {
                if (sliceBeats > 0 && sliceBeats <= 4 && beatsSinceActivation > 0 && beatsSinceActivation % sliceBeats == 0)
                {
                    return true;
                }

                if (context.AllowAccent && beatsSinceActivation >= minimumHoldBeats && onPerceivedBoundary)
                {
                    return true;
                }
                }
            }

            return context.ClampHitCount >= 3 || context.RetimeStress >= 0.18f;
        }

        public float EstimateSwitchPressure(DanceSelectionContext context, DanceVariantCandidate activeVariant, int defaultPhraseBeats)
        {
            if (context == null)
            {
                return 0f;
            }

            if (context.Forced || context.SegmentChanged || activeVariant == null)
            {
                return 1f;
            }

            var beatsSinceActivation = Mathf.Max(0, context.BeatsSinceLastSwitch);
            var sliceBeats = activeVariant.sliceBeats > 0
                ? activeVariant.sliceBeats
                : Mathf.Max(1, activeVariant.sourceClip != null ? activeVariant.sourceClip.EffectiveNativePhraseBeats(defaultPhraseBeats) : defaultPhraseBeats);

            if (context.IsStrongBeatWindow)
            {
                var strongBeatPressure = 0.58f + (Mathf.Clamp01(context.GrooveStrength) * 0.18f);
                if (beatsSinceActivation < ResolveMinimumHoldBeats(context.EnergyMode))
                {
                    strongBeatPressure -= 0.32f;
                }

                if (string.Equals(context.CurrentRole, "accent", StringComparison.OrdinalIgnoreCase))
                {
                    strongBeatPressure -= 0.1f;
                }

                return Mathf.Clamp01(strongBeatPressure);
            }

            if (IsGentleMode(context.EnergyMode))
            {
                return Mathf.Clamp01(0.08f + (Mathf.Clamp01(context.GrooveStrength) * 0.12f));
            }

            if (context.ClampHitCount >= 3 || context.RetimeStress >= 0.18f)
            {
                return Mathf.Clamp01(0.74f + context.RetimeStress);
            }

            if (activeVariant.IsAccent)
            {
                return Mathf.Clamp01(beatsSinceActivation / (float)Mathf.Max(1, sliceBeats));
            }

            if (sliceBeats > 0 && beatsSinceActivation > 0 && beatsSinceActivation % sliceBeats == 0)
            {
                return 0.62f;
            }

            return Mathf.Clamp01(0.16f + (Mathf.Clamp01(context.GrooveStrength) * 0.18f));
        }

        public DanceSelectionResult SelectNextClip(DanceSelectionContext context)
        {
            var result = new DanceSelectionResult
            {
                CandidateCount = variantPool.Count,
                Reason = "no_variants",
                DiversityReason = "no_pool",
            };

            if (variantPool.Count == 0)
            {
                return result;
            }

            var targetEnergy = ResolveTargetEnergyBand(context.CurrentSegment, context.EnergyMode);
            DanceVariantCandidate bestCandidate = null;
            var bestScore = float.NegativeInfinity;
            var bestDiversityReason = string.Empty;

            foreach (var candidate in variantPool)
            {
                if (candidate == null || candidate.sourceClip == null)
                {
                    continue;
                }

                var score = ScoreVariant(candidate, context, targetEnergy, out var diversityReason);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestCandidate = candidate;
                    bestDiversityReason = diversityReason;
                }
            }

            result.SelectedVariant = bestCandidate;
            result.Score = bestScore;
            result.Reason = BuildReason(context, targetEnergy, bestCandidate, bestScore);
            result.DiversityReason = bestDiversityReason;
            return result;
        }

        private void BuildVariantPool()
        {
            variantPool.Clear();
            if (manifest == null || manifest.clips == null)
            {
                return;
            }

            var seenIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (var clip in manifest.clips)
            {
                if (clip == null || string.IsNullOrEmpty(clip.clipId))
                {
                    continue;
                }

                var offsets = SanitizeDistinctPositive(clip.EffectiveEntryOffsetsBeats, 0);
                var slices = SanitizeDistinctPositive(clip.EffectiveSliceBeatsOptions, clip.EffectiveNativePhraseBeats(manifest.defaultPhraseBeats));
                var baseRole = clip.EffectiveRole;

                foreach (var sliceBeats in slices)
                {
                    foreach (var startOffsetBeats in offsets)
                    {
                        AddVariant(seenIds, clip, baseRole, startOffsetBeats, sliceBeats);

                        var allowAccentVariant = sliceBeats <= 4 && (clip.accentBias >= 0.35f || string.Equals(baseRole, "accent", StringComparison.OrdinalIgnoreCase));
                        if (allowAccentVariant && !string.Equals(baseRole, "accent", StringComparison.OrdinalIgnoreCase))
                        {
                            AddVariant(seenIds, clip, "accent", startOffsetBeats, sliceBeats);
                        }
                    }
                }
            }
        }

        private void AddVariant(HashSet<string> seenIds, MotionClipEntry clip, string role, int startOffsetBeats, int sliceBeats)
        {
            var safeRole = string.IsNullOrEmpty(role) ? "loop" : role.ToLowerInvariant();
            var safeSliceBeats = Mathf.Max(1, sliceBeats);
            var safeOffsetBeats = Mathf.Max(0, startOffsetBeats);
            var variantId = string.Format("{0}__{1}_o{2}_s{3}", clip.clipId, safeRole, safeOffsetBeats, safeSliceBeats);
            if (!seenIds.Add(variantId))
            {
                return;
            }

            variantPool.Add(new DanceVariantCandidate
            {
                variantId = variantId,
                sourceClip = clip,
                varietyGroup = clip.EffectiveVarietyGroup,
                role = safeRole,
                startOffsetBeats = safeOffsetBeats,
                sliceBeats = safeSliceBeats,
            });
        }

        private float ScoreVariant(DanceVariantCandidate candidate, DanceSelectionContext context, string targetEnergy, out string diversityReason)
        {
            var clip = candidate.sourceClip;
            var notes = new List<string>(8);
            var score = 0f;

            if (IsSegmentPreferred(clip, context.CurrentSegment))
            {
                score += 4f;
                notes.Add("segment");
            }

            if (string.Equals(clip.energyBand, targetEnergy, StringComparison.OrdinalIgnoreCase))
            {
                score += 3f;
                notes.Add("energy");
            }

            var tempoFit = ScoreTempoFit(clip, context.DanceLocalBpm, context.EnergyMode);
            score += tempoFit * 5f;
            notes.Add(string.Format("tempo {0:0.00}", tempoFit));

            if (!string.IsNullOrEmpty(context.CurrentClipId) && !string.Equals(context.CurrentClipId, clip.clipId, StringComparison.Ordinal) && tempoFit < TempoFitGuardThreshold)
            {
                score -= 9f;
                notes.Add("tempo guard");
            }

            if (context.Forced)
            {
                score += 1f;
            }

            if (context.SegmentChanged)
            {
                score += candidate.IsAccent ? 2.25f : 1.5f;
            }

            if (!string.IsNullOrEmpty(context.CurrentVariantId) && string.Equals(context.CurrentVariantId, candidate.variantId, StringComparison.Ordinal))
            {
                score -= manifest.clips.Length > 1 ? 5f : 0.5f;
                notes.Add("same variant");
            }

            if (!string.IsNullOrEmpty(context.CurrentClipId) && string.Equals(context.CurrentClipId, clip.clipId, StringComparison.Ordinal))
            {
                score -= manifest.clips.Length > 1 ? 2.75f : 0.25f;
                score -= context.RetimeStress * 3f;
                notes.Add("same clip");
            }

            if (!string.IsNullOrEmpty(context.CurrentVarietyGroup))
            {
                if (string.Equals(context.CurrentVarietyGroup, candidate.EffectiveVarietyGroup, StringComparison.Ordinal))
                {
                    score += 1.75f;
                    notes.Add("continuity group");
                }
                else if (!context.SegmentChanged)
                {
                    var groupJumpPenalty = IsGentleMode(context.EnergyMode) ? 1.25f : 0.85f;
                    score -= groupJumpPenalty;
                    notes.Add(string.Format("group jump -{0:0.00}", groupJumpPenalty));
                }
            }

            if (candidate.IsLoop && !string.Equals(context.CurrentRole, "accent", StringComparison.OrdinalIgnoreCase))
            {
                score += 0.55f;
                notes.Add("loop continuity");
            }

            if (candidate.startOffsetBeats == 0)
            {
                score += 0.6f;
                notes.Add("clean entry");
            }
            else
            {
                var offsetPenalty = IsGentleMode(context.EnergyMode) ? 1.05f : 0.4f;
                score -= offsetPenalty;
                notes.Add(string.Format("offset -{0:0.00}", offsetPenalty));
            }

            if (candidate.sliceBeats >= 8)
            {
                var longPhraseBonus = IsGentleMode(context.EnergyMode) ? 1.35f : 0.9f;
                score += longPhraseBonus;
                notes.Add(string.Format("long +{0:0.00}", longPhraseBonus));
            }
            else if (!context.SegmentChanged)
            {
                var shortPhrasePenalty = IsGentleMode(context.EnergyMode) ? 3.0f : 0.85f;
                score -= shortPhrasePenalty;
                notes.Add(string.Format("short -{0:0.00}", shortPhrasePenalty));
            }

            if (context.IsStrongBeatWindow)
            {
                if (!string.IsNullOrEmpty(context.CurrentClipId) && !string.Equals(context.CurrentClipId, clip.clipId, StringComparison.Ordinal))
                {
                    var strongSwitchBonus = IsGentleMode(context.EnergyMode) ? 0.4f : 1.25f;
                    score += strongSwitchBonus;
                    notes.Add(string.Format("strong +{0:0.00}", strongSwitchBonus));
                }
                else if (!string.IsNullOrEmpty(context.CurrentClipId) && string.Equals(context.CurrentClipId, clip.clipId, StringComparison.Ordinal))
                {
                    var strongHoldPenalty = IsGentleMode(context.EnergyMode) ? 0.55f : 1.5f;
                    score -= strongHoldPenalty;
                    notes.Add(string.Format("strong hold -{0:0.00}", strongHoldPenalty));
                    if (context.BeatsSinceLastSwitch >= MinimumHoldBeatsBeforeStrongSwitch * 2)
                    {
                        score -= 1.25f;
                        notes.Add("strong occupancy");
                    }
                }

                if (candidate.IsAccent && (context.SegmentChanged || context.GrooveStrength >= 0.72f))
                {
                    score += 0.65f;
                    notes.Add("accent +0.65");
                }

                if (CountMatchesLast(recentGroupHistory, candidate.EffectiveVarietyGroup, 2) == 0)
                {
                    score += 0.5f;
                    notes.Add("group +0.5");
                }

                if (candidate.sliceBeats <= 4)
                {
                    var strongShortPenalty = IsGentleMode(context.EnergyMode) ? 1.4f : 0.35f;
                    score -= strongShortPenalty;
                    notes.Add(string.Format("strong short -{0:0.00}", strongShortPenalty));
                }
            }

            if (candidate.IsAccent)
            {
                if (!context.SegmentChanged)
                {
                    score -= 2.25f;
                    notes.Add("accent continuity");
                }

                if (IsAccentAllowed(candidate, context, targetEnergy))
                {
                    var accentScore = 0.65f + (clip.accentBias * 3.2f);
                    accentScore += Mathf.Clamp01(context.GrooveStrength) * 1.6f;
                    accentScore += context.SegmentChanged ? 1.1f : 0f;
                    accentScore += context.IsDownbeat ? 0.8f : 0f;
                    score += accentScore;
                    notes.Add(string.Format("accent {0:0.00}", accentScore));
                }
                else
                {
                    score -= 9f;
                    notes.Add("accent blocked");
                }

                if (CountTailMatches(recentRoleHistory, "accent") > 0)
                {
                    score -= 10f;
                    notes.Add("accent repeat");
                }
            }
            else if (string.Equals(context.CurrentRole, "accent", StringComparison.OrdinalIgnoreCase))
            {
                score += 0.9f;
                notes.Add("return loop");
            }

            if (IsGentleMode(context.EnergyMode))
            {
                switch ((clip.energyBand ?? string.Empty).ToLowerInvariant())
                {
                    case "low_energy":
                        score += 2.1f;
                        notes.Add("gentle low");
                        break;
                    case "mid_energy":
                        score += 1.0f;
                        notes.Add("gentle mid");
                        break;
                    case "high_energy":
                        score -= 6.25f;
                        notes.Add("gentle high");
                        break;
                }

                if (candidate.startOffsetBeats > 0)
                {
                    score -= 1.1f;
                    notes.Add("gentle clean");
                }
            }

            var clipRecentCount = CountMatches(recentClipHistory, clip.clipId);
            var groupRecentCount = CountMatches(recentGroupHistory, candidate.EffectiveVarietyGroup);
            var variantRecentCount = CountMatches(recentVariantHistory, candidate.variantId);
            var noveltyScore = 2.1f;
            noveltyScore += clipRecentCount == 0 ? 0.9f : -0.75f * clipRecentCount;
            noveltyScore += groupRecentCount == 0 ? 0.65f : -0.6f * groupRecentCount;
            noveltyScore += variantRecentCount == 0 ? 0.35f : -0.9f * variantRecentCount;
            noveltyScore += candidate.startOffsetBeats > 0 ? 0.25f : 0f;
            noveltyScore += candidate.sliceBeats <= 4 ? 0.2f : 0f;
            score += noveltyScore;
            notes.Add(string.Format("novelty {0:0.00}", noveltyScore));

            var clipTail = CountTailMatches(recentClipHistory, clip.clipId);
            if (clipTail >= clip.EffectiveMaxConsecutiveSelections)
            {
                score -= 12f;
                notes.Add("clip block");
            }
            else if (clipTail > 0)
            {
                var penalty = clipTail * 1.35f;
                score -= penalty;
                notes.Add(string.Format("clip -{0:0.00}", penalty));
            }

            var groupTail = CountTailMatches(recentGroupHistory, candidate.EffectiveVarietyGroup);
            if (groupTail >= 2)
            {
                var penalty = groupTail * 1.45f;
                score -= penalty;
                notes.Add(string.Format("group -{0:0.00}", penalty));
            }

            var recentGroupWindowCount = CountMatchesLast(recentGroupHistory, candidate.EffectiveVarietyGroup, 4);
            if (recentGroupWindowCount >= 2)
            {
                score -= 3.5f;
                notes.Add("group saturation");
            }

            if (lastActivationBeatByClip.TryGetValue(clip.clipId, out var lastBeat))
            {
                var cooldown = Mathf.Max(0, clip.cooldownBeats);
                var beatsSinceUse = context.CurrentBeatIndex - lastBeat;
                if (beatsSinceUse < cooldown && manifest.clips.Length > 1)
                {
                    score -= 10f;
                    notes.Add("clip cooldown");
                }
            }

            if (lastActivationBeatByGroup.TryGetValue(candidate.EffectiveVarietyGroup, out var lastGroupBeat))
            {
                var beatsSinceGroup = context.CurrentBeatIndex - lastGroupBeat;
                if (beatsSinceGroup < clip.EffectiveGroupCooldownBeats && manifest.clips.Length > 2)
                {
                    score -= 5.75f;
                    notes.Add("group cooldown");
                }
            }

            score += ScoreRetimeProfile(clip, context);
            var jitter = DeterministicJitter(context.SongId, context.PhraseIndex, candidate.variantId);
            score += jitter;
            score += StableTieBreaker(candidate.variantId);
            notes.Add(string.Format("jitter {0:+0.00;-0.00;0.00}", jitter));

            diversityReason = string.Join(", ", notes.ToArray());
            return score;
        }

        private static bool IsAccentAllowed(DanceVariantCandidate candidate, DanceSelectionContext context, string targetEnergy)
        {
            if (!candidate.IsAccent)
            {
                return true;
            }

            if (IsGentleMode(context.EnergyMode))
            {
                return false;
            }

            if (context.Forced || context.SegmentChanged)
            {
                return true;
            }

            if (string.Equals(context.CurrentRole, "accent", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!context.AllowAccent)
            {
                return false;
            }

            return context.SegmentChanged || (context.IsStrongBeatWindow && context.GrooveStrength >= 0.72f) || string.Equals(targetEnergy, "high_energy", StringComparison.OrdinalIgnoreCase);
        }

        private static int[] SanitizeDistinctPositive(int[] values, int fallback)
        {
            var normalized = new List<int>();
            var seen = new HashSet<int>();
            if (values != null)
            {
                for (var i = 0; i < values.Length; i++)
                {
                    var safeValue = Mathf.Max(0, values[i]);
                    if (seen.Add(safeValue))
                    {
                        normalized.Add(safeValue);
                    }
                }
            }

            if (normalized.Count == 0)
            {
                normalized.Add(Mathf.Max(0, fallback));
            }

            normalized.Sort();
            return normalized.ToArray();
        }

        private static void AppendHistory(List<string> history, string value, int maxItems)
        {
            if (string.IsNullOrEmpty(value))
            {
                return;
            }

            history.Add(value);
            var overflow = history.Count - maxItems;
            if (overflow > 0)
            {
                history.RemoveRange(0, overflow);
            }
        }

        private static int CountMatches(List<string> history, string value)
        {
            var count = 0;
            for (var i = 0; i < history.Count; i++)
            {
                if (string.Equals(history[i], value, StringComparison.Ordinal))
                {
                    count++;
                }
            }

            return count;
        }

        private static int CountMatchesLast(List<string> history, string value, int windowSize)
        {
            var count = 0;
            var startIndex = Mathf.Max(0, history.Count - windowSize);
            for (var i = startIndex; i < history.Count; i++)
            {
                if (string.Equals(history[i], value, StringComparison.Ordinal))
                {
                    count++;
                }
            }

            return count;
        }

        private static int CountTailMatches(List<string> history, string value)
        {
            var count = 0;
            for (var i = history.Count - 1; i >= 0; i--)
            {
                if (!string.Equals(history[i], value, StringComparison.Ordinal))
                {
                    break;
                }

                count++;
            }

            return count;
        }

        private static float ScoreTempoFit(MotionClipEntry clip, float localBpm, string energyMode)
        {
            var ratio = localBpm / Mathf.Max(1f, clip.EffectiveNativeBpm);
            ResolveEffectiveSpeedRange(clip, energyMode, out var minSpeed, out var maxSpeed);
            var clampedRatio = Mathf.Clamp(ratio, minSpeed, maxSpeed);
            var stress = Mathf.Abs(ratio - clampedRatio) + Mathf.Abs(1f - clampedRatio);
            return Mathf.Clamp01(1f - stress);
        }

        private static float ScoreRetimeProfile(MotionClipEntry clip, DanceSelectionContext context)
        {
            var profile = string.IsNullOrEmpty(clip.retimeProfile) ? "smooth" : clip.retimeProfile.ToLowerInvariant();
            switch (profile)
            {
                case "gentle":
                    return context.DanceLocalBpm < clip.EffectiveNativeBpm ? 0.45f : -0.1f;
                case "punchy":
                    return context.DanceLocalBpm >= clip.EffectiveNativeBpm ? 0.45f : 0f;
                default:
                    return 0.25f - (context.RetimeStress * 0.5f);
            }
        }

        private static bool IsSegmentPreferred(MotionClipEntry clip, SongSegmentData segment)
        {
            if (clip.preferredSegments == null || clip.preferredSegments.Length == 0 || segment == null)
            {
                return false;
            }

            foreach (var label in clip.preferredSegments)
            {
                if (string.Equals(label, segment.label, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private static string ResolveTargetEnergyBand(SongSegmentData segment, string energyMode)
        {
            var resolvedMode = string.IsNullOrEmpty(energyMode) ? "balanced" : energyMode.ToLowerInvariant();
            if (segment == null || string.IsNullOrEmpty(segment.label))
            {
                return resolvedMode == "gentle" ? "low_energy" : "mid_energy";
            }

            switch (resolvedMode)
            {
                case "gentle":
                    switch (segment.label.ToLowerInvariant())
                    {
                        case "chorus":
                        case "instrumental":
                            return "mid_energy";
                        default:
                            return "low_energy";
                    }
                case "driving":
                    switch (segment.label.ToLowerInvariant())
                    {
                        case "intro":
                        case "outro":
                            return "mid_energy";
                        default:
                            return "high_energy";
                    }
                default:
                    switch (segment.label.ToLowerInvariant())
                    {
                        case "intro":
                        case "outro":
                            return "low_energy";
                        case "chorus":
                        case "instrumental":
                            return "high_energy";
                        default:
                            return "mid_energy";
                    }
            }
        }

        private static int ResolveMinimumHoldBeats(string energyMode)
        {
            if (string.Equals(energyMode, "gentle", StringComparison.OrdinalIgnoreCase))
            {
                return 12;
            }

            if (string.Equals(energyMode, "driving", StringComparison.OrdinalIgnoreCase))
            {
                return 4;
            }

            return MinimumHoldBeatsBeforeStrongSwitch;
        }

        private static bool IsGentleMode(string energyMode)
        {
            return string.Equals(energyMode, "gentle", StringComparison.OrdinalIgnoreCase);
        }

        private static void ResolveEffectiveSpeedRange(MotionClipEntry clip, string energyMode, out float minSpeed, out float maxSpeed)
        {
            minSpeed = clip.EffectiveSpeedMin;
            maxSpeed = clip.EffectiveSpeedMax;

            if (string.Equals(energyMode, "gentle", StringComparison.OrdinalIgnoreCase))
            {
                minSpeed = Mathf.Min(minSpeed, 0.72f);
                maxSpeed = Mathf.Min(maxSpeed, 1.02f);
            }
            else if (string.Equals(energyMode, "driving", StringComparison.OrdinalIgnoreCase))
            {
                maxSpeed = Mathf.Max(maxSpeed, 1.2f);
            }
        }

        private static bool IsPerceivedBeatBoundary(DanceSelectionContext context)
        {
            var beatGrouping = Mathf.Max(1, context.BeatGrouping);
            return context.CurrentBeatIndex % beatGrouping == 0;
        }

        private static float DeterministicJitter(string songId, int phraseIndex, string variantId)
        {
            var seed = string.Format("{0}|{1}|{2}", songId ?? "song", phraseIndex, variantId ?? "variant");
            unchecked
            {
                var hash = 17;
                for (var i = 0; i < seed.Length; i++)
                {
                    hash = (hash * 31) + seed[i];
                }

                var normalized = Mathf.Abs(hash % 1000) / 1000f;
                return (normalized - 0.5f) * 0.35f;
            }
        }

        private static float StableTieBreaker(string id)
        {
            if (string.IsNullOrEmpty(id))
            {
                return 0f;
            }

            unchecked
            {
                var hash = 17;
                for (var i = 0; i < id.Length; i++)
                {
                    hash = hash * 31 + id[i];
                }

                return Mathf.Abs(hash % 100) / 1000f;
            }
        }

        private static string BuildReason(DanceSelectionContext context, string targetEnergy, DanceVariantCandidate candidate, float score)
        {
            if (candidate == null || candidate.sourceClip == null)
            {
                return "selection_failed";
            }

            var trigger = context.Forced
                ? "force"
                : context.SegmentChanged
                    ? "segment_change"
                    : context.IsStrongBeatWindow
                        ? "strong_beat"
                        : context.ClampHitCount >= 3 || context.RetimeStress >= 0.18f
                            ? "retime_escape"
                            : candidate.IsAccent
                                ? "accent_window"
                                : "phrase_boundary";

            return string.Format(
                "{0}:{1} -> {2} [{3}] {4}b @ +{5}b ({6}, {7:0.0} BPM, score {8:0.00})",
                trigger,
                string.IsNullOrEmpty(context.StrongBeatReason) ? "none" : context.StrongBeatReason,
                candidate.SourceClipId,
                candidate.EffectiveRole,
                candidate.sliceBeats,
                candidate.startOffsetBeats,
                targetEnergy,
                candidate.EffectiveNativeBpm,
                score);
        }
    }
}
