#if UNITY_INCLUDE_TESTS
using System;
using System.Collections;
using System.Reflection;
using DanceDemo;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace DanceDemoTests
{
    public class DanceDemoPlaybackFlowTests
    {
        private const string ScenePath = "Assets/MainScene.unity";
        private static readonly BindingFlags InstanceFlags = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            foreach (var bootstrap in UnityEngine.Object.FindObjectsOfType<DanceDemoBootstrap>(true))
            {
                UnityEngine.Object.DestroyImmediate(bootstrap.gameObject);
            }

            SceneManager.LoadScene(ScenePath);
            yield return null;
            yield return WaitForBootstrapReady();
        }

        [UnityTest]
        public IEnumerator Startup_EntersIdleWithoutAutoPlayback()
        {
            var bootstrap = GetBootstrap();
            var conductor = GetPrivateField<MusicConductor>(bootstrap, "conductor");
            var controller = GetPrivateField<CharacterDanceController>(bootstrap, "characterController");
            var debugState = GetPrivateField<DanceDebugState>(bootstrap, "debugState");

            Assert.NotNull(bootstrap);
            Assert.NotNull(conductor);
            Assert.NotNull(controller);
            Assert.NotNull(debugState);
            Assert.AreEqual("Idle", GetPrivateField<object>(bootstrap, "playbackState").ToString());
            Assert.IsFalse(conductor.IsPlaying);
            Assert.IsFalse(debugState.IsPlaying);
            Assert.That(debugState.RuntimeMessage, Does.Contain("Ready"));
            Assert.That(controller.CurrentClipId, Is.Null.Or.Empty);
            yield break;
        }

        [UnityTest]
        public IEnumerator SwitchingSongs_ReturnsToIdleWithoutAutoPlay()
        {
            var bootstrap = GetBootstrap();
            InvokePrivateMethod(bootstrap, "HandleSongSelectionChanged", "audio1_mp3");
            yield return WaitForBootstrapReady();

            var conductor = GetPrivateField<MusicConductor>(bootstrap, "conductor");
            var controller = GetPrivateField<CharacterDanceController>(bootstrap, "characterController");
            var debugState = GetPrivateField<DanceDebugState>(bootstrap, "debugState");
            var selectedSongId = GetPrivateField<string>(bootstrap, "selectedSongId");
            var loadedSongId = GetPrivateField<string>(bootstrap, "loadedSongId");

            Assert.AreEqual("audio1_mp3", selectedSongId);
            Assert.AreEqual("audio1_mp3", loadedSongId);
            Assert.AreEqual("Idle", GetPrivateField<object>(bootstrap, "playbackState").ToString());
            Assert.IsFalse(conductor.IsPlaying);
            Assert.That(debugState.RuntimeMessage, Does.Contain("Song changed. Press Start."));
            Assert.That(controller.CurrentClipId, Is.Null.Or.Empty);
        }

        [UnityTest]
        public IEnumerator StartPauseStart_ConfiguresInitialClip_AndResumesClipTime()
        {
            var bootstrap = GetBootstrap();
            InvokePrivateMethod(bootstrap, "HandleStartPlaybackRequested");
            yield return WaitUntil(() =>
            {
                var controller = GetPrivateField<CharacterDanceController>(bootstrap, "characterController");
                return controller != null && !string.IsNullOrEmpty(controller.CurrentClipId);
            }, 5f);

            var conductor = GetPrivateField<MusicConductor>(bootstrap, "conductor");
            var controllerAfterStart = GetPrivateField<CharacterDanceController>(bootstrap, "characterController");

            Assert.IsTrue(conductor.IsPlaying);
            Assert.That(controllerAfterStart.CurrentClipId, Is.Not.Null.And.Not.Empty);

            yield return new WaitForSeconds(0.75f);
            var runningClipTime = controllerAfterStart.CurrentClipTimeSec;
            Assert.Greater(runningClipTime, 0.05f);

            InvokePrivateMethod(bootstrap, "HandlePausePlaybackRequested");
            var pausedClipTime = controllerAfterStart.CurrentClipTimeSec;
            yield return new WaitForSeconds(0.5f);
            Assert.That(controllerAfterStart.CurrentClipTimeSec, Is.EqualTo(pausedClipTime).Within(0.02f));
            Assert.IsFalse(conductor.IsPlaying);

            InvokePrivateMethod(bootstrap, "HandleStartPlaybackRequested");
            yield return new WaitForSeconds(0.6f);
            Assert.IsTrue(conductor.IsPlaying);
            Assert.Greater(controllerAfterStart.CurrentClipTimeSec, pausedClipTime + 0.03f);
        }

        private static DanceDemoBootstrap GetBootstrap()
        {
            return UnityEngine.Object.FindObjectOfType<DanceDemoBootstrap>(true);
        }

        private static IEnumerator WaitForBootstrapReady()
        {
            yield return WaitUntil(() =>
            {
                var bootstrap = GetBootstrap();
                if (bootstrap == null)
                {
                    return false;
                }

                var isReloading = GetPrivateField<bool>(bootstrap, "isReloading");
                var debugState = GetPrivateField<DanceDebugState>(bootstrap, "debugState");
                return !isReloading && debugState != null && !string.IsNullOrEmpty(debugState.RuntimeMessage);
            }, 8f);
        }

        private static IEnumerator WaitUntil(Func<bool> condition, float timeoutSec)
        {
            var endTime = Time.realtimeSinceStartup + timeoutSec;
            while (Time.realtimeSinceStartup < endTime)
            {
                if (condition())
                {
                    yield break;
                }

                yield return null;
            }

            Assert.Fail("Timed out waiting for expected runtime condition.");
        }

        private static T GetPrivateField<T>(object instance, string fieldName)
        {
            var field = instance.GetType().GetField(fieldName, InstanceFlags);
            Assert.NotNull(field, "Missing field: " + fieldName);
            return (T)field.GetValue(instance);
        }

        private static object InvokePrivateMethod(object instance, string methodName, params object[] args)
        {
            var method = instance.GetType().GetMethod(methodName, InstanceFlags);
            Assert.NotNull(method, "Missing method: " + methodName);
            return method.Invoke(instance, args);
        }
    }
}
#endif
