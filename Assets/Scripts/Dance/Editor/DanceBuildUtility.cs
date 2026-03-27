using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace DanceDemo.Editor
{
    public static class DanceBuildUtility
    {
        private const string OutputDirectory = "Builds/macOS";

        [MenuItem("Tools/Dance Demo/Build macOS Standalone")]
        public static void BuildMacStandalone()
        {
            var scenes = new[] { "Assets/MainScene.unity" };
            Directory.CreateDirectory(OutputDirectory);

            var options = new BuildPlayerOptions
            {
                scenes = scenes,
                target = BuildTarget.StandaloneOSX,
                locationPathName = Path.Combine(OutputDirectory, "DanceDemo.app"),
                options = BuildOptions.None,
            };

            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log("Dance demo build result: " + report.summary.result);
        }

        public static void PerformBuildFromCommandLine()
        {
            BuildMacStandalone();
        }
    }
}
