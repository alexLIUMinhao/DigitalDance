using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace DanceDemo
{
    public class DanceDebugUI : MonoBehaviour
    {
        public event Action StartPlaybackRequested;
        public event Action PausePlaybackRequested;
        public event Action<string> SongSelectionChanged;
        public event Action ForceSwitchRequested;
        public event Action ReloadDataRequested;
        public event Action QuitRequested;

        private Canvas canvas;
        private GameObject debugPanel;
        private GameObject compactPanel;
        private Text statusText;
        private Text compactText;
        private Dropdown compactSongDropdown;
        private Button compactPrevSongButton;
        private Button compactNextSongButton;
        private Button startButton;
        private Button pauseButton;
        private Button forceSwitchButton;
        private Button reloadButton;
        private Button quitButton;
        private Button compactStartButton;
        private Button compactPauseButton;
        private bool isDebugVisible;
        private bool isUpdatingSongDropdown;
        private readonly List<string> compactSongIds = new List<string>();

        public void Initialize()
        {
            if (canvas != null)
            {
                return;
            }

            var canvasGo = new GameObject("DanceDebugCanvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.transform.SetParent(transform, false);
            canvas = canvasGo.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 999;
            EnsureEventSystem();

            var scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);

            debugPanel = CreateDebugPanel(canvasGo.transform);
            statusText = CreateText(debugPanel.transform, "Dance Status", new Vector2(16f, -16f), new Vector2(720f, 560f), 20);
            startButton = CreateButton(debugPanel.transform, "Start", new Vector2(16f, -610f), () => StartPlaybackRequested?.Invoke());
            pauseButton = CreateButton(debugPanel.transform, "Pause", new Vector2(196f, -610f), () => PausePlaybackRequested?.Invoke());
            forceSwitchButton = CreateButton(debugPanel.transform, "Next Beat Switch", new Vector2(376f, -610f), () => ForceSwitchRequested?.Invoke());
            reloadButton = CreateButton(debugPanel.transform, "Reload JSON", new Vector2(556f, -610f), () => ReloadDataRequested?.Invoke());
            quitButton = CreateButton(debugPanel.transform, "Quit", new Vector2(556f, -656f), () => QuitRequested?.Invoke());

            compactPanel = CreateCompactPanel(canvasGo.transform);
            compactText = CreateText(compactPanel.transform, "Choreography", new Vector2(16f, -14f), new Vector2(388f, 138f), 18);
            compactPrevSongButton = CreateButton(compactPanel.transform, "◀", new Vector2(16f, -162f), () => SelectRelativeSong(-1), new Vector2(34f, 32f));
            compactSongDropdown = CreateDropdown(compactPanel.transform, new Vector2(58f, -162f), new Vector2(188f, 32f), HandleSongDropdownChanged);
            compactNextSongButton = CreateButton(compactPanel.transform, "▶", new Vector2(254f, -162f), () => SelectRelativeSong(1), new Vector2(34f, 32f));
            compactStartButton = CreateButton(compactPanel.transform, "Start", new Vector2(16f, -204f), () => StartPlaybackRequested?.Invoke(), new Vector2(96f, 30f));
            compactPauseButton = CreateButton(compactPanel.transform, "Pause", new Vector2(126f, -204f), () => PausePlaybackRequested?.Invoke(), new Vector2(96f, 30f));

            isDebugVisible = false;
            debugPanel.SetActive(isDebugVisible);
            ClearUiSelection();
        }

        public void Render(DanceDebugState state)
        {
            if (state == null)
            {
                return;
            }

            if (statusText != null)
            {
                var builder = new StringBuilder(512);
                foreach (var line in state.ToLines())
                {
                    builder.AppendLine(line);
                }

                statusText.text = builder.ToString();
            }

            if (compactText != null)
            {
                var compactBuilder = new StringBuilder(192);
                foreach (var line in state.ToCompactLines())
                {
                    compactBuilder.AppendLine(line);
                }

                compactText.text = compactBuilder.ToString();
            }

            UpdateButtonInteractivity(state);
        }

        public void SetSongOptions(IReadOnlyList<SongCatalogEntry> songs, string selectedSongId)
        {
            if (compactSongDropdown == null)
            {
                return;
            }

            isUpdatingSongDropdown = true;
            compactSongDropdown.ClearOptions();
            compactSongIds.Clear();

            var options = new List<Dropdown.OptionData>();
            var selectedIndex = 0;
            if (songs != null)
            {
                for (var i = 0; i < songs.Count; i++)
                {
                    var song = songs[i];
                    if (song == null || string.IsNullOrEmpty(song.songId))
                    {
                        continue;
                    }

                    compactSongIds.Add(song.songId);
                    options.Add(new Dropdown.OptionData(song.EffectiveDisplayName));
                    if (string.Equals(song.songId, selectedSongId, StringComparison.Ordinal))
                    {
                        selectedIndex = options.Count - 1;
                    }
                }
            }

            if (options.Count == 0)
            {
                options.Add(new Dropdown.OptionData("No songs"));
                compactSongDropdown.interactable = false;
                SetButtonState(compactPrevSongButton, false);
                SetButtonState(compactNextSongButton, false);
                compactSongDropdown.options = options;
                compactSongDropdown.value = 0;
                compactSongDropdown.RefreshShownValue();
                isUpdatingSongDropdown = false;
                return;
            }

            compactSongDropdown.interactable = true;
            SetButtonState(compactPrevSongButton, compactSongIds.Count > 1);
            SetButtonState(compactNextSongButton, compactSongIds.Count > 1);
            compactSongDropdown.options = options;
            compactSongDropdown.value = Mathf.Clamp(selectedIndex, 0, options.Count - 1);
            compactSongDropdown.RefreshShownValue();
            isUpdatingSongDropdown = false;
            ClearUiSelection();
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                QuitRequested?.Invoke();
            }

            if (Input.GetKeyDown(KeyCode.Tab))
            {
                isDebugVisible = !isDebugVisible;
                if (debugPanel != null)
                {
                    debugPanel.SetActive(isDebugVisible);
                }
            }
        }

        private static GameObject CreateDebugPanel(Transform parent)
        {
            var panelGo = new GameObject("Panel", typeof(Image));
            panelGo.transform.SetParent(parent, false);

            var rectTransform = panelGo.GetComponent<RectTransform>();
            rectTransform.anchorMin = new Vector2(0f, 1f);
            rectTransform.anchorMax = new Vector2(0f, 1f);
            rectTransform.pivot = new Vector2(0f, 1f);
            rectTransform.anchoredPosition = new Vector2(24f, -24f);
            rectTransform.sizeDelta = new Vector2(780f, 730f);

            var image = panelGo.GetComponent<Image>();
            image.color = new Color(0f, 0f, 0f, 0.78f);
            return panelGo;
        }

        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null)
            {
                return;
            }

            var existing = FindObjectOfType<EventSystem>();
            if (existing != null)
            {
                return;
            }

            var eventSystemGo = new GameObject("DanceDebugEventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
            DontDestroyOnLoad(eventSystemGo);
        }

        private static void ClearUiSelection()
        {
            if (EventSystem.current == null)
            {
                return;
            }

            EventSystem.current.SetSelectedGameObject(null);
        }

        private static GameObject CreateCompactPanel(Transform parent)
        {
            var panelGo = new GameObject("CompactPanel", typeof(Image));
            panelGo.transform.SetParent(parent, false);

            var rectTransform = panelGo.GetComponent<RectTransform>();
            rectTransform.anchorMin = new Vector2(1f, 0f);
            rectTransform.anchorMax = new Vector2(1f, 0f);
            rectTransform.pivot = new Vector2(1f, 0f);
            rectTransform.anchoredPosition = new Vector2(-28f, 28f);
            rectTransform.sizeDelta = new Vector2(420f, 246f);

            var image = panelGo.GetComponent<Image>();
            image.color = new Color(0f, 0f, 0f, 0.46f);
            return panelGo;
        }

        private void HandleSongDropdownChanged(int index)
        {
            if (isUpdatingSongDropdown)
            {
                return;
            }

            if (index < 0 || index >= compactSongIds.Count)
            {
                return;
            }

            Debug.Log("UI song selection changed to: " + compactSongIds[index]);
            SongSelectionChanged?.Invoke(compactSongIds[index]);
        }

        private void SelectRelativeSong(int delta)
        {
            if (compactSongDropdown == null || compactSongIds.Count <= 1)
            {
                return;
            }

            var nextIndex = compactSongDropdown.value + delta;
            if (nextIndex < 0)
            {
                nextIndex = compactSongIds.Count - 1;
            }
            else if (nextIndex >= compactSongIds.Count)
            {
                nextIndex = 0;
            }

            compactSongDropdown.value = nextIndex;
            compactSongDropdown.RefreshShownValue();
        }

        private static Dropdown CreateDropdown(Transform parent, Vector2 anchoredPosition, Vector2 size, Action<int> onValueChanged)
        {
            var root = new GameObject("SongDropdown", typeof(Image), typeof(Dropdown));
            root.transform.SetParent(parent, false);
            var rootRect = root.GetComponent<RectTransform>();
            rootRect.anchorMin = new Vector2(0f, 1f);
            rootRect.anchorMax = new Vector2(0f, 1f);
            rootRect.pivot = new Vector2(0f, 1f);
            rootRect.anchoredPosition = anchoredPosition;
            rootRect.sizeDelta = size;

            var rootImage = root.GetComponent<Image>();
            rootImage.color = new Color(0.15f, 0.15f, 0.15f, 0.95f);

            var dropdown = root.GetComponent<Dropdown>();
            dropdown.targetGraphic = rootImage;
            var dropdownNavigation = dropdown.navigation;
            dropdownNavigation.mode = Navigation.Mode.None;
            dropdown.navigation = dropdownNavigation;

            var caption = CreateText(root.transform, "Select Song", new Vector2(10f, -4f), new Vector2(size.x - 36f, 24f), 16);
            caption.alignment = TextAnchor.MiddleLeft;
            caption.horizontalOverflow = HorizontalWrapMode.Overflow;
            caption.verticalOverflow = VerticalWrapMode.Truncate;

            var arrow = CreateText(root.transform, "▼", new Vector2(size.x - 24f, -4f), new Vector2(18f, 24f), 15);
            arrow.alignment = TextAnchor.MiddleCenter;

            var template = new GameObject("Template", typeof(Image), typeof(ScrollRect));
            template.transform.SetParent(root.transform, false);
            var templateRect = template.GetComponent<RectTransform>();
            templateRect.anchorMin = new Vector2(0f, 0f);
            templateRect.anchorMax = new Vector2(1f, 0f);
            templateRect.pivot = new Vector2(0.5f, 1f);
            templateRect.anchoredPosition = new Vector2(0f, 2f);
            templateRect.sizeDelta = new Vector2(0f, 140f);
            var templateImage = template.GetComponent<Image>();
            templateImage.color = new Color(0.1f, 0.1f, 0.1f, 0.98f);
            var scrollRect = template.GetComponent<ScrollRect>();
            template.SetActive(false);

            var viewport = new GameObject("Viewport", typeof(Image), typeof(Mask));
            viewport.transform.SetParent(template.transform, false);
            var viewportRect = viewport.GetComponent<RectTransform>();
            viewportRect.anchorMin = Vector2.zero;
            viewportRect.anchorMax = Vector2.one;
            viewportRect.offsetMin = Vector2.zero;
            viewportRect.offsetMax = Vector2.zero;
            var viewportImage = viewport.GetComponent<Image>();
            viewportImage.color = new Color(1f, 1f, 1f, 0.02f);
            viewport.GetComponent<Mask>().showMaskGraphic = false;

            var content = new GameObject("Content", typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(ContentSizeFitter));
            content.transform.SetParent(viewport.transform, false);
            var contentRect = content.GetComponent<RectTransform>();
            contentRect.anchorMin = new Vector2(0f, 1f);
            contentRect.anchorMax = new Vector2(1f, 1f);
            contentRect.pivot = new Vector2(0.5f, 1f);
            contentRect.anchoredPosition = Vector2.zero;
            contentRect.sizeDelta = new Vector2(0f, 28f);
            var layout = content.GetComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(0, 0, 0, 0);
            layout.spacing = 0f;
            layout.childControlHeight = true;
            layout.childControlWidth = true;
            layout.childForceExpandHeight = false;
            layout.childForceExpandWidth = true;
            var fitter = content.GetComponent<ContentSizeFitter>();
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            fitter.horizontalFit = ContentSizeFitter.FitMode.Unconstrained;

            var item = new GameObject("Item", typeof(Toggle));
            item.transform.SetParent(content.transform, false);
            var itemRect = item.GetComponent<RectTransform>();
            itemRect.anchorMin = new Vector2(0f, 1f);
            itemRect.anchorMax = new Vector2(1f, 1f);
            itemRect.pivot = new Vector2(0.5f, 1f);
            itemRect.sizeDelta = new Vector2(0f, 28f);

            var itemBackground = new GameObject("Item Background", typeof(Image));
            itemBackground.transform.SetParent(item.transform, false);
            var itemBackgroundRect = itemBackground.GetComponent<RectTransform>();
            itemBackgroundRect.anchorMin = Vector2.zero;
            itemBackgroundRect.anchorMax = Vector2.one;
            itemBackgroundRect.offsetMin = Vector2.zero;
            itemBackgroundRect.offsetMax = Vector2.zero;
            var itemBackgroundImage = itemBackground.GetComponent<Image>();
            itemBackgroundImage.color = new Color(0.18f, 0.18f, 0.18f, 0.96f);

            var checkmark = new GameObject("Item Checkmark", typeof(Image));
            checkmark.transform.SetParent(item.transform, false);
            var checkmarkRect = checkmark.GetComponent<RectTransform>();
            checkmarkRect.anchorMin = new Vector2(0f, 0.5f);
            checkmarkRect.anchorMax = new Vector2(0f, 0.5f);
            checkmarkRect.pivot = new Vector2(0f, 0.5f);
            checkmarkRect.anchoredPosition = new Vector2(8f, 0f);
            checkmarkRect.sizeDelta = new Vector2(12f, 12f);
            var checkmarkImage = checkmark.GetComponent<Image>();
            checkmarkImage.color = new Color(0.8f, 0.8f, 0.8f, 1f);

            var itemLabel = CreateText(item.transform, "Option", new Vector2(28f, -2f), new Vector2(size.x - 40f, 24f), 16);
            itemLabel.alignment = TextAnchor.MiddleLeft;
            itemLabel.horizontalOverflow = HorizontalWrapMode.Overflow;
            itemLabel.verticalOverflow = VerticalWrapMode.Truncate;

            var toggle = item.GetComponent<Toggle>();
            toggle.targetGraphic = itemBackgroundImage;
            toggle.graphic = checkmarkImage;
            toggle.isOn = true;
            var toggleNavigation = toggle.navigation;
            toggleNavigation.mode = Navigation.Mode.None;
            toggle.navigation = toggleNavigation;

            scrollRect.content = contentRect;
            scrollRect.viewport = viewportRect;
            scrollRect.horizontal = false;
            scrollRect.vertical = true;
            scrollRect.movementType = ScrollRect.MovementType.Clamped;

            dropdown.template = templateRect;
            dropdown.captionText = caption;
            dropdown.itemText = itemLabel;
            dropdown.onValueChanged.AddListener(index => onValueChanged?.Invoke(index));
            return dropdown;
        }

        private static Text CreateText(Transform parent, string initialText, Vector2 anchoredPosition, Vector2 size, int fontSize)
        {
            var go = new GameObject("StatusText", typeof(Text));
            go.transform.SetParent(parent, false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0f, 1f);
            rect.anchorMax = new Vector2(0f, 1f);
            rect.pivot = new Vector2(0f, 1f);
            rect.anchoredPosition = anchoredPosition;
            rect.sizeDelta = size;

            var text = go.GetComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            text.alignment = TextAnchor.UpperLeft;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            text.color = Color.white;
            text.text = initialText;
            text.raycastTarget = false;
            return text;
        }

        private void UpdateButtonInteractivity(DanceDebugState state)
        {
            var canStart = !state.IsBusy && !state.IsPlaying;
            var canPause = !state.IsBusy && state.IsPlaying;
            SetButtonState(startButton, canStart);
            SetButtonState(compactStartButton, canStart);
            SetButtonState(pauseButton, canPause);
            SetButtonState(compactPauseButton, canPause);
            if (compactSongDropdown != null)
            {
                compactSongDropdown.interactable = !state.IsBusy && compactSongIds.Count > 0;
            }

            if (compactPrevSongButton != null)
            {
                SetButtonState(compactPrevSongButton, !state.IsBusy && compactSongIds.Count > 1);
            }

            if (compactNextSongButton != null)
            {
                SetButtonState(compactNextSongButton, !state.IsBusy && compactSongIds.Count > 1);
            }
        }

        private static void SetButtonState(Button button, bool interactable)
        {
            if (button == null)
            {
                return;
            }

            button.interactable = interactable;
            var image = button.GetComponent<Image>();
            if (image != null)
            {
                image.color = interactable
                    ? new Color(0.18f, 0.18f, 0.18f, 1f)
                    : new Color(0.12f, 0.12f, 0.12f, 0.55f);
            }
        }

        private static Button CreateButton(Transform parent, string label, Vector2 anchoredPosition, Action onClick)
        {
            return CreateButton(parent, label, anchoredPosition, onClick, new Vector2(160f, 36f));
        }

        private static Button CreateButton(Transform parent, string label, Vector2 anchoredPosition, Action onClick, Vector2 size)
        {
            var buttonGo = new GameObject(label, typeof(Image), typeof(Button));
            buttonGo.transform.SetParent(parent, false);
            var rect = buttonGo.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0f, 1f);
            rect.anchorMax = new Vector2(0f, 1f);
            rect.pivot = new Vector2(0f, 1f);
            rect.anchoredPosition = anchoredPosition;
            rect.sizeDelta = size;

            var image = buttonGo.GetComponent<Image>();
            image.color = new Color(0.18f, 0.18f, 0.18f, 1f);

            var button = buttonGo.GetComponent<Button>();
            var buttonNavigation = button.navigation;
            buttonNavigation.mode = Navigation.Mode.None;
            button.navigation = buttonNavigation;
            button.onClick.AddListener(() => onClick?.Invoke());

            var text = CreateText(buttonGo.transform, label, new Vector2(10f, -6f), new Vector2(size.x - 20f, 24f), 16);
            text.fontSize = 16;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = Color.white;

            return button;
        }
    }
}
