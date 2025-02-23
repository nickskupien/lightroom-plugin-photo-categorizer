Lightroom Auto Tagger
=====================

Installation:
1. Extract the `LightroomAutoTag` folder into:
   - macOS: ~/Library/Application Support/Adobe/Lightroom/Modules/
   - Windows: C:\Users\YourName\AppData\Roaming\Adobe\Lightroom\Modules\

2. Open Lightroom Classic.
3. Go to File > Plugin Manager.
4. Click Add, select the `LightroomAutoTag` folder, and click Enable.

Usage:
1. Select photos in Lightroom.
2. Go to File > Plug-in Extras > Run AI Auto-Tagging.
3. The AI-generated tags will be added as keywords.

Setup:
- For **Local CLIP AI**: Install dependencies:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install pillow
  pip install git+https://github.com/openai/CLIP.git
  ```

  ```
  python3 -c "import torch; print('Torch:', torch.__version__)"
  python3 -c "import torchvision; print('Torchvision:', torchvision.__version__)"
  python3 -c "import clip; print('CLIP installed successfully')"
  ```

  Ensure `clip_tagging.py` is in the same directory.

- Symlink to lightroom plugins directory
  ```
  ln -s ~/Documents/__Documents__/Areas/Coding/lighroom-deep-sort/ ~/Library/Application\ Support/Adobe/Lightroom/Modules
  ```

Making Changes?
- Whenever you add new Lua files or significantly change the plugin folder structure,
  a Lightroom restart is often necessary to ensure it picks up the updated file paths
  and clears out any stale cache.


Enjoy AI-powered auto-tagging in Lightroom! 🚀
