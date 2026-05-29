# PupsN_Edge - TODO & Next Steps

You stopped here because your battery was low. I have prepared all the necessary scripts to finish the setup.

### 1. Export AI Models to ONNX
The scripts are ready to convert your `yolo26n.pt` and the Hugging Face `OSNet` into the correct `.onnx` formats required by the system.

When you are back online, simply open a terminal in this directory and run:

```bash
bash run_exports.sh
```

**What this script does:**
1. Activates your Python virtual environment.
2. Installs required dependencies (`ultralytics`, `torchreid`, etc.).
3. Runs `export_yolo.py` which takes your `yolo26n.pt` and exports it directly to `models/yolo26nano.onnx`.
4. Runs `export_osnet.py` which clones the Torchreid repository, downloads the OSNet weights from Hugging Face (`kaiyangzhou/osnet`), and exports it directly to `models/osnet_x1_0.onnx`.

### 2. Verify Models
After the script finishes, ensure both models are present in the `models/` directory:
- `models/yolo26nano.onnx`
- `models/osnet_x1_0.onnx`

### 3. Run the System
Start the main edge camera server:
```bash
source venv/bin/activate
python app.py
```

### 4. Push Models to GitHub (Optional)
Because I updated your `.gitignore` to allow `.onnx` tracking, you can push the converted models so you don't have to compile them again:
```bash
git add models/*.onnx
git commit -m "Add compiled YOLO and OSNet ONNX models"
git push origin master
```
