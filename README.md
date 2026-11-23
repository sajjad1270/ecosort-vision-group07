# EcoSort Vision — Streamlit Deployment

**DIU Group 07 · CSE437 · Energy & Environment**

This is the permanent-URL version of our EcoSort Vision waste-image classification demo. It uses the **original trained MobileNetV2 checkpoint** from Milestone 4 — not a pretrained public classifier and not retraining. It predicts nine RealWaste categories.

## Files you must upload together to GitHub

1. `app.py` — Streamlit interface.
2. `model_utils.py` — real model loading and image preprocessing.
3. `mobilenet_v2_best.pth` — trained model weights (approximately 15.6 MiB).
4. `requirements.txt` — dependencies (CPU-only PyTorch).
5. `README.md` — project information.

> Upload the **extracted five files**, not the outer ZIP. They must all be at the **top level** of your GitHub repository. No Colab session or raw dataset ZIP is needed.

## বাংলা — Google Chrome দিয়ে ফ্রি Deploy করার নিয়ম

**1.** ZIP download করে **Extract All** করো। ভেতরে ওপরের পাঁচটি ফাইল থাকবে।

**2.** [GitHub](https://github.com/signup) account তৈরি/Sign in করো। তারপর [New repository](https://github.com/new) খুলে Repository name দাও `ecosort-vision-group07`, visibility `Public`, এবং **Create repository** চাপো।

**3.** Repo-র **Add file → Upload files** (অথবা নতুন repo-তে **uploading an existing file**) চাপো। Extract করা **পাঁচটি ফাইল** select করে upload করো। সব ফাইল root-এ আছে দেখে **Commit changes** চাপো। `mobilenet_v2_best.pth` ফাইলটির size প্রায় 16 MB; GitHub web uploader-এ 25 MiB-এর কম ফাইল upload করা যায়।

**4.** [Streamlit Community Cloud](https://share.streamlit.io/) খোলো। **Continue with GitHub** দিয়ে login করো; GitHub permission চাইলে Authorize করো।

**5.** **Create app → Yup, I have an app** বেছে নাও। Repository `YOUR_USERNAME/ecosort-vision-group07`, Branch `main`, Main file path `app.py` দাও। চাইলে App URL name `ecosort-vision-group07` (available হলে) দাও। **Advanced settings → Python 3.11** নির্বাচন করা সুপারিশ করা হয়। **Deploy** চাপো।

**6.** Deploy সফল হলে `https://YOUR-CHOSEN-NAME.streamlit.app` লিংক পাবে। ছবি upload করে **Classify Image** চাপো। Result ও Top 3 predictions দেখা গেলে Screenshot নাও।

**7.** স্যারকে Website link, GitHub repository link এবং 2–3টি screenshot দাও। Colab-এর temporary Gradio URL-এর পরিবর্তে Streamlit link ব্যবহার করো।

## Problem solving

- `FileNotFoundError: mobilenet_v2_best.pth` → `.pth` file repo-র root-এ `app.py`-এর পাশে upload করো।
- `ModuleNotFoundError: torchvision` বা `torch` → `requirements.txt` repo root-এ আছে কিনা দেখো; Streamlit **Manage app → Logs** দেখো।
- `Error installing requirements` → app-র Python version **3.11** সেট করো এবং Streamlit Reboot/Redeploy করো।
- `App is sleeping` → visitor খুললে আবার চালু হতে পারে। Free hosting-এর জন্য uninterrupted 24/7 uptime নিশ্চিত নয়।
- `Model loads, but prediction differs` → verify image format/preprocessing and use original checkpoint. This code matches Group 07's Milestone 6 preprocessing (224×224 RGB/ImageNet normalization).

## Local quick test (optional)

In a terminal after installing Python 3.11:
```sh
pip install -r requirements.txt
streamlit run app.py
```

For a local-only app, open `http://localhost:8501`.

## Research and academic notes

- Dataset: [RealWaste — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- Original Milestone 4 RealWaste held-out test accuracy: 84.99%; macro F1: 85.00%. These are measurements from the original course experiment, not a new benchmark.
- The confidence threshold (0.54) is an exploratory validation value. It is not proof of reliability on new real-world photos.
- Model source: original Group 07 `mobilenet_v2_best.pth` checkpoint.

