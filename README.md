# 🐔 Poultry Disease Detection Using AI  
### *An Intelligent Deep Learning–Based Web Application for Poultry Health Monitoring*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black)
![Deployment](https://img.shields.io/badge/Deployed-Render-green)
![Status](https://img.shields.io/badge/Status-Live-success)

---

## 🚀 Live Application  
🌐 **Try the application here:**  
👉 **https://poultry-disease-detection-using-ai.onrender.com**

> ⏳ *Note:* Since this is hosted on a free cloud tier, the first prediction may take a few seconds due to cold start.

---

## 📌 Project Overview

Poultry farming plays a crucial role in the agricultural economy, but disease outbreaks such as **Coccidiosis**, **Newcastle Disease**, and **Salmonella** can lead to severe losses.  
This project presents an **AI-powered web application** that helps farmers and poultry health professionals **identify poultry diseases from images** using **Deep Learning and Transfer Learning techniques**.

The system enables users to upload poultry images and instantly receive disease predictions, helping in **early diagnosis, prevention, and better farm management**.

---

## 🎯 Key Features

✅ Image-based poultry disease prediction  
✅ Supports **4 major poultry health classes**  
✅ Real-time inference using a trained deep learning model  
✅ Simple, user-friendly web interface  
✅ Cloud-deployed and publicly accessible  
✅ Optimized for low-resource environments  

---

## 🧠 Diseases Detected

The model classifies poultry images into the following categories:

- 🦠 **Coccidiosis**
- ✅ **Healthy**
- 🦠 **New Castle Disease**
- 🦠 **Salmonella**

---

## 🏗️ System Architecture (High Level)

1. User uploads a poultry image via the web interface  
2. Image is preprocessed (resizing, normalization)  
3. Deep Learning model performs inference  
4. Predicted disease class is returned  
5. Result is displayed on the web page  

---

## 🧪 Machine Learning Approach

- **Model Type:** Convolutional Neural Network (CNN)
- **Technique:** Transfer Learning
- **Base Model:** MobileNet (pretrained on ImageNet)
- **Framework:** TensorFlow & Keras
- **Input Size:** 224×224 (optimized during deployment)
- **Output:** Multi-class classification

---

## 🛠️ Technology Stack

### 🔹 Backend & AI
- **Python 3.10**
- **TensorFlow 2.12**
- **Keras**
- **NumPy**
- **Pillow (Image Processing)**

### 🔹 Web Framework
- **Flask**
- **Jinja2 Templates**
- **Werkzeug**

### 🔹 Frontend
- **HTML5**
- **CSS3**
- **Responsive UI Design**

### 🔹 Deployment
- **Gunicorn (WSGI Server)**
- **Render Cloud Platform**

---

## ☁️ Deployment Details

- Hosted on **Render (Free Tier)**
- Configured with optimized Gunicorn settings
- Lazy model loading implemented to handle memory constraints
- Publicly accessible URL for real-time testing

---

## 📷 How to Use the Application

1. Open the live link  
2. Upload a poultry image  
3. Click **Predict**  
4. View the predicted disease result  

---

## 🧩 Use Cases

- Poultry farmers  
- Veterinary professionals  
- Agricultural researchers  
- Smart farming systems  
- Academic and educational demonstrations  

---

## ⚠️ Limitations

- Free cloud hosting may introduce cold-start delays  
- Prediction accuracy depends on image quality  
- Designed for educational and prototype-level usage  

---

## 🌱 Future Enhancements

- Add confidence score for predictions  
- Support real-time camera input  
- Mobile application integration  
- More disease categories  
- TensorFlow Lite optimization  
- Farmer advisory recommendations  

---

## 👨‍💻 Project Owner

**Karthik Reddy M**  
**🔗 GitHub:** [https://github.com/karthikredddy7github](https://github.com/karthikredddy7github) 

🎓 *Artificial Intelligence & Machine Learning Enthusiast* 

💡 *Focus: AI for Agriculture & Healthcare*

- If you want to contribute to his projects connect him on Linkedin.
- **🔗 LinkedIn:** [[https://in.linkedin.com/in/karthik4253](https://in.linkedin.com/in/karthik4253)]
---

## 🤝 Acknowledgements

- TensorFlow & Keras Team  
- Flask Community  
- Render Cloud Platform  
- Open-source contributors  

---

## 📜 License

This project is developed for **educational and academic purposes**.  
You are free to explore, learn, and build upon this work with proper attribution.

---

## ⭐ If you like this project

Please consider giving the repository a ⭐ on GitHub — it helps and motivates further development!

---

### 📩 Feedback & Suggestions
Feel free to open issues or suggest improvements.  
Happy coding! 🚀
