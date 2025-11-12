# Symptom-Based Diagnostic Decision Support AI System (DDSAS)

A machine learning project predicting the likelihood of chronic diseases based on patient symptoms and health metrics. This end-to-end system demonstrates MLOps principles, with data preprocessing, model training, MLflow experiment tracking, and a local Streamlit UI demo.  

It provides a **data-driven decision support tool** for healthcare settings.

---

# Business Objective

The **DiseaseFeatureClassifiers** delivers significant value to all stakeholders in the healthcare ecosystem. By leveraging **AI-driven symptom analysis**, the system **increases diagnostic precision and productivity**, reducing the cognitive and administrative burden on **frontline healthcare professionals**. It **improves patient outcomes** through timely and accurate risk assessment, while offering a **suggestive approach to reduce operational costs** for clinics and hospitals by optimizing workflows and minimizing unnecessary tests or interventions. Additionally, the tool helps **free up critical resources**, including emergency room capacity, enabling healthcare institutions to focus on patients with the most urgent needs, thereby enhancing **overall efficiency and cost-effectiveness**.


---

## 🎯 Stakeholders

The stakeholders of DDSAS include:

- **Physicians & Healthcare Providers:** Aid in disease diagnosis, improve accuracy, and reduce errors.  
- **Patients:** Benefit from timely and accurate diagnosis for better health outcomes.  
- **Healthcare Institutions & Clinics:** Improve workflows, standardize care, and reduce operational costs.  
- **Health IT & Data Scientists:** Build, track, and maintain ML models efficiently.  
- **Regulatory Bodies:** Ensure compliance with safety, ethics, and medical standards.  
- **Insurance Companies:** Optimize cost-effectiveness by minimizing misdiagnosis.  
- **Medical Educators & Researchers:** Train professionals and advance diagnostic methods.  
- **Patients’ Families & Caregivers:** Indirectly benefit from improved care decisions.  

This project helps all stakeholders make **informed, timely decisions** in healthcare.

---

## 🧩 Folder Structure

```
DiseaseFeatureClassifiers/
├── data #Raw Dataset
│   ├── Disease_Prediction.csv
├── src
│   ├── streamlit_app.py # Interactive demo UI
├── requirements.txt
└── README.md # Project overview
```
---


## 🤝 Team Members

[Junaid Khan](https://www.linkedin.com/in/junaid-devops)• 
[Adam Healey](https://www.linkedin.com/in/adam-healey) • 
[Ali Hyder](https://www.linkedin.com/in/ali-hyder-iith1041) • 
[Olga Nazarenko](https://www.linkedin.com/in/olga-nazarenko0) • 
[Pradeep Venkatesan](https://www.linkedin.com/in/pradeep-venkatesan-tech/)

---

## 📦 Technologies Used

| Component           | Technology       | Purpose                        |
|--------------------|----------------|--------------------------------|
| Data Handling       | pandas, numpy   | Clean & prepare dataset        |
| Machine Learning    | scikit-learn, xgboost | Train predictive models   |
| Experiment Tracking | MLflow          | Log experiments & model metrics|
| UI / Demo           | Streamlit       | Local interactive interface   |

---

## Sample Classification Models to Try

| Model                  | Description                                   | Assigned To         |
|------------------------|-----------------------------------------------|------------------|
| Logistic Regression    | Baseline probabilistic classifier             |    AH  |
| Random Forest          | Ensemble of decision trees, robust to overfitting |        ALH|
| XGBoost                | Gradient boosting, effective on tabular data | ON        |
| LightGBM               | Fast gradient boosting, handles large data   |   JK |
| SVM                    | Good for high-dimensional, complex boundaries | PV |
| Neural Networks (MLP)  | Deep learning for complex feature interactions |       |

---

## Sample Clustering Models to Try

| Model                  | Description                                   | Assigned To       |
|------------------------|-----------------------------------------------|----------------|
| KMeans                 | Partition-based clustering                     |   ON  |
| DBSCAN                 | Density-based, finds arbitrarily shaped clusters |      ALH|
| Agglomerative          | Hierarchical clustering                        | PV |


---

## ⚠️ Risks & Uncertainty

**Accuracy of Prognosis:**  
The reliability of predictions must be evaluated to ensure their correctness. In healthcare, incorrect predictions, such as false negatives and positives, could lead to inappropriate patient management and harm.

**Data Quality:**  
Inaccurate or incomplete data can lead to flawed predictions, impacting overall outcomes.To mitigate this risk, we will implement rigorous data validation processes and conduct regular audits to ensure data integrity.

**Model Bias:**  
Bias present in the data may skew results, necessitating thorough examination and adjustment. We will regularly review our model for biases and make necessary adjustments to ensure equitable treatment across diverse patient populations.

**Clinical Risk:**
It is crucial to assess the potential clinical risks associated with false predictions. A comprehensive evaluation of the impact on patient safety will be conducted as part of our testing process.

**Regulatory and Ethical Constraints:**
Compliance with regulatory and ethical standards is essential. This includes ensuring explainability and auditability of the model’s decisions, as well as safeguarding patient data privacy.

**Scope Definition:**
This project is designed as a support tool rather than an autonomous diagnostic system. Clarifying its scope will help manage expectations in clinical settings.

---

## 📊 Monitoring and Mitigation Plan

To manage these identified risks effectively, we will focus on:

**Data Quality Checks:**
Implement routine checks on data inputs to identify and address any inaccuracies promptly.

**Collaborative Feedback:**
Establish informal channels with healthcare professionals for ongoing feedback, facilitating minor adjustments based on practical insights.

**Basic User Guidelines:**
Provide simplified guidelines that outline best practices and limitations of the tool for users without extensive training requirements.

**Feedback Loop for Incidents:** 
Create a straightforward feedback process to report issues as they arise, allowing for quick responses without complex protocols.

---

## 📦 Demo

Video Link:
---


## ⚙️ Setup & Usage

1. **Clone the repository**
```bash
git clone https://github.com/jkhan2211/DiseaseFeatureClassifiers.git
cd DiseaseFeatureClassifiers
```

2. **Clone the repository**

```bash
pip install -r requirements.txt
```


