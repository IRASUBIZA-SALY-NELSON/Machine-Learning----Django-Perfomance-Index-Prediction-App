# Django ML Prediction API – Student Performance Index

## 📚 Assignment Overview

This project demonstrates an end-to-end Django application that:
1. ✅ Creates a Django REST API project
2. ✅ Stores student performance data in a database
3. ✅ Exposes data via REST API endpoints
4. ✅ Trains & deploys a machine learning model
5. ✅ Predicts performance for new student data

**Dataset Target**: Performance Index (Regression Task)

---

## 🚀 Project Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install django djangorestframework pandas scikit-learn joblib

# 3. Navigate to project directory
cd student_ml

# 4. Run migrations
python manage.py makemigrations
python manage.py migrate
```

---

## 📂 Project Structure

```
student_ml/
├── performance/                  # Main app
│   ├── migrations/              # Database migrations
│   ├── models.py                # StudentPerformance model
│   ├── serializers.py           # DRF serializers
│   ├── views.py                 # API views
│   ├── urls.py                  # App URL routing
│   ├── load_data.py             # Data loading script
│   ├── train_model.py           # ML training script
│   └── model.pkl                # Trained model (generated)
├── student_ml/                  # Project settings
│   ├── settings.py              # Django settings
│   └── urls.py                  # Main URL routing
├── StudentPerformance.csv       # Dataset
├── db.sqlite3                   # SQLite database
└── manage.py                    # Django management script
```

---

## 🗄️ Database Model

### StudentPerformance Model

```python
class StudentPerformance(models.Model):
    hours_studied = models.IntegerField()
    previous_scores = models.IntegerField()
    extracurricular = models.BooleanField()
    sleep_hours = models.IntegerField()
    sample_papers = models.IntegerField()
    performance_index = models.FloatField()
```

**Fields Description**:
- `hours_studied`: Number of hours the student studies daily
- `previous_scores`: Student's previous exam scores
- `extracurricular`: Whether student participates in extracurricular activities (Yes/No → True/False)
- `sleep_hours`: Average hours of sleep per night
- `sample_papers`: Number of sample papers practiced
- `performance_index`: **Target variable** - Overall performance score (0-100)

---

## 📊 Loading Data into Database

### Step 1: Prepare CSV File
Ensure `StudentPerformance.csv` is in the project root directory.

### Step 2: Load Data
```bash
python manage.py shell
```

```python
>>> from performance.load_data import run
>>> run()
Successfully loaded 36 records into the database
```

This script:
- Reads the CSV file
- Converts "Yes"/"No" to Boolean values
- Removes duplicates
- Inserts data into the database

---

## 🤖 Training the Machine Learning Model

### Step 1: Train the Model
```bash
python manage.py shell
```

```python
>>> from performance.train_model import train
>>> train()
Model Performance:
  MSE: 4.31
  R² Score: 0.9884
Model trained and saved to performance/model.pkl
```

### Model Details
- **Algorithm**: Random Forest Regressor
- **Estimators**: 200 trees
- **Train/Test Split**: 80/20
- **Features**: All columns except Performance Index
- **Target**: Performance Index

### Model Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures average squared difference between predictions and actual values
- **R² Score**: Coefficient of determination (0.9884 = 98.84% variance explained)

---

## 🌐 API Endpoints

### Prediction Endpoint

**URL**: `POST /api/predict/`

**Request Body**:
```json
{
  "hours_studied": 6,
  "previous_scores": 78,
  "extracurricular": true,
  "sleep_hours": 7,
  "sample_papers": 3
}
```

**Response**:
```json
{
  "predicted_performance_index": 72.45,
  "input_data": {
    "hours_studied": 6,
    "previous_scores": 78,
    "extracurricular": true,
    "sleep_hours": 7,
    "sample_papers": 3
  }
}
```

### Testing the API

#### Using cURL:
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "hours_studied": 6,
    "previous_scores": 78,
    "extracurricular": true,
    "sleep_hours": 7,
    "sample_papers": 3
  }'
```

#### Using Python requests:
```python
import requests

url = 'http://127.0.0.1:8000/api/predict/'
data = {
    'hours_studied': 6,
    'previous_scores': 78,
    'extracurricular': True,
    'sleep_hours': 7,
    'sample_papers': 3
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 🏃 Running the Server

```bash
python manage.py runserver
```

Server will start at: `http://127.0.0.1:8000/`

API endpoint available at: `http://127.0.0.1:8000/api/predict/`

---

## 🔍 Machine Learning Pipeline Explained

### 1. Problem Definition
- **Task Type**: Regression (predicting continuous values)
- **Goal**: Predict student performance index based on study habits

### 2. Data Collection
- Dataset: StudentPerformance.csv with student records

### 3. Data Exploration (EDA)
- Statistical analysis using `.describe()`
- Correlation analysis to identify important features
- Visualization of data distribution

### 4. Data Preprocessing
- **Duplicate Removal**: `df.drop_duplicates()`
- **Categorical Encoding**: Convert "Yes"/"No" to 1/0
- **Missing Values**: Check and handle null values

### 5. Feature Selection
- Selected features:
  - Hours Studied
  - Previous Scores
  - Extracurricular Activities
  - Sleep Hours
  - Sample Question Papers Practiced

### 6. Data Splitting
- Training Set: 80%
- Testing Set: 20%

### 7. Model Selection
- **Algorithm**: Random Forest Regressor
- **Why?**: Handles non-linear relationships well, robust to outliers

### 8. Model Training
- Fit model on training data
- Learn patterns between features and target

### 9. Model Evaluation
- **R² Score**: 0.9884 (98.84% accuracy)
- **MSE**: 4.31 (low prediction error)

### 10. Model Deployment
- Save model using `joblib`
- Load in Django view for predictions

---

## ❓ Assignment Questions & Answers

### Question a: What is the purpose of joblib?

**Answer**:

`joblib` is a Python library used for **efficient serialization and deserialization of Python objects**, particularly for machine learning models.

**Key Purposes**:
1. **Model Persistence**: Save trained models to disk so they don't need to be retrained every time
2. **Efficient Storage**: Uses optimized binary format for large numpy arrays
3. **Fast Loading**: Quick deserialization for production use
4. **Memory Efficiency**: Handles large objects better than standard pickle

**In Our Project**:
```python
# Saving the model
joblib.dump(model, 'performance/model.pkl')

# Loading the model
model = joblib.load('performance/model.pkl')
```

### Question b: Mention other libraries that can achieve the same as joblib

**Answer**:

Several libraries can serialize/deserialize ML models:

1. **pickle** (Python Standard Library)
   ```python
   import pickle

   # Save
   with open('model.pkl', 'wb') as f:
       pickle.dump(model, f)

   # Load
   with open('model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```
   - **Pros**: Built-in, no installation needed
   - **Cons**: Slower for large numpy arrays

2. **dill** (Enhanced pickle)
   ```python
   import dill

   with open('model.pkl', 'wb') as f:
       dill.dump(model, f)
   ```
   - **Pros**: Can serialize more Python objects than pickle
   - **Cons**: External dependency

3. **cloudpickle**
   ```python
   import cloudpickle

   cloudpickle.dump(model, open('model.pkl', 'wb'))
   ```
   - **Pros**: Better for cloud deployments
   - **Cons**: Slightly slower

4. **ONNX** (Open Neural Network Exchange)
   ```python
   from skl2onnx import convert_sklearn

   onnx_model = convert_sklearn(model, initial_types=[...])
   ```
   - **Pros**: Platform-independent, language-agnostic
   - **Cons**: More complex setup

5. **HDF5** (via h5py)
   ```python
   import h5py

   # For deep learning models (Keras/TensorFlow)
   model.save('model.h5')
   ```
   - **Pros**: Efficient for large datasets
   - **Cons**: Primarily for neural networks

**Comparison Table**:

| Library | Speed | Size Efficiency | Ease of Use | Best For |
|---------|-------|----------------|-------------|----------|
| joblib | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Scikit-learn models |
| pickle | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Small models |
| dill | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Complex objects |
| cloudpickle | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Cloud deployment |
| ONNX | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Cross-platform |

**Recommendation**: For scikit-learn models, **joblib is the best choice** due to its optimization for numpy arrays and ease of use.

---

## 🚀 Deployment Notes

### Production Recommendations:

1. **Database**:
   - Switch from SQLite to PostgreSQL or MySQL
   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.postgresql',
           'NAME': 'student_performance_db',
           'USER': 'your_user',
           'PASSWORD': 'your_password',
           'HOST': 'localhost',
           'PORT': '5432',
       }
   }
   ```

2. **Model Storage**:
   - Store `model.pkl` outside the repository
   - Use cloud storage (AWS S3, Google Cloud Storage)
   - Version your models

3. **Security**:
   - Set `DEBUG = False`
   - Use environment variables for secrets
   - Add authentication to API endpoints

4. **Performance**:
   - Use caching (Redis)
   - Implement rate limiting
   - Load model once at startup (not per request)

5. **Monitoring**:
   - Log predictions for model performance tracking
   - Implement error tracking (Sentry)
   - Monitor API response times

---

## 📝 Key Learnings

### ML Pipeline Steps:
1. ✅ Problem Definition
2. ✅ Data Collection
3. ✅ Data Exploration (EDA)
4. ✅ Data Preprocessing
5. ✅ Feature Selection
6. ✅ Data Splitting
7. ✅ Model Selection
8. ✅ Model Training
9. ✅ Model Evaluation
10. ✅ Model Deployment

### Django Integration:
- ✅ Created REST API with DRF
- ✅ Integrated ML model with Django views
- ✅ Implemented data persistence
- ✅ Built production-ready endpoints

---

## 🎯 Testing Examples

### Example 1: High Performer
```json
{
  "hours_studied": 9,
  "previous_scores": 95,
  "extracurricular": true,
  "sleep_hours": 8,
  "sample_papers": 7
}
```
Expected: High performance index (~90+)

### Example 2: Average Performer
```json
{
  "hours_studied": 5,
  "previous_scores": 70,
  "extracurricular": false,
  "sleep_hours": 6,
  "sample_papers": 4
}
```
Expected: Medium performance index (~55-65)

### Example 3: Low Performer
```json
{
  "hours_studied": 2,
  "previous_scores": 45,
  "extracurricular": false,
  "sleep_hours": 4,
  "sample_papers": 1
}
```
Expected: Low performance index (~20-30)

---

## 🛠️ Troubleshooting

### Issue: "Model not loaded"
**Solution**: Train the model first using `train_model.py`

### Issue: "No module named 'performance'"
**Solution**: Ensure 'performance' is in INSTALLED_APPS

### Issue: CSV file not found
**Solution**: Place StudentPerformance.csv in the project root (student_ml/)

### Issue: Migration errors
**Solution**:
```bash
python manage.py makemigrations performance
python manage.py migrate
```

---

## 📚 References

- Django Documentation: https://docs.djangoproject.com/
- Django REST Framework: https://www.django-rest-framework.org/
- Scikit-learn: https://scikit-learn.org/
- Joblib: https://joblib.readthedocs.io/

---

## 👨‍💻 Author
**Student Performance Prediction Assignment**
*Machine Learning & Django Integration Project*

---

## 📄 License
This project is created for educational purposes as part of an ML assignment.
# Machine-Learning----Django-Perfomance-Index-Prediction-App
