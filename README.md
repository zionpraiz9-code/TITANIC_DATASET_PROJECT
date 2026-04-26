# 🚢 Titanic Survival Prediction: A Generalization-Focused Approach

This project implements an end-to-end Machine Learning pipeline to predict passenger survival on the Titanic. The core objective was to solve the "Overfitting Problem"—moving from a model that simply memorizes the data to one that understands the underlying patterns of survival.

## 📊 Project Performance
* **Final Test Accuracy:** $82.68\%$
* **Model Stability:** $3.41\%$ Gap (Training vs. Test)
* **Cross-Validation:** $83.83\%$ Mean Accuracy across 10 folds
* **Model Type:** Optimized XGBoost Classifier

---

## 🛠️ The 6-Phase Engineering Workflow

### Phase 1: Data Forensics
Used `missingno` to map data gaps and analyzed survival correlations. It was discovered that **Sex**, **Pclass**, and **Title** were the strongest indicators of survival.

### Phase 2: Professional Data Cleaning
* **Intelligent Imputation:** Instead of using a flat average, missing ages were filled based on the passenger's **Title** (Master, Miss, Mr, etc.), which is a much more accurate proxy for age.
* **Feature Transformation:** Converted 'Cabin' into a binary 'Has_Cabin' feature to capture socio-economic advantages.

### Phase 3: Advanced Feature Engineering (31 Features)
Engineered features to capture "hidden" survival signals:
* **Title Grouping:** Simplified 17+ unique titles into 5 core categories.
* **Family Dynamics:** Created `FamilySize` and `IsAlone` to account for the "Women and Children First" protocol.
* **Interactions:** Created `Age_Class` to identify high-priority survival groups (wealthy children).

### Phase 4: Hyperparameter Tuning (The "Anti-Overfit" Strategy)
To bridge the initial $10.5\%$ performance gap, I implemented strict **Regularization** constraints:
* **Max Depth (3-5):** Limits the complexity of the trees to prevent memorization.
* **Gamma (0.5 - 2.0):** Acts as a "complexity penalty" for new branches.
* **Subsampling:** Trains on random slices of data to ensure the model is robust and not biased toward specific rows.

### Phase 5: Reliability Testing
Verified the model using **Stratified 10-Fold Cross-Validation**. This "Stress Test" confirmed the model performs consistently across different groups of passengers. 

*(Upload your CV Bar Chart here)*

### Phase 6: Model Interpretability (SHAP)
Using SHAP values, we can see the mathematical "logic" behind the predictions. The model correctly identifies that being female, being in 1st Class, and having a "Master" or "Miss" title were the highest drivers for survival.

---

## 📝 Technical Deep-Dive Notes

### 1. Why XGBoost?
XGBoost (Extreme Gradient Boosting) was chosen for its superior handling of tabular data and its built-in regularization. In a small dataset like the Titanic, models tend to overfit quickly; XGBoost’s ability to penalize complexity via `gamma` allowed me to "tame" the model until the Training and Test scores converged.

### 2. The Logic of "Title-Based" Imputation
A common mistake is filling missing ages with the median age of the whole ship ($\approx 28$). However, a passenger with the title **"Master"** is statistically much younger, while a **"Colonel"** is much older. By imputing based on `Title`, I reduced the noise in the `Age` feature.

### 3. Handling the "Generalization Gap"
The most critical part of this project was closing the performance gap. By capping `max_depth` at 5, I forced the model to stop creating "niche" rules that only applied to a few people, ensuring the logic was broad enough to apply to the hidden test set.

### 4. Mathematical Reliability
The Stratified 10-Fold CV yielded a low Standard Deviation ($\sigma = 0.0259$). This proves that the model's accuracy is not a result of a "lucky" data split, but a result of robust feature engineering.

---

## 🔍 Strategic Project Takeaways

### 💡 Key Insight: The "91% Illusion"
Initially, the model achieved over $91\%$ accuracy on training data—a "red flag" for overfitting. By intentionally reducing training accuracy to $86.1\%$, the model became better at predicting **new** data, resulting in a higher test score ($82.68\%$).

### 🚧 Challenges & Solutions
| Challenge | Solution |
| :--- | :--- |
| **High Variance** | Implemented `max_depth` constraints and `subsampling`. |
| **Missing Age Data** | Title-based Imputation to preserve age/status correlations. |
| **Categorical Complexity** | Used `ColumnTransformer` pipelines to prevent data leakage. |

### 🚀 Future Improvements
1.  **Ensemble Voting:** Combining XGBoost with a Logistic Regression and SVM for higher stability.
2.  **Deck Mapping:** Analyzing specific Cabin decks (A, B, C) to see if vertical location affected survival.
3.  **Ticket Grouping:** Investigating if passengers traveling on the same ticket survived as a unit.

---

## 📂 How to Use
1. **Clone the repo:** `git clone https://github.com/yourusername/titanic-project.git`
2. **Load the model:**
```python
import joblib
model = joblib.load('titanic_final_model.joblib')
# predictions = model.predict(new_data)
