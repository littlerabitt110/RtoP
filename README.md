# 🧠 From Recursion to Prediction

This project presents a novel approach to predicting the computational effort of solving the **Traveling Salesman Problem (TSP)** using **backtracking algorithms**, through **machine learning techniques**. 

Instead of brute-force execution, our framework accurately estimates runtime **before** the algorithm is run — saving both time and compute resources. 🕒⚙️

This can support:
- 🚀 Intelligent solver selection  
- 📈 Runtime estimation  
- 🧠 Resource-aware scheduling  

Our experiments demonstrate that regression-based ML models achieve **99% prediction accuracy** for estimating computational workload in TSP backtracking solvers.
## 📊 Results Visualization

![TSP Backtracking Prediction Results](./figures/figure1_new.jpg)


---

### 📌 Key Contributions  
```markdown
## 📌 Key Contributions

- ✅ Proposed an ML-based pipeline to predict TSP solver runtime
- ✅ Engineered features from TSP instances for model training
- ✅ Benchmarked 12 ML models across categories
- ✅ Found regression models best capture backtracking effort
- ✅ Achieved up to 99% accuracy in runtime prediction
