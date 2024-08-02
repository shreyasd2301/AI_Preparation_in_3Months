# AI Preparation in 3 Months
Prepare End to End AI &amp; be able to apply for Jobs 

- [Weekly Syllabus and tasks](weekly-schedule/weekly-schedule.md)

## Syallabus

- [Statistics](#statistics)
- [Linear Algebra](#linear-algebra)
- [Calculus](#calculus)
- [Programming](#programming)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Pt.1: Fundamentals](#machine-learning-pt1-fundamentals)
- [Machine Learning Pt.2: Algorithms](#machine-learning-pt2-algorithms)
- [Model Evaluation and Validation](#model-evaluation-and-validation)
- [Deep Learning](#deep-learning)
- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
- [Computer Vision (CV)](#computer-vision-cv)
- [Data Niches](#data-niches)
- [Deployment](#deployment)

---

## Statistics

Statistics can be thought of as the "foundational toolkit" for understanding and interpreting data. It's crucial for tasks like data analysis, building probabilistic models, and evaluating model performance and so on. To build a general intuition for the statistics required for data science, focus on the following areas:


![Illustration-of-a-bivariate-Gaussian-distribution-The-marginal-and-joint-probability_W640](https://github.com/user-attachments/assets/32db0836-d58e-40e5-978a-f8c55539e52a)


- **Descriptive Statistics**: Measures of Central Tendency (mean, median, mode), Measures of Dispersion (range, variance, standard deviation), skewness and kurtosis.
- **Inferential Statistics**: Hypothesis testing, Confidence intervals, P-values, t-tests, chi-square tests.
- **Probability Theory**: Basic probability, Conditional probability, Bayes' theorem.
- **Distributions**: Normal distribution, Binomial distribution, Poisson distribution.
- **Bayesian Statistics (optional)**: Bayesian inference, Prior and posterior probabilities.


📚 **References:**

---

## Linear algebra

Linear algebra is fundamental for understanding how data is represented and manipulated in machine learning algorithms. It forms the backbone of many ML techniques, especially in deep learning where operations on large matrices are common.

![Linear_subspaces_with_shading](https://github.com/user-attachments/assets/6c1ca9a4-c4dd-4cb5-ad60-af7ea59815ad)


- **Vectors & Vector Spaces**: Addition, subtraction, scalar multiplication, dot products, linear transformations.
- **Matrices & Matrix Operation**s: Basic concepts, addition, subtraction, multiplication, transposition, inversion.
- **Eigenvectors & Eigenvalues**: Understanding their significance in data transformations and dimensionality reduction (PCA).

📚 **References:**

---

## Calculus 

Calculus is essential for understanding the optimization techniques used in machine learning, particularly for gradient-based learning algorithms.

![matlab](https://github.com/user-attachments/assets/65cb4b6a-15f7-4d46-92a7-e9cef469f250)


- **Differential Calculus**: Derivatives, Partial derivatives, Gradients.
- **Integral Calculus**: Integration, Area under curves, Summations.
- **Multivariable Calculus**: Functions of multiple variables.

📚 **References:**

--- 

## Programming 

The importance of programming is probably self explanatory. Programming skills are essential for implementing machine learning algorithms, working with data and everything else (This roadmap mainly focuses on Python).

- **Python**: General syntax, Control flow, Functions, Classes, File handling, Loops and so on.
- **Essential Data Libraries**: NumPy, pandas, matplotlib, seaborn, OpenCV, NLTK, Spacy etc...
- **Tools**: Jupyter Notebooks, git/github, package/environment managers.

📚 **References:**

---

## Data Preprocessing

A simple analogy would be to think of data as the ingredients required to cook a meal. The better the quality the better the meal. "Your ML model is only as good as your data".

- **Data Cleaning**: Handling/Imputing missing values and Removing duplicates.
- **Detecting & Handling Outliers**: Visual Methods (Eg: Box Plots), Statistical Methods (Eg: Std, Z-score etc.), capping/flooring etc..
- **Data Transformation**: Normalization (Feature scaling), log normalization, Standardization, Encoding categorical variables.
- **Feature Engineering**: Creating new features (Domain Knowledge), Selecting relevant features.
- **Handling Imbalanced Data**: Resampling (Over-Sampling, Under-Sampling), SMOTE, Class weight assignments
- **Data Splitting**: Train-test split, Cross-validation.

📚 **References:**

---

## Machine Learning Pt.1: Fundamentals

Before going into machine learning algorithms, it is important to undertand the different terminologies and concepts that underly these algorithms (note that this section only introduces these concepts)

![overfitting-in-machine-learning](https://github.com/user-attachments/assets/091b34e6-656f-4514-89dd-8156b36a20c0)


- **Types of Models**: Supervised, Unsupervised, Reinforcement 
- **Bias and Variance**: Overfitting and Underfitting
- **Regularization**: L1, L2, Dropout and Early stopping
- **Cost Functions**: MSE (regression), Log loss/binary cross entropy (Binary Classification), categorical cross entropy (Multi-class classification with one-hot encoded features), sparse categorical cross entropy (Multi-class classifcation with integer features)
- **Optimizers**: Gradient descent, Stochastic gradient descent, small batch gradient descent, RMSprop, Adam

📚 **References:**

---

## Machine Learning Pt.2: Algorithms

The statistical learning algorithms that do all the work we associate "Ai" with

- **Supervised Algorithms**: Linear regression, Logistic regression, Decision trees, Random forests, Support vector machines, k-nearest neighbors
- **Unsupervised Algorithms**: Clustering (k-Means, Hierarchical), Dimensionality reduction (PCA, t-SNE, LDA), recommenders (collaborative and content-based) and anomaly detection (Variational autoencoders)
- **Ensemble Methods**: Stacking, Boosting (adaboost, XGboost etc.) and Bagging (bootstrap aggregation)
- **Reinforcement Learning**: Q-learning, Deep Q-networks, Policy gradient methods, RLHF etc..
- **Machine Learning Libraries**: Scikit-learn, Gymnasium (Reinforcement) etc.
- **Extra**: [Choosing a model](https://youtube.com/playlist?list=PLVZqlMpoM6kaeaNvBTyoKdhMpOZzXu6AS&si=EBnWyp1PNv3STXSv)

📚 **References:**

---

## Model Evaluation and Validation

It is important to understand how your model is performing and whether it would be useful in the real world. Eg: A model with high variance would not generalize well and would therefore not perform well on unseen data.

- **Metrics: Accuracy**, Precision, Recall, F1 Score, confusion matrix, ROC-AUC, MSE, R squared etc
- **Validation Techniques**: Cross-validation, Hyperparameter tuning (Grid search, Random search), learning curves
- **Model Explainability (Optional)**: SHAP
 
📚 **References:**

---

## Deep Learning

This is where it gets interesting. Ever wondered how tools like GPT or Midjourney work? this is where it all starts. The idea is to build deep networks of "neurons" or "perceptrons" that use non-linear activations to understand abstract concepts that would generally confuse standard statistical models.

![1_ZXAOUqmlyECgfVa81Sr6Ew](https://github.com/user-attachments/assets/0026e041-c729-42e3-b074-a398d6034156)


- **Neural Networks**: architecture, Perceptrons, Activation functions (non-saturated), weights & initialization, optimizers (SGD, RMSprop, Adam etc), cost/loss functions, biases, back/forward propagation, gradient clipping, vanishing/exploding gradients, batch normalization etc..
- **Convolutional Neural Networks (CNNs)**: convolutional layers, pooling layers, kernels, feature maps
- **Recurrent Neural Networks and variants (RNNs/LSTMs/GRUs)**: Working with sequential data like audio, text, time series etc..
- **Transformers**: Text generation, attention mechanism
- **Autoencoders**: Latent variable representation (encoding/decoding) & feature extraction
- **Generative Adversarial Networks (GANs)**: Generators & Discriminators
- **Deep Learning Libraries**: Keras, Tensorflow, Pytorch etc...

📚 **References:**

---

## Natural Language Processing (NLP)

As the name suggests, this part focuses on technologies that deal with natural langauge (primarily text data).

<img width="798" alt="Screen Shot 2024-07-26 at 2 07 41 PM" src="https://github.com/user-attachments/assets/75ad50f0-80bd-48ac-a48c-3f7631718226">


- **Text Preprocessing**: Tokenization, Lemmatization, Stemming, stop-word removal
- **Text Representation (Feature Extraction)**: N-grams, Bag of Words, TF-IDF, Word Embeddings (Word2Vec, GloVe), FastText, Cosine similarity
- **NLP Models**: Sentiment analysis, Named entity recognition (NER), Machine translation, Language models etc.
- **Model Optimization**: Quantization, Pruning, Fine tuning, Prompt tuning
- **Specialized Techniques**: Zero-shot learning, Few-shot learning, Prompt Engineering, sequence/tree of prompts, self supervised learning, semi-supervised learning, RAG, topic modeling (LDA, NMF)
- **Large Language Models(LLMs)**: Working with APIs, local/Open-Source LLMs, huggingface transformers, Langchain etc.

📚 **References:**

---

## Computer Vision (CV)

As the name suggests, this part focuses on technologies that deal with visual data (primarily images and video).

<img width="800" alt="Screen Shot 2024-07-26 at 2 06 20 PM" src="https://github.com/user-attachments/assets/ce46fb5c-fee4-46c6-8464-46d0bdd6e94e">


- **Image Preprocessing**: Resizing, Normalization, Augmentation, Noise Reduction
- **Feature Extraction**: Edge Detection, Histograms of Oriented Gradients (HOG), Keypoints and Descriptors (SIFT, SURF)
- **Image Representation**: [Convolutional](https://youtube.com/playlist?list=PLVZqlMpoM6kanmbatydXVhZpu3fkaTcbJ&si=qq9dWdAlIAxePwoF) Neural Networks (CNNs), Transfer Learning, Feature Maps
- **CV Models**: Object Detection (YOLO, Faster R-CNN), Image Classification (ResNet, VGG), Image Segmentation (U-Net, Mask R-CNN), Image Generation (GANs, VAEs), Stable Diffusion
- **Model Optimization**: Quantization, Pruning, Knowledge Distillation, Hyperparameter Tuning
- **Specialized Techniques**: Transfer Learning, Few-Shot Learning, Style Transfer, Image Super-Resolution, Zero-Shot Learning
- **Advanced Topics**:
  - **3D Vision**: Depth Estimation, 3D Reconstruction, and Stereo Vision.
  - **Video Analysis**: Action Recognition, Object Tracking, Video Segmentation.
- **Working with APIs and Tools**: OpenCV, TensorFlow, PyTorch, Pre-trained Models, Deployment (ONNX, TensorRT)

📚 **References:**

---

## Data Niches

This section covers more complex, abstract and niche data types.

<img width="546" alt="Screen Shot 2024-07-28 at 4 15 54 AM" src="https://github.com/user-attachments/assets/041f7658-12eb-4e85-8823-2cb4edb2bca1">



- **Time series Data**: Trends, seasonality, noise, cycles, stationarity, decomposition, differencing, autocorrelation, forecasting, libraries (Statsmodels, Prophet, PyFlux etc..) and models (ARIMA, Exponential smoothing, SARIMA etc)

- **Geospatial Data**: Coordinate systems (latitude, longitude), projections, shapefiles, GeoJSON, spatial joins, distance calculations, mapping libraries (GeoPandas, Folium, Shapely, etc.), and spatial analysis techniques (buffering, overlay, spatial clustering, etc.)

- **Network/graph Data**: Graph representation, nodes, edges, adjacency matrices, graph traversal algorithms (BFS, DFS), centrality measures (degree, closeness, betweenness, eigenvector),link measures, community detection, network visualization (NetworkX, Gephi, Cytoscape), deep graph library and PyTorch Geometric

- **Audio Data**: Feature extraction (MFCCs, spectrograms), audio preprocessing (normalization, noise reduction), signal processing, audio classification, libraries (Librosa, PyDub), and techniques (Fourier Transform, Short-Time Fourier Transform)

- **Multimodal Data**: Data integration (e.g., text and images, audio and video), feature fusion, modality alignment, joint representation learning, applications in recommendation systems, multi-modal sentiment analysis, and data fusion techniques.

- **Biological Data**: DNA sequences, protein structures, genomic variations, biological networks, molecular dynamics, libraries (Biopython, PySCeS), and techniques (sequence alignment, structural bioinformatics).

- **Astronomical Data**: Star catalogs, galaxy surveys, light curves, cosmic microwave background data, celestial object tracking, libraries (AstroPy, HEAsoft), and techniques (source extraction, cosmic event detection).


📚 **References:**

---

## Deployment

Deploying machine learning models to production.

- **Model Serving**: Flask, FastAPI, Streamlit, TensorFlow Serving
- **Persisting Models**: Pickle & Joblib
- **Containerization**: Docker, Kubernetes
- **Cloud Services**: AWS, Google Cloud, Azure and IBM Watson

📚 **References:**



#### In the coming days, you will be creating GitHub repositories for
- ML Projects
- DL projects
- NLP projects
- GenAI projects
(End to End projects)
- Notes
  

