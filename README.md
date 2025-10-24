\documentclass[a4paper,12pt]{article}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{caption}

\begin{document}

\begin{center}
    \includegraphics[width=60mm]{Project_ML/Figs/TUDelft_logo_rgb.png}

    \vspace{10mm}
    {\LARGE \textbf{Project CE\_ARR}}\\[2mm]
    {\large \textbf{Final Report}}\\[6mm]
    {\large \textbf{EE4C12 Machine Learning for Electrical Engineering Applications}}\\[2mm]
    \textbf{Azat Idayatov, aidayatov@tudelft.nl (6551505)}\\
    \textbf{Giorgio Recchilongo, grecchilongo@tudelft.nl (6549632)}
\end{center}

\vspace{8mm}

\section*{Summary}

This project focused on classifying various types of cardiac arrhythmias from ECG data using machine learning. Several model architectures were evaluated, including Multi-Layer Perceptrons (MLPs), Support Vector Machines (SVMs), Logistic Regression, and tree-based ensembles such as Random Forest and LightGBM. To address the strong class imbalance in the data, we applied class weighting and SMOTE oversampling techniques. After thorough hyperparameter tuning, the LightGBM model demonstrated the best performance, achieving a macro-F1 score of 0.76, a macro-accuracy of 0.98, and a macro-recall of 0.75 on the test set. These results highlight the effectiveness of gradient boosting methods for unbalanced medical classification problems.

\section*{Machine Learning Pipeline}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{Project_ML/Figs/pipeline.png}
    \caption{Machine learning pipeline for arrhythmia classification.}
    \label{fig:pipeline}
\end{figure}

Figure~\ref{fig:pipeline} illustrates the workflow adopted in this project. The initial stage, \emph{Data Preprocessing}, involved ECG acquisition and QRS peak detection using the Pan-Tompkins algorithm. Individual heartbeats were segmented, and the dataset was split into training, validation, and test partitions. To avoid data leakage, standard normalization was fitted on the training data and then applied to all samples.

Next, during \emph{Feature Design}, categorical target labels were one-hot encoded to prevent bias. The \emph{Model Exploration} phase involved training baseline models and evaluating their performance on the validation set, guiding the selection of promising candidates for further refinement. Selected models then underwent hyperparameter tuning using cross-validation. Finally, the best model was trained on the complete training data and assessed on the unseen test set.

\section*{Task 1: Model Selection}

We evaluated a range of models for arrhythmia classification:

\begin{itemize}
    \item \textbf{Linear Models (Logistic Regression, LinearSVC)} were used as baselines to gauge whether the task was linearly separable and to provide a computationally efficient benchmark.
    \item \textbf{Non-Linear SVM (RBF kernel)} was included to explore the benefit of non-linear decision boundaries, despite higher computational cost.
    \item \textbf{Tree-Based Ensembles (Random Forest, LightGBM)} were considered for their robustness and efficiency, particularly when dealing with large and complex datasets.
    \item \textbf{Multi-Layer Perceptron (MLP)} models were explored, with various techniques applied to address class imbalance, including class weighting, SMOTE, and undersampling.
\end{itemize}

\subsection*{Performance Metrics}

Given the medical context, minimizing false negatives was prioritized. Misclassifying a pathological sample as healthy presents a greater clinical risk than a false positive. Therefore, the F1 score was selected as the principal metric, as it balances precision and recall. When F1 scores were similar, recall was used as a tiebreaker.

\subsection*{Class Imbalance Handling}

The dataset was highly imbalanced, with the majority class (no disease) vastly outnumbering others (see Figure~\ref{fig:train-dis}). To ensure fair evaluation, all metrics were macro-averaged, giving equal importance to each class.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{Project_ML/Figs/train-set-distribution.png}
    \caption{Class distribution in the training set.}
    \label{fig:train-dis}
\end{figure}

\subsection*{Results}

Table~\ref{tab:model_comparison} presents validation performance for the baseline models.

\begin{table}[H]
    \centering
    \caption{Validation performance of baseline models (macro-averaged).}
    \label{tab:model_comparison}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Model} & \textbf{Recall (Macro)} & \textbf{F1 (Macro)} \\
        \midrule
        Simple MLP & 0.37 & 0.38 \\
        Weighted MLP & 0.67 & 0.30 \\
        SMOTE MLP & 0.62 & 0.45 \\
        Downsampled MLP & 0.76 & 0.52 \\
        Random Forest & 0.73 & 0.57 \\
        LightGBM & 0.68 & 0.72 \\
        SVC (RBF) & 0.86 & 0.61 \\
        Downsampled SVC (RBF) & 0.79 & 0.49 \\
        SVC (Linear) & 0.59 & 0.30 \\
        Logistic Regression & 0.69 & 0.35 \\
        \bottomrule
    \end{tabular}
\end{table}

Linear models exhibited poor performance, suggesting the task is not linearly separable. Non-linear SVMs and tree-based models fared better, with LightGBM delivering the highest F1 score with low computation time. Among MLP variants, random undersampling of the majority class yielded the best results.

Figures~\ref{fig:conf-simp-mlp} and~\ref{fig:conf-rand-for} show confusion matrices for the simple MLP and Random Forest, respectively.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{Project_ML/Figs/conf-mat-simp-mlp.png}
    \caption{Confusion matrix: simple MLP model.}
    \label{fig:conf-simp-mlp}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{Project_ML/Figs/conf-mat-rand-for.png}
    \caption{Confusion matrix: Random Forest model.}
    \label{fig:conf-rand-for}
\end{figure}

\section*{Task 2: Optimization and Final Results}

The top-performing models (Downsampled MLP, Random Forest, LightGBM) were selected for hyperparameter optimization via \texttt{RandomizedSearchCV}. Table~\ref{tab:model_results} summarizes their post-optimization performance:

\begin{table}[H]
    \centering
    \caption{Validation set metrics after optimization. (M: macro-averaged, W: weighted)}
    \label{tab:model_results}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Model} & \textbf{Accuracy} & \textbf{Recall (M)} & \textbf{F1 (M)} & \textbf{Recall (W)} & \textbf{F1 (W)} \\
        \midrule
        Downsampled MLP & 0.90 & 0.76 & 0.50 & 0.85 & 0.88 \\ 
        Random Forest & 0.97 & 0.61 & 0.66 & 0.98 & 0.97 \\
        LightGBM & 0.98 & 0.71 & 0.72 & 0.98 & 0.98 \\
        \bottomrule
    \end{tabular}
\end{table}

LightGBM was ultimately chosen as the final model due to its superior macro F1 and recall. The test set results are shown in Table~\ref{tab:final_metrics}:

\begin{table}[H]
    \centering
    \caption{Final metrics: optimized LightGBM on test set.}
    \label{tab:final_metrics}
    \begin{tabular}{lc}
        \toprule
        \textbf{Metric} & \textbf{Score} \\
        \midrule
        Accuracy & 0.9796 \\
        Recall (Macro) & 0.7492 \\
        Recall (Weighted) & 0.9796 \\
        F1 Score (Macro) & 0.7645 \\
        F1 Score (Weighted) & 0.9794 \\
        Precision (Macro) & 0.7958 \\
        Precision (Weighted) & 0.9794 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{Project_ML/Figs/conf-mat-test.png}
    \caption{Confusion matrix: LightGBM model on test set.}
    \label{fig:conf_mat_test}
\end{figure}

\section*{Conclusion}

Through systematic exploration and optimization, we found that advanced tree-based ensembles, particularly LightGBM, outperformed other models for arrhythmia classification from ECG data. The main challenge was severe class imbalance, which we managed by combining class weighting, synthetic oversampling, and careful metric selection. The final LightGBM model achieved a macro F1 of 0.76 and high accuracy, demonstrating the value of gradient boosting for complex, imbalanced medical datasets.

\end{document}
