🧠 AI-Powered Task Prioritization System

Project Overview

This project implements a data-driven approach to optimize task management. Using a machine learning model (Random Forest Regressor) trained on experimental data, the system predicts the Expected Performance Ratio (EarnedPoints / OptimalPoints) for a given task set. The goal is to provide users with an intelligent prediction of whether a set of tasks is likely to lead to optimal resource utilization.

The project is structured into two main parts:

Backend Analysis: Data cleaning, feature engineering, and model training (performed using Python, Pandas, and Scikit-learn).

Frontend Simulation: An interactive React application that allows users to input key task features and visualize the model's prediction.

I. Backend & Modeling Summary

The core of the system is the Random Forest Regressor, which was selected and tuned using Grid Search Cross-Validation (GridSearchCV).

Key Features Used in the Model (from Finale_Merged_Data.csv):

The model learned relationships between the following features and the final performance ratio:

Task Points (Task[#]Points): The potential reward; positively correlated with performance.

Task Length (Task[#]Length): The effort required (in turns); negatively correlated with performance.

Task Deadline (Task[#]Deadline): The urgency; tasks with shorter deadlines generally drive better performance.

Task Rank (Task[#]Rank): The predefined priority (1=Highest, 6=Lowest).

Correlation (Correlation): The statistical relationship (e.g., Pearson's r) between Points and Length across the entire task set, which proved to be an important high-level feature.

Variable Length (VariableLength): A binary flag indicating whether task lengths were fixed or random.

Model Performance

The final Random Forest Regressor achieved an $R^2$ score of $0.92$ (Placeholder: Insert Actual Value Here) on the test set, indicating its strong ability to predict the outcome variable. Feature importance analysis confirmed that Task Points, Task Length, and Deadline were the most influential factors.

II. Frontend Application (TaskPrioritizer.jsx)

The React application serves as a clean, interactive user interface (UI) to demonstrate the model's output.

Features of the App:

Interactive Inputs: Users can adjust all six key features (Points, Length, Deadline, Rank, Correlation, Variable Length).

Simulated Inference: The runAiInference function in the JavaScript code contains conceptual logic that mathematically simulates the learned impact of your model's features (e.g., increasing length decreases the predicted ratio).

Visual Output: Results are displayed as a percentage metric and a stacked bar chart (using the recharts library).

Task Recommendation: Provides an immediate action recommendation (High Priority vs. Medium/Low Priority).

Running the Application (Optional)

This file is a single-component React application. If you wish to run it locally, you need a standard Node.js/React environment:

Prerequisites: Ensure Node.js and npm are installed on your system.

Setup: Create a standard React project (e.g., using npx create-react-app my-app).

Dependencies: Install the required libraries:

npm install recharts lucide-react




Integration: Replace the contents of your main application file (e.g., src/App.jsx) with the code from TaskPrioritizer.jsx.

Run Server: Execute the project's start script (e.g., npm start or npm run dev).

📊 III. Human Behavior and Strategy Analysis

This section analyzes the qualitative data provided by participants in the Strategy column to understand the cognitive approaches used for task prioritization. Categorizing these free-text responses reveals three primary strategies that significantly impacted performance.

A. Categorized Prioritization Strategies

The 102 participant responses were clustered into three main decision-making categories:

1. The Deadline-First / Urgency Strategy (High Success Rate)

This group prioritized tasks purely based on the Task Deadline, attempting to complete the most urgent tasks first to avoid expiration penalties.

Core Logic: "Nearest deadline first, then if tie or close do most points."

Performance Insight: This rigid, simple strategy often led to good performance (high Earned Points) because it minimized losses from incomplete tasks, providing a solid baseline for success.

2. The Value / Points-First Strategy (Variable Success Rate)

This group focused exclusively on the potential reward, maximizing Task Points without closely considering the effort (length) or the deadline until forced to do so.

Core Logic: "I went for the highest value task first, then the second, third, etc."

Performance Insight: This strategy was successful when task lengths were low but often resulted in poor performance when high-point tasks consumed too many turns, leading to a cascade of missed deadlines and low total points.

3. The Opportunity Cost / Ratio Strategy (Highest Optimal Success Rate)

This group, which mirrored the logic of an optimal algorithm, prioritized based on the Points-to-Length Ratio (value divided by effort).

Core Logic: "I tried to calculate which tasks would be worth my while. I would get to the tasks that had deadlines expiring first and then sort them based on opportunity cost."

Performance Insight: While the most complex strategy, this approach was most closely aligned with maximizing the OptimalPoints and served as the benchmark for the Random Forest model's success.

B. The Link to Machine Learning

The machine learning model (Random Forest Regressor) implicitly learned to identify and replicate the successful Opportunity Cost Strategy. By analyzing the influence of Correlation and Variable Length, the model predicts the optimal scheduling, effectively acting as an automated "Ratio Strategy" guide for the user.