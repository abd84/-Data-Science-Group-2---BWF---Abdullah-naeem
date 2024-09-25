# Recipe Recommendation System

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Recommendation by Ingredients](#recommendation-by-ingredients)
  - [Recommendation by Recipe Name](#recommendation-by-recipe-name)
  - [Recommendation by Nutrition](#recommendation-by-nutrition)
- [Acknowledgments](#acknowledgments)

## Overview

The **Recipe Recommendation System** is an intelligent web application that assists users in finding recipes based on their preferences. It utilizes Apache Spark's powerful data processing capabilities to recommend recipes based on various inputs: ingredients, recipe names, or nutritional values. The system is designed for flexibility and scalability, making it suitable for handling large datasets efficiently.

## Features

- **Search by Ingredients**: Input ingredients to get a list of recipes that utilize those ingredients.
- **Search by Recipe Name**: Enter a recipe name to find similar recipes based on their ingredients.
- **Search by Nutrition**: Filter recipes based on specific nutritional requirements (fat, protein, carbohydrates).
- **Full Recipe Details**: Each recommendation includes detailed information such as ingredients, nutritional values, cooking directions, and preparation time.
- **User-Friendly Interface**: A simple web interface that allows users to easily navigate through options and view recommendations.

## Technologies Used

- **Python**: The main programming language used for backend development.
- **Apache Spark**: Utilized for big data processing, enabling fast and scalable computations.
- **Streamlit**: A lightweight web framework that provides the backbone for the web application.
- **HTML/CSS**: Used for the frontend interface design.
- **CSV**: The dataset is stored in a CSV format, making it easy to read and manipulate.

## Dataset

The dataset used for this project is a CSV file named `df_cleaned.csv`, which contains various details about recipes. Key columns in the dataset include:

- **recipe_name**: The name of the recipe.
- **ingredients**: A list of ingredients required for the recipe.
- **nutrition**: Nutritional information, including total fat, protein, and carbohydrates (in grams).
- **directions**: Step-by-step instructions for preparing the recipe.
- **timing**: The time required to prepare and cook the recipe.

**Note**: Make sure to place the dataset file in the root directory of the project.

## Architecture

The architecture of the application is structured as follows:

1. **Data Ingestion**: The application reads the recipe data from a CSV file into a Spark DataFrame.
2. **Data Processing**:
   - **Tokenization**: Ingredients are tokenized into words for better analysis.
   - **TF-IDF Transformation**: The Term Frequency-Inverse Document Frequency (TF-IDF) method is applied to quantify the importance of each ingredient in the dataset.
3. **Recommendation Algorithms**:
   - **Centered Cosine Similarity**: Utilized for calculating similarities between recipes based on user input.
4. **Web Application**: The Streamlit web server handles user requests and serves HTML templates to display the results.

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- Apache Spark (with PySpark installed)
- Streamlit
- Other required libraries (see `requirements.txt` for the full list)

# How It Works

## Recommendation by Ingredients
When users enter ingredients, the following steps are executed:

### Tokenization: The ingredients are tokenized into individual words for analysis.
### TF-IDF Transformation: 
The tokenized words are transformed into a TF-IDF representation, which helps determine the importance of each ingredient in the context of the entire dataset.
### Centered Cosine Similarity Calculation: 
The application calculates the centered cosine similarity between the user's input and each recipe in the dataset.
### Display Recommendations: 
Recipes with a similarity score above a defined threshold are displayed, along with their full details.

# Recommendation by Recipe Name

## For searching by recipe name:
The application filters the dataset for recipes that contain the input recipe name.
The ingredients of the found recipe are processed similarly to the ingredient-based recommendation.
Similar recipes are identified using centered cosine similarity, and recommendations are displayed.

# Recommendation by Nutrition
## For nutritional recommendations:

The user inputs desired nutritional values for fat, protein, and carbohydrates.
The application filters recipes based on these values, allowing a small margin for variability.
Recommended recipes are displayed with their full details.

# Acknowledgments

Special thanks to the open-source community for the tools and libraries used in this project.
Thanks to the dataset providers for making recipe data available.
