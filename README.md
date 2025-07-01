# Simple Recipe Web Application

This is a simple web application built with Flask that allows users to store and view recipes. Recipes are saved in a local SQLite database.

## Features

- Add new recipes with name, ingredients, instructions, and an optional image URL.
- View a list of all saved recipes.
- View the details of a specific recipe.

## Project Structure

```
/
|-- app.py                  # Main Flask application, defines routes and database models
|-- recipes.db              # SQLite database file (created automatically)
|-- requirements.txt        # Python dependencies
|-- static/
|   |-- style.css           # Basic CSS for styling
|-- templates/
|   |-- base.html           # Base HTML template with navigation
|   |-- index.html          # Displays all recipes
|   |-- add_recipe.html     # Form to add a new recipe
|   |-- recipe_detail.html  # Displays details of a single recipe
|-- README.md               # This file
```

## Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # If this were a git repo:
    # git clone <repository_url>
    # cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will start, and the database file (`recipes.db`) will be created automatically if it doesn't exist.

5.  **Open your web browser and navigate to:**
    `http://127.0.0.1:5000/`

## How to Use

-   Navigate to the homepage to see all recipes.
-   Click on "Add Recipe" in the navigation bar to go to the form for adding a new recipe.
-   Fill in the recipe details and submit the form.
-   Click on a recipe name from the homepage to view its details.

## Future Enhancements (Ideas)

-   Edit existing recipes.
-   Delete recipes.
-   User authentication to manage personal recipes.
-   Search and filter recipes.
-   Categorize recipes (e.g., by cuisine, meal type).
-   Allow uploading recipe images instead of just URLs.
